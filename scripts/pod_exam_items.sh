#!/usr/bin/env bash
# Item commands for scripts/pod_exam.sh (sourced): `item_<name> <limit> <outdir>` writes the
# item's outputs into <outdir> (limit empty = full run, else the probe size) and
# `item_total <name>` prints its full size (units or questions) for the probe projection.
# GLiNER runs with the pins of scripts/gliner-requirements.txt layered over the project env
# (`uv run --with-requirements`), not as a project dependency: gliner 0.2.29 needs protobuf to
# convert DeBERTa-v3's spm.model, else transformers 5.16.1 falls back to tiktoken and fails
# (measured 2026-10-06 on the laptop; the old project's pod lock had protobuf 7.36.2).
SPLIT="${SPLIT:-test}"
SPLIT_DIR="data/phase07/$SPLIT"
GLINER_MODEL_DIR="${GLINER_MODEL_DIR:-/workspace/gliner-model}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

main_py() { uv run --frozen --group pod python "$@"; }

gliner_py() {
  uv run --frozen --group pod --with-requirements scripts/gliner-requirements.txt \
    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match python "$@"
}

item_total() {  # units for corpus passes, questions otherwise (lines of the split's files)
  case "$1" in
    gliner|g-l) wc -l < "$SPLIT_DIR/units.jsonl" ;;
    *) wc -l < "$SPLIT_DIR/questions.jsonl" ;;
  esac
}

item_setup() {
  uv sync --frozen --group pod
  main_py -c 'import torch; x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item(); torch.cuda.synchronize(); print(torch.__version__, torch.cuda.get_device_name(0), y)' \
    | tee "$2/cuda.txt"
}

item_gliner() {
  local limit=()
  [ -n "$1" ] && limit=(--limit-units "$1")
  gliner_py -m edge_rag.cli qasper-gliner "$SPLIT" --out "$2" --model-dir "$GLINER_MODEL_DIR" \
    "${limit[@]}"
}

# QASPER runners (plan increment 5c): thin adapters in src/edge_rag/pod/exam_qasper.py over
# the Phase 02-06 pod modules. Inputs uploaded before the pod starts: the split's units and
# questions, the laptop's pooled Dense, BM25 and hop lists under $LAPTOP_DIR/pooled (increment
# 8), the old Phase 17 pairs under $OLD17_DIR and the Phase 06 pairs files (C3 checks).
LAPTOP_DIR="${LAPTOP_DIR:-data/phase07/$SPLIT/laptop}"
OLD17_DIR="${OLD17_DIR:-/workspace/old17}"
GL_INDEX="${GL_INDEX:-/workspace/qasper-gl-index}"
COLBERT_ENV=src/edge_rag/pod/colbert_env
UTIL_MARGIN_MIB="${UTIL_MARGIN_MIB:-23000}"  # as scripts/pod_ga1.sh (deviation 06.4)

colbert_py() { PYTHONPATH="$PWD/src" uv run --frozen --project "$COLBERT_ENV" python "$@"; }
exam_py() { main_py -m edge_rag.pod.exam_qasper "$@"; }

limit_q() { [ -n "$1" ] && echo "--limit-questions $1"; }

pooled_inputs() {  # --first name=path for G-L (this pod) and the laptop's pooled lists
  echo "--first g-l=$OUT/g-l/g-l.jsonl.gz"
  for name in dense bm25 hop; do echo "--first $name=$LAPTOP_DIR/pooled/$name.jsonl.gz"; done
}

item_g-l() {
  local limit=()
  [ -n "$1" ] && limit=(--limit-units "$1")
  colbert_py -m edge_rag.pod.exam_qasper g-l "$SPLIT_DIR" "$2" --index "$GL_INDEX-item" "${limit[@]}"
  rm -rf "$GL_INDEX-item"
}

item_c3() {  # J-strong rescoring of old Phase 17 pairs, G-R2 fidelity and determinism
  main_py -m edge_rag.pod.rerank --check-old --old17-dir "$OLD17_DIR"
  main_py -m edge_rag.pod.qwen_rerank --check
  cp data/phase02/checks/c3-strong-rescore.json data/phase06/checks/c3-g-r2.json "$2/"
}

# shellcheck disable=SC2046
item_j-strong() { exam_py rerank "$SPLIT_DIR" "$2" $(pooled_inputs) $(limit_q "$1") --systems g-r j-rrf4; }
# shellcheck disable=SC2046
item_g-r2() { exam_py rerank "$SPLIT_DIR" "$2" $(pooled_inputs) $(limit_q "$1") --qwen --systems g-r2; }
# shellcheck disable=SC2046
item_j-rrf3() { exam_py rerank "$SPLIT_DIR" "$2" $(pooled_inputs) $(limit_q "$1") --systems j-rrf3; }
# shellcheck disable=SC2046
item_within-j-strong() { exam_py rerank "$SPLIT_DIR" "$2" $(limit_q "$1") --systems within-j-strong; }

item_g-a1() {  # A100 pod: G-L index over the split (built once), its server, then Search-R1
  if [ ! -f "$GL_INDEX.built" ]; then
    colbert_py -m edge_rag.pod.exam_qasper g-l "$SPLIT_DIR" "$GL_INDEX.ranking" --index "$GL_INDEX"
    touch "$GL_INDEX.built"
  fi
  colbert_py -m edge_rag.pod.exam_qasper serve "$SPLIT_DIR" "$2" --index "$GL_INDEX"     > "$2.serve.out" 2>&1 &
  local server=$! status=0 total used util
  until curl -sf http://127.0.0.1:8000/ > /dev/null; do
    kill -0 "$server" 2> /dev/null || { echo "retrieval server died" >&2; return 1; }
    sleep 10
  done
  read -r total used <<< "$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ' | tr ',' ' ')"
  util=$(python3 -c "print(min(0.80, int(($total - $used - $UTIL_MARGIN_MIB) / $total * 100) / 100))")
  echo "GPU total_mib $total server_used_mib $used vllm_util $util" > "$2/gpu_split.txt"
  # shellcheck disable=SC2046
  exam_py ga1 "$SPLIT_DIR" "$2" --gpu-memory-utilization "$util" $(limit_q "$1") || status=1
  kill "$server" 2> /dev/null || true
  wait "$server" || true
  return "$status"
}
