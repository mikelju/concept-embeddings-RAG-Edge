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

pending() {  # a QASPER runner not written yet (plan increment 5b): fails, so the item is not run
  echo "[ERROR] item $1: QASPER runner not written yet (plan increment 5b)" >&2
  return 1
}

item_g-l() { pending g-l; }
item_c3() { pending c3; }
item_j-strong() { pending j-strong; }
item_g-r2() { pending g-r2; }
item_j-rrf3() { pending j-rrf3; }
item_within-j-strong() { pending within-j-strong; }
item_g-a1() { pending g-a1; }
