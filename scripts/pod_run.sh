#!/usr/bin/env bash
# Phase 02 pod session (plan D1): G-L, then G-R, then G-A1, resumable stage by stage.
# Run from the repository root at the frozen commit, e.g.
#   POD_COST_PER_HR=0.74 OLD_DATA_ROOT=/workspace/old-data OLD17_DIR=/workspace/old17 \
#     nohup bash scripts/pod_run.sh > /workspace/pod_run.out 2>&1 &
# Progress: `STAGE_DONE <name>` (or STAGE_SKIP / STAGE_FAIL) lines in $LOG.
# A stage is skipped when its output manifest exists.
set -euo pipefail

: "${POD_COST_PER_HR:?set POD_COST_PER_HR to the pod costPerHr}"
: "${OLD_DATA_ROOT:?set OLD_DATA_ROOT to the uploaded old data root}"
OLD17_DIR="${OLD17_DIR:-/workspace/old17}"
# Sets for G-L and G-R; a set whose G-L stage failed twice is left out (recipe stop conditions).
GL_SETS="${GL_SETS:-multihop-rag musique hotpotqa-dev}"
GR_SETS="${GR_SETS-$GL_SETS}"
GA1_SETS="${GA1_SETS-multihop-rag musique}"
LOG="${LOG:-/workspace/pod_run.log}"
export POD_COST_PER_HR OLD_DATA_ROOT
# FlashInfer's sampler JIT-builds with the image's nvcc 12.4, which rejects `--compress-mode`
# (session 3); greedy decoding takes the argmax either way (plan D13).
export VLLM_USE_FLASHINFER_SAMPLER=0
OUT=data/phase02
COLBERT_ENV=src/edge_rag/pod/colbert_env

main_py() { uv run --frozen --group pod python "$@"; }
colbert_py() { PYTHONPATH="$PWD/src" uv run --frozen --project "$COLBERT_ENV" python "$@"; }

stage() {  # stage <name> <marker file> <command...>
  local name=$1 marker=$2
  shift 2
  if [ -n "$marker" ] && [ -f "$marker" ]; then
    echo "STAGE_SKIP $name" >> "$LOG"
    return 0
  fi
  echo "STAGE_START $name $(date -u +%FT%TZ)" >> "$LOG"
  if "$@"; then
    echo "STAGE_DONE $name $(date -u +%FT%TZ)" >> "$LOG"
  else
    echo "STAGE_FAIL $name $(date -u +%FT%TZ)" >> "$LOG"
    exit 1
  fi
}

sync_envs() {
  # Functions run inside `if` lose `set -e`: every step returns on failure explicitly.
  uv sync --frozen --group pod || return 1
  uv sync --frozen --project "$COLBERT_ENV" || return 1
  # remote-gpu trap: vllm 0.30.0 is a CUDA 13 build next to torch cu126.
  local site
  site="$(main_py -c 'import site; print(site.getsitepackages()[0])')" || return 1
  echo "$site/nvidia/cu13/lib" > "$OUT/.cu13_path"
}

cuda_check() {  # a real kernel in both environments, not only `is_available()`
  local code='import torch; x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item(); torch.cuda.synchronize(); print(torch.__version__, torch.cuda.get_device_name(0), y)'
  main_py -c "$code" || return 1
  colbert_py -c "$code" || return 1
}

serve_and_run() {  # serve_and_run <set> <command...>: G-L retrieval server around a command
  local set=$1
  shift
  # On the GPU beside vLLM's 0.8 of an 80 GB card; the CPU server was too slow for G-A1 (plan D15, F6).
  colbert_py -m edge_rag.pod.colbert --set "$set" --serve --port 8000 &
  local server=$!
  until curl -sf http://127.0.0.1:8000/ > /dev/null; do
    kill -0 "$server" 2> /dev/null || { echo "retrieval server died" >&2; return 1; }
    sleep 5
  done
  local status=0
  "$@" || status=$?
  kill "$server"
  wait "$server" || true
  # `kill` reaches the `uv run` wrapper, not its python child: end the server itself.
  pkill -f "[e]dge_rag.pod.colbert --set $set --serve" || true
  return "$status"
}

mkdir -p "$OUT"
stage sync "" sync_envs
export LD_LIBRARY_PATH="$(cat "$OUT/.cu13_path"):${LD_LIBRARY_PATH:-}"
stage cuda "" cuda_check

stage probe-g-l "$OUT/probe/units1000/musique/g-l.manifest.json" \
  colbert_py -m edge_rag.pod.colbert --set musique --limit-units 1000 --limit-questions 100

for set in $GL_SETS; do
  # fast-plaid's GPU path on every set; D12's CPU centroid update was for the 24 GB card (plan D15).
  stage "g-l-$set" "$OUT/rankings/$set/g-l.manifest.json" \
    colbert_py -m edge_rag.pod.colbert --set "$set"
done

stage g-r-check-old "$OUT/checks/c3-strong-rescore.json" \
  main_py -m edge_rag.pod.rerank --check-old --old17-dir "$OLD17_DIR"

for set in $GR_SETS; do
  stage "g-r-$set" "$OUT/rankings/$set/g-r.manifest.json" \
    main_py -m edge_rag.pod.rerank --set "$set"
done

stage c4-compare "$OUT/checks/c4-infer-vs-vllm-musique.json" \
  serve_and_run musique main_py -m edge_rag.pod.searchr1 --set musique --compare-infer 50

for set in $GA1_SETS; do
  stage "g-a1-$set" "$OUT/rankings/$set/g-a1.manifest.json" \
    serve_and_run "$set" main_py -m edge_rag.pod.searchr1 --set "$set"
done

# Digests of every output except the PLAID indexes (rebuilt, not downloaded).
(cd "$OUT" && find . -type f ! -path './index/*' ! -name sha256sums.txt ! -name .cu13_path \
  -print0 | sort -z | xargs -0 sha256sum > sha256sums.txt)
echo "STAGE_DONE sha256sums $(date -u +%FT%TZ)" >> "$LOG"
echo "STAGE_DONE all" >> "$LOG"
