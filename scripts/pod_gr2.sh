#!/usr/bin/env bash
# Phase 06 G-R2 pod session (plan increment 3): C3 check, then Qwen3-Reranker-0.6B over the
# uploaded G-L top-100 pairs, set by set. Run from the repository root at the frozen commit,
# with data/phase06/pairs/ uploaded, e.g.
#   POD_COST_PER_HR=0.74 nohup bash scripts/pod_gr2.sh > /workspace/pod_gr2.out 2>&1 &
# Progress: `STAGE_DONE <name>` (or STAGE_SKIP / STAGE_FAIL) lines in $LOG; a sampler line every
# 30 s in $SAMPLES (UTC time, last progress line, cgroup memory). No corpus is read.
# A failed C3 check stops the session before any scoring (spec: G-R2 aborted, no retry).
set -euo pipefail

: "${POD_COST_PER_HR:?set POD_COST_PER_HR to the pod costPerHr}"
SETS="${SETS:-musique multihop-rag hotpotqa-dev}"
LOG="${LOG:-/workspace/pod_gr2.log}"
SAMPLES="${SAMPLES:-/workspace/pod_gr2.samples}"
PROGRESS="${PROGRESS:-/workspace/pod_gr2.progress}"
export POD_COST_PER_HR
OUT=data/phase06

main_py() { uv run --frozen --group pod python "$@"; }

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

cuda_check() {  # a real kernel, not only `is_available()`
  main_py -c 'import torch; x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item(); torch.cuda.synchronize(); print(torch.__version__, torch.cuda.get_device_name(0), y)'
}

sampler() {  # every 30 s: UTC time, last progress line, cgroup v2 memory.current and memory.peak
  while true; do
    echo "$(date -u +%FT%TZ) | $(tail -n 1 "$PROGRESS" 2> /dev/null) | current $(cat /sys/fs/cgroup/memory.current 2> /dev/null) peak $(cat /sys/fs/cgroup/memory.peak 2> /dev/null)" >> "$SAMPLES"
    sleep 30
  done
}

sums() {  # sha256sums.txt over every checks and scores file so far
  (cd "$OUT" && find checks scores -type f ! -name 'sha256sums.*' -print0 2> /dev/null | sort -z | xargs -0 -r sha256sum > sha256sums.tmp && mv sha256sums.tmp sha256sums.txt)
}

mkdir -p "$OUT/scores" "$OUT/checks"
sampler &
SAMPLER=$!
trap 'kill "$SAMPLER" 2> /dev/null || true' EXIT

(cd "$OUT/pairs" && sha256sum -c /workspace/pairs.sha256)
stage sync "" uv sync --frozen --group pod
stage cuda "" cuda_check
export -f main_py
stage c3-check "$OUT/checks/c3-g-r2.json" \
  bash -c 'set -o pipefail; main_py -m edge_rag.pod.qwen_rerank --check 2>&1 | tee -a "$1"' _ "$PROGRESS"
sums
for set in $SETS; do
  stage "score-$set" "$OUT/scores/$set.manifest.json" \
    bash -c 'set -o pipefail; main_py -m edge_rag.pod.qwen_rerank --score "$1" 2>&1 | tee -a "$2"' _ "$set" "$PROGRESS"
  sums
done

echo "STAGE_DONE all" >> "$LOG"
