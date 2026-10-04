#!/usr/bin/env bash
# Phase 04 pod session (plan D6): J-strong over the uploaded uncached pairs, set by set.
# Run from the repository root at the frozen commit, with data/phase04/pairs/ uploaded, e.g.
#   POD_COST_PER_HR=0.74 nohup bash scripts/pod_judge.sh > /workspace/pod_judge.out 2>&1 &
# Progress: `STAGE_DONE <name>` (or STAGE_SKIP / STAGE_FAIL) lines in $LOG; a sampler line every
# 30 s in $SAMPLES (UTC time, last progress line, cgroup memory). No corpus is read.
# Per set, `rerank --pairs` checks C4 on the timing sample before scoring and exits non-zero on
# a failed check, which stops the session. A stage is skipped when its scores manifest exists.
set -euo pipefail

: "${POD_COST_PER_HR:?set POD_COST_PER_HR to the pod costPerHr}"
SETS="${SETS:-musique multihop-rag hotpotqa-dev}"
LOG="${LOG:-/workspace/pod_judge.log}"
SAMPLES="${SAMPLES:-/workspace/pod_judge.samples}"
PROGRESS="${PROGRESS:-/workspace/pod_judge.progress}"
export POD_COST_PER_HR
OUT=data/phase04/scores

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

mkdir -p "$OUT"
sampler &
SAMPLER=$!
trap 'kill "$SAMPLER" 2> /dev/null || true' EXIT

stage sync "" uv sync --frozen --group pod
stage cuda "" cuda_check

sums() {  # rewrite sha256sums.txt over every scores file so far, so a finished set is hashed at once
  (cd "$OUT" && find . -type f ! -name 'sha256sums.*' -print0 | sort -z | xargs -0 -r sha256sum > sha256sums.tmp && mv sha256sums.tmp sha256sums.txt)
}

export -f main_py
for set in $SETS; do
  stage "judge-$set" "$OUT/$set.manifest.json" \
    bash -c 'set -o pipefail; main_py -m edge_rag.pod.rerank --pairs "$1" 2>&1 | tee -a "$2"' _ "$set" "$PROGRESS"
  sums
done

sums
echo "STAGE_DONE sha256sums $(date -u +%FT%TZ)" >> "$LOG"
echo "STAGE_DONE all" >> "$LOG"
