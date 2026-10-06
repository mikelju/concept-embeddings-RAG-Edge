#!/usr/bin/env bash
# Phase 06 G-A1 pod session (plan increment 5; recipe docs/plans/fase-06-rivals/ga1_recipe.md):
# the G-L HotpotQA index rebuilt with Phase 02 D17's settings, then Search-R1 through the Phase 02
# driver, unchanged, on the preregistered 1,000 qids, searching that index through the Phase 02
# retrieval server on the same GPU. Run from the repository root at the frozen commit, with
# data/phase06/ga1/ uploaded and the old HotpotQA data under $OLD_DATA_ROOT/phase9, e.g.
#   POD_COST_PER_HR=1.59 SPENT_S=900 OLD_DATA_ROOT=/workspace/old-data \
#     nohup bash scripts/pod_ga1.sh > /workspace/pod_ga1.out 2>&1 &
# Progress: `STAGE_DONE <name>` (or STAGE_SKIP / STAGE_FAIL) and `PACE` lines in $LOG; a sampler
# line every 30 s in $SAMPLES. A non-empty C3 diff, a spend or a pace projection past the cut
# (`STOP_CUT`) stops the session.
set -euo pipefail

: "${POD_COST_PER_HR:?set POD_COST_PER_HR to the pod costPerHr}"
: "${OLD_DATA_ROOT:?set OLD_DATA_ROOT to the uploaded old data root (holds phase9/)}"
CUT_USD="${CUT_USD:-6.9}"                    # spec: hard cut for G-A1, 1.5 x its projection
SPENT_S="${SPENT_S:-0}"                      # pod seconds used before this script started
GL_SEARCH_QUESTIONS="${GL_SEARCH_QUESTIONS:-100}"  # the build's own search; empty = all 7,405
# Deviation 06.4: 23000 covers the idle server (21,105 MiB) plus its search peak (7.27 GiB OOM at 6144).
UTIL_MARGIN_MIB="${UTIL_MARGIN_MIB:-23000}"  # GPU memory left free beside vLLM and the server
LOG="${LOG:-/workspace/pod_ga1.log}"
PROGRESS="${PROGRESS:-/workspace/pod_ga1.progress}"
SAMPLES="${SAMPLES:-/workspace/pod_ga1.samples}"
PACE=/workspace/pod_ga1.pace
STAGE_FILE=/workspace/pod_ga1.stage
SET=hotpotqa-dev
DRIVER_COMMIT=33c0668bab40adaadfe716426f80a0d6cc98273d  # Phase 02 session 4: G-A1 ran here
INDEX_COMMIT=f3f499e1c4b519b00d675fee1834c78cd31707d8   # Phase 02 session 6: D17 HotpotQA build
QIDS_SHA256=6cebd41ff9e5e9f02e27ebb019620de0e37b77794fc6a783a2ff0d80fb3a62bd
BUNDLE=data/phase06/ga1
OUT=$BUNDLE/pod
P2=data/phase02  # the Phase 02 modules write here (common.PHASE_DIR); scratch on the pod
COLBERT_ENV=src/edge_rag/pod/colbert_env
START=$(date +%s)
export HF_HOME=/workspace/hf POD_COST_PER_HR OLD_DATA_ROOT
export VLLM_USE_FLASHINFER_SAMPLER=0  # Phase 02 D13: greedy takes the argmax either way

main_py() { uv run --frozen --group pod python "$@"; }
colbert_py() { PYTHONPATH="$PWD/src" uv run --frozen --project "$COLBERT_ENV" python "$@"; }

stage() {  # stage <name> <marker file> <command...>
  local name=$1 marker=$2
  shift 2
  if [ -n "$marker" ] && [ -f "$marker" ]; then
    echo "STAGE_SKIP $name" >> "$LOG"
    return 0
  fi
  echo "$name" > "$STAGE_FILE"
  echo "STAGE_START $name $(date -u +%FT%TZ)" >> "$LOG"
  if "$@"; then
    echo "STAGE_DONE $name $(date -u +%FT%TZ)" >> "$LOG"
  else
    echo "STAGE_FAIL $name $(date -u +%FT%TZ)" >> "$LOG"
    exit 1
  fi
}

sampler() {
  while true; do
    echo "$(date -u +%FT%TZ) | $(tail -n 1 "$PROGRESS" 2> /dev/null | tr '\r' ' ' | tail -c 200) | gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2> /dev/null) | current $(cat /sys/fs/cgroup/memory.current 2> /dev/null) peak $(cat /sys/fs/cgroup/memory.peak 2> /dev/null) | disk $(df -B1G --output=used /workspace 2> /dev/null | tail -n 1)" >> "$SAMPLES"
    sleep 30
  done
}

watchdog() {  # the hard cut on spend, and the pace projection of the gated stages
  local main=$1 stage units
  while true; do
    sleep 60
    stage=$(cat "$STAGE_FILE" 2> /dev/null || echo none)
    if [ "$stage" = g-l ]; then
      units=$(grep -o '[0-9]* units encoded' "$PROGRESS" 2> /dev/null | tail -n 1 | cut -d' ' -f1 || true)
      [ -n "$units" ] && echo "$(date +%s) $units" >> "$PACE"
    fi
    if ! python3 src/edge_rag/pod/ga1_pace.py --stage "$stage" --progress "$PROGRESS" --pace "$PACE" \
      --spent "$((SPENT_S + $(date +%s) - START))" --rate "$POD_COST_PER_HR" --cut "$CUT_USD" >> "$LOG"; then
      echo "STOP_CUT $stage $(date -u +%FT%TZ)" >> "$LOG"
      pkill -f "[e]dge_rag.pod" || true
      pkill -f "[V]LLM" || true
      kill "$main" 2> /dev/null || true
      return 0
    fi
  done
}

checks() {  # uploads against the laptop's digests
  (cd "$BUNDLE" && sha256sum -c upload.sha256) || return 1
  (cd "$OLD_DATA_ROOT" && sha256sum -c "$OLDPWD/$BUNDLE/old.sha256") || return 1
}

c3_diff() {  # spec C3: the driver against Phase 02's G-A1 commit, the index path against D17's
  git diff "$DRIVER_COMMIT" -- src/edge_rag/pod/searchr1.py > "$OUT/c3-g-a1.diff"
  git diff "$INDEX_COMMIT" -- src/edge_rag/pod/colbert.py "$COLBERT_ENV" >> "$OUT/c3-g-a1.diff"
  git status --porcelain --untracked-files=all > "$OUT/c3-g-a1.status"
  git rev-parse HEAD > "$OUT/commit.txt"
  [ ! -s "$OUT/c3-g-a1.diff" ]
}

sync_envs() {  # as Phase 02's pod_run.sh
  uv sync --frozen --group pod || return 1
  uv sync --frozen --project "$COLBERT_ENV" || return 1
  local site
  site="$(main_py -c 'import site; print(site.getsitepackages()[0])')" || return 1
  echo "$site/nvidia/cu13/lib" > /workspace/.cu13_path
}

cuda_check() {  # a real kernel in both environments, not only `is_available()`
  local code='import torch; x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item(); torch.cuda.synchronize(); print(torch.__version__, torch.cuda.get_device_name(0), y)'
  main_py -c "$code" || return 1
  colbert_py -c "$code" || return 1
}

gl_build() {  # Phase 02 D17: one create call (pod_run.sh's hotpotqa-dev arguments)
  local limit=()
  [ -n "$GL_SEARCH_QUESTIONS" ] && limit=(--limit-questions "$GL_SEARCH_QUESTIONS")
  colbert_py -m edge_rag.pod.colbert --set "$SET" --chunk-units 6000000 "${limit[@]}" 2>&1 \
    | tee -a "$PROGRESS"
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
  cp "$P2/rankings/$SET/g-l.manifest.json" "$P2/rankings/$SET/g-l.jsonl.gz" "$OUT/"
  du -sb "$P2/index/$SET" > "$OUT/index.du"
  (cd "$P2/index/$SET" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$OUT/index.sha256"
}

ga1() {  # the Phase 02 retrieval server on the GPU, then the driver on the 1,000 qids
  colbert_py -m edge_rag.pod.colbert --set "$SET" --serve --port 8000 > /workspace/serve.out 2>&1 &
  local server=$! status=0 total used util
  until curl -sf http://127.0.0.1:8000/ > /dev/null; do
    kill -0 "$server" 2> /dev/null || { echo "retrieval server died" >&2; return 1; }
    sleep 10
  done
  # vLLM's share: the driver's 0.80 when the card has room beside the server, else what is free
  read -r total used <<< "$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ' | tr ',' ' ')"
  util=$(python3 -c "print(min(0.80, int(($total - $used - $UTIL_MARGIN_MIB) / $total * 100) / 100))")
  echo "GPU total_mib $total server_used_mib $used vllm_util $util" | tee -a "$LOG" > "$OUT/gpu_split.txt"
  python3 -c "import sys; sys.exit(0 if $util >= 0.30 else 1)" \
    || { echo "server leaves too little GPU memory for vLLM" >&2; kill "$server"; return 1; }
  main_py -m edge_rag.pod.ga1_subset --qids "$BUNDLE/hotpotqa-dev-1000.txt" \
    --qids-sha256 "$QIDS_SHA256" --gpu-memory-utilization "$util" 2>&1 | tee -a "$PROGRESS"
  status=${PIPESTATUS[0]}
  kill "$server" 2> /dev/null || true
  wait "$server" || true
  pkill -f "[e]dge_rag.pod.colbert --set $SET --serve" || true
  [ "$status" = 0 ] || return 1
  cp "$P2/rankings/$SET/g-a1.jsonl.gz" "$P2/rankings/$SET/g-a1.manifest.json" \
    "$P2/searchr1/$SET/g-a1/trace.jsonl.gz" "$P2/searchr1/$SET/g-a1/subset.json" "$OUT/"
}

sums() {
  cp "$LOG" "$PROGRESS" "$SAMPLES" /workspace/serve.out "$OUT/" 2> /dev/null || true
  (cd "$OUT" && find . -maxdepth 1 -type f ! -name 'sha256sums.*' -printf '%P\0' | sort -z \
    | xargs -0 -r sha256sum > sha256sums.tmp && mv sha256sums.tmp sha256sums.txt)
}

mkdir -p "$OUT"
sampler &
SAMPLER=$!
watchdog $$ &
WATCHDOG=$!
trap 'kill "$SAMPLER" "$WATCHDOG" 2> /dev/null || true; sums' EXIT

stage checks "" checks
stage c3-diff "" c3_diff
stage sync "" sync_envs
export LD_LIBRARY_PATH="$(cat /workspace/.cu13_path):${LD_LIBRARY_PATH:-}"
stage cuda "" cuda_check
stage g-l "$OUT/index.sha256" gl_build
stage g-a1 "$OUT/subset.json" ga1
echo "STAGE_DONE all $(date -u +%FT%TZ)" >> "$LOG"
