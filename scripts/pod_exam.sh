#!/usr/bin/env bash
# Phase 07 pod stage (plan increment 5b, spec "Money and stopping rule", plan D9).
# Runs the items of `python -m edge_rag.pod.exam_stage order` in that fixed order (4090: setup,
# GLiNER, G-L, C3 checks, J-strong for G-R and j-rrf4, G-R2, J-strong for j-rrf3, within-paper
# J-strong; A100: setup, G-A1). Each item with a probe size first runs its probe (first 50
# questions, or first 1,000 units for a corpus pass) into $OUT/probe/<item>/; `decide` projects
# the full run from that pace and the item runs only if it fits the remaining cut (cut minus
# time x rate already spent), else `$OUT/<item>.not-run.json` records `not run (cut)` and the
# next item is still probed. A finished item writes its outputs into $OUT/<item>/, then
# `$OUT/<item>.complete.json` (sha256 per file) last; the laptop's `scripts/exam_fetch.sh`
# downloads it at once (D9). A failed item is recorded and the order goes on.
# The hard-cut watchdog with self-termination is `scripts/pod_watchdog.sh` (C9): at the cut it
# stops the running item, flushes sha256sums.txt and removes the pod; no grace period.
# Item commands come from $ITEMS_LIB (default scripts/pod_exam_items.sh): `item_<name> <limit>
# <outdir>` (limit empty for the full run) and `item_total <name>` (units or questions).
# LAPTOP_NOT_RUN (laptop dry run only, empty on the pod), "item: reason; item: reason", records
# each named item as `not run (laptop: <reason>)` in `$OUT/<item>.not-run.json` without running it.
# Run from the repository root at the frozen commit, e.g.
#   POD=4090 POD_COST_PER_HR=0.74 HARD_CUT_USD=2.27 nohup bash scripts/pod_exam.sh \
#     > /workspace/pod_exam.out 2>&1 &
set -uo pipefail

: "${POD_COST_PER_HR:?set POD_COST_PER_HR to the pod costPerHr}"
: "${HARD_CUT_USD:?set HARD_CUT_USD to the hard cut of this pod}"
POD="${POD:-4090}"
OUT="${OUT:-data/phase07/pod}"
LOG="${LOG:-/workspace/pod_exam.log}"
SAMPLES="${SAMPLES:-/workspace/pod_exam.samples}"
PROGRESS="${PROGRESS:-/workspace/pod_exam.progress}"
ITEMS_LIB="${ITEMS_LIB:-scripts/pod_exam_items.sh}"
SPENT_S="${SPENT_S:-0}"
START=$(date +%s)
export POD_COST_PER_HR OUT PROGRESS

stage_py() {  # the stage rules; STAGE_PYTHON (one interpreter path) replaces uv in tests
  if [ -n "${STAGE_PYTHON:-}" ]; then
    "$STAGE_PYTHON" -m edge_rag.pod.exam_stage "$@"
  else
    uv run --frozen python -m edge_rag.pod.exam_stage "$@"
  fi
}

now_s() { date +%s.%N | sed 's/N$/0/'; }

stop_work() {  # the running item and its children
  local pid
  pid=$(cat "$OUT/.item.pid" 2> /dev/null) || return 0
  pkill -TERM -P "$pid" 2> /dev/null || true
  kill -TERM "$pid" 2> /dev/null || true
}

flush() {  # sha256sums.txt over every output so far, where the laptop reads them
  (cd "$OUT" && find . -type f ! -name 'sha256sums.*' ! -name '.item.pid' -print0 | sort -z \
    | xargs -0 -r sha256sum > sha256sums.tmp && mv sha256sums.tmp sha256sums.txt)
}

run_item() {  # run_item <item> <limit> <outdir>: in the background, its pid where stop_work reads it
  mkdir -p "$3"
  "item_$1" "$2" "$3" >> "$PROGRESS" 2>&1 &
  echo $! > "$OUT/.item.pid"
  wait $!
}

# shellcheck source=scripts/pod_watchdog.sh
source scripts/pod_watchdog.sh

laptop_reason() {  # laptop_reason <item>: its LAPTOP_NOT_RUN reason, else exit 1
  local entry entries
  IFS=';' read -ra entries <<< "${LAPTOP_NOT_RUN:-}"
  for entry in "${entries[@]}"; do
    entry="${entry#"${entry%%[![:space:]]*}"}"
    if [ "${entry%%:*}" = "$1" ]; then echo "${entry#*: }"; return 0; fi
  done
  return 1
}

# shellcheck disable=SC1090
source "$ITEMS_LIB"
mkdir -p "$OUT/probe"
watchdog_start

for item in $(stage_py order --pod "$POD" | tr -d '\r'); do
  if stage_py check "$OUT" "$item"; then
    echo "ITEM_SKIP $item already complete" >> "$LOG"
    continue
  fi
  if reason=$(laptop_reason "$item"); then
    printf '{"item": "%s", "state": "not run (laptop: %s)"}
' "$item" "$reason"       > "$OUT/$item.not-run.json"
    echo "ITEM_NOT_RUN_LAPTOP $item $(date -u +%FT%TZ)" >> "$LOG"
    continue
  fi
  size=$(stage_py probe-size "$item" | tr -d '\r')
  if [ -n "$size" ]; then
    echo "PROBE_START $item $size $(date -u +%FT%TZ)" >> "$LOG"
    t0=$(now_s)
    if ! run_item "$item" "$size" "$OUT/probe/$item"; then
      echo "ITEM_FAIL $item probe $(date -u +%FT%TZ)" >> "$LOG"
      continue
    fi
    t1=$(now_s)
    total=$(item_total "$item")
    spent=$((SPENT_S + $(date +%s) - START))
    seconds=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')
    if ! stage_py decide --probe-seconds "$seconds" --probe-count "$size" --total "$total" \
        --cut-usd "$HARD_CUT_USD" --spent-seconds "$spent" --rate "$POD_COST_PER_HR" \
        > "$OUT/probe/$item.decision.json"; then
      cp "$OUT/probe/$item.decision.json" "$OUT/$item.not-run.json"
      echo "ITEM_NOT_RUN_CUT $item $(date -u +%FT%TZ)" >> "$LOG"
      continue
    fi
  fi
  echo "ITEM_START $item $(date -u +%FT%TZ)" >> "$LOG"
  if run_item "$item" "" "$OUT/$item"; then
    files=$(cd "$OUT" && find "$item" -type f | sort | tr '\n' ' ')
    # shellcheck disable=SC2086
    stage_py mark "$OUT" "$item" $files
    echo "ITEM_DONE $item $(date -u +%FT%TZ)" >> "$LOG"
  else
    echo "ITEM_FAIL $item $(date -u +%FT%TZ)" >> "$LOG"
  fi
done

rm -f "$OUT/.item.pid"
flush
echo "STAGE_DONE all $(date -u +%FT%TZ)" >> "$LOG"
