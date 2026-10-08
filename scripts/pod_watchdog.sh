#!/usr/bin/env bash
# Phase 07 hard-cut watchdog with pod self-termination (spec C9, plan increment 1).
# Extends the Phase 06 sampler of scripts/pod_gr2.sh (`over_cut`, `STOP_CUT`): every WATCH_S s it
# writes a sample line to $SAMPLES (UTC time, last progress line, cgroup memory); once
# (SPENT_S + elapsed) x POD_COST_PER_HR reaches HARD_CUT_USD it writes `STOP_CUT`, calls the
# caller's `stop_work`, then `flush` (outputs and their sums written where the laptop reads them),
# writes `FLUSHED`, and removes the pod: `runpodctl remove pod $RUNPOD_POD_ID`, else
# `runpodctl pod delete` (newer runpodctl), else the GraphQL `podTerminate` with the pod's own
# RUNPOD_API_KEY (never printed); each attempt is a `TERMINATE_TRY <how>` line in $LOG.
# SIMULATE_DELETE=1 (laptop dry run only) replaces the removal with a `TERMINATE_SIMULATED` line.
# Source it from a stage script that sets POD_COST_PER_HR, HARD_CUT_USD, LOG, SAMPLES, PROGRESS
# and defines `stop_work` and `flush`, then call `watchdog_start` (it sets the EXIT trap: a normal
# end stops the watchdog, a cut leaves it running until the pod is removed).

SPENT_S="${SPENT_S:-0}"
WATCH_S="${WATCH_S:-30}"
RUNPOD_GRAPHQL_URL="${RUNPOD_GRAPHQL_URL:-https://api.runpod.io/graphql}"
START="${START:-$(date +%s)}"

over_cut() {  # true once (SPENT_S + elapsed) x rate reaches HARD_CUT_USD; false without a cut
  [ -n "$HARD_CUT_USD" ] && awk -v s="$((SPENT_S + $(date +%s) - START))" -v r="$POD_COST_PER_HR" \
    -v c="$HARD_CUT_USD" 'BEGIN { exit !(s * r / 3600 >= c) }'
}

terminate_self() {  # runpodctl (old, then new syntax), then GraphQL podTerminate; first success wins
  if [ "${SIMULATE_DELETE:-0}" = 1 ]; then
    echo "TERMINATE_SIMULATED $(date -u +%FT%TZ) laptop dry run, no RunPod call" >> "$LOG"
    return 0
  fi
  local id=${RUNPOD_POD_ID:?set RUNPOD_POD_ID} response
  echo "TERMINATE_TRY runpodctl-remove $(date -u +%FT%TZ)" >> "$LOG"
  runpodctl remove pod "$id" >> "$LOG" 2>&1 && return 0
  echo "TERMINATE_TRY runpodctl-pod-delete $(date -u +%FT%TZ)" >> "$LOG"
  runpodctl pod delete "$id" >> "$LOG" 2>&1 && return 0
  echo "TERMINATE_TRY graphql $(date -u +%FT%TZ)" >> "$LOG"
  if [ -n "${RUNPOD_API_KEY:-}" ] && response=$(curl -sS --max-time 30 \
      -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$id\\\"}) }\"}" \
      "$RUNPOD_GRAPHQL_URL") && [[ "$response" != *'"errors"'* ]]; then
    return 0
  fi
  echo "TERMINATE_FAIL $(date -u +%FT%TZ) the laptop backup timer must remove the pod" >> "$LOG"
  return 1
}

cut_watchdog() {  # every WATCH_S s a sample; at the cut: STOP_CUT, stop_work, flush, FLUSHED, remove
  while true; do
    echo "$(date -u +%FT%TZ) | $(tail -n 1 "$PROGRESS" 2> /dev/null) | current $(cat /sys/fs/cgroup/memory.current 2> /dev/null) peak $(cat /sys/fs/cgroup/memory.peak 2> /dev/null)" >> "$SAMPLES"
    if over_cut; then
      echo "STOP_CUT $(date -u +%FT%TZ) hard cut $HARD_CUT_USD USD reached" >> "$LOG"
      stop_work || true
      flush || echo "FLUSH_FAIL $(date -u +%FT%TZ)" >> "$LOG"
      echo "FLUSHED $(date -u +%FT%TZ)" >> "$LOG"
      terminate_self
      return
    fi
    sleep "$WATCH_S"
  done
}

watchdog_start() {  # background watchdog; the EXIT trap stops it unless the cut was reached
  cut_watchdog &
  WATCHDOG=$!
  trap 'grep -q "^STOP_CUT" "$LOG" 2> /dev/null || kill "$WATCHDOG" 2> /dev/null || true' EXIT
}
