#!/usr/bin/env bash
# Phase 07 laptop side of plan D9: download each pod item as soon as it is marked complete.
# Every POLL_S s, lists the pod's `<item>.complete.json` and `<item>.not-run.json`; for a new
# marker it copies the marker and the item's folder into $LOCAL_OUT, then accepts the item only
# when `python -m edge_rag.pod.exam_stage check` passes on the local copy (sha256 per file),
# else it retries on the next poll. A `not-run` record is copied as is. It stops when the pod's
# log has `STAGE_DONE all` or `STOP_CUT`, or when the pod cannot be reached POLL_FAILS times.
# Transport is $REMOTE_LS / $REMOTE_GET (default ssh and scp to $POD_SSH on $POD_PORT); the
# tests replace them with local commands. Example (laptop, Git Bash, repository root):
#   POD_SSH=root@1.2.3.4 POD_PORT=40022 bash scripts/exam_fetch.sh
set -uo pipefail

REMOTE_OUT="${REMOTE_OUT:-/workspace/edge-rag/data/phase07/pod}"
REMOTE_LOG="${REMOTE_LOG:-/workspace/pod_exam.log}"
LOCAL_OUT="${LOCAL_OUT:-data/phase07/pod}"
POLL_S="${POLL_S:-60}"
POLL_FAILS="${POLL_FAILS:-10}"
KEY="${KEY:-$HOME/.ssh/runpod_ed25519}"
FETCH_LOG="${FETCH_LOG:-$LOCAL_OUT/fetch.log}"

remote_ls() {  # remote_ls <glob or file>: names, one per line
  if [ -n "${REMOTE_LS:-}" ]; then "$REMOTE_LS" "$@"; return; fi
  ssh -i "$KEY" -p "$POD_PORT" -o BatchMode=yes "$POD_SSH" "cd '$REMOTE_OUT' && ls -1 $1 2> /dev/null; true"
}

remote_get() {  # remote_get <relative path> <local folder>: recursive copy
  if [ -n "${REMOTE_GET:-}" ]; then "$REMOTE_GET" "$@"; return; fi
  scp -r -i "$KEY" -P "$POD_PORT" -o BatchMode=yes "$POD_SSH:$REMOTE_OUT/$1" "$2/"
}

remote_log() {
  if [ -n "${REMOTE_LOG_CMD:-}" ]; then "$REMOTE_LOG_CMD"; return; fi
  ssh -i "$KEY" -p "$POD_PORT" -o BatchMode=yes "$POD_SSH" "cat '$REMOTE_LOG'"
}

stage_py() {
  if [ -n "${STAGE_PYTHON:-}" ]; then
    "$STAGE_PYTHON" -m edge_rag.pod.exam_stage "$@"
  else
    uv run --frozen python -m edge_rag.pod.exam_stage "$@"
  fi
}

fetch_round() {  # one poll: every new complete item and not-run record
  local markers marker item
  markers=$(remote_ls '*.complete.json *.not-run.json') || return 1
  for marker in $markers; do
    marker=${marker%$'\r'}
    case "$marker" in
      *.complete.json)
        item=${marker%.complete.json}
        stage_py check "$LOCAL_OUT" "$item" && continue
        remote_get "$item" "$LOCAL_OUT" && remote_get "$marker" "$LOCAL_OUT" || continue
        if stage_py check "$LOCAL_OUT" "$item"; then
          echo "FETCHED $item $(date -u +%FT%TZ)" >> "$FETCH_LOG"
        else
          echo "FETCH_BAD $item $(date -u +%FT%TZ) sha256 mismatch, retry" >> "$FETCH_LOG"
          rm -f "$LOCAL_OUT/$marker"
        fi ;;
      *.not-run.json)
        [ -f "$LOCAL_OUT/$marker" ] && continue
        remote_get "$marker" "$LOCAL_OUT" && echo "FETCHED_NOT_RUN ${marker%.not-run.json} $(date -u +%FT%TZ)" >> "$FETCH_LOG" ;;
    esac
  done
}

mkdir -p "$LOCAL_OUT"
fails=0
while true; do
  if fetch_round; then fails=0; else fails=$((fails + 1)); fi
  log=$(remote_log 2> /dev/null) || log=""
  if [[ "$log" == *"STAGE_DONE all"* || "$log" == *"STOP_CUT"* ]]; then
    fetch_round
    echo "FETCH_END $(date -u +%FT%TZ)" >> "$FETCH_LOG"
    break
  fi
  if [ "$fails" -ge "$POLL_FAILS" ]; then
    echo "FETCH_LOST $(date -u +%FT%TZ) pod unreachable $fails times" >> "$FETCH_LOG"
    exit 1
  fi
  sleep "$POLL_S"
done
