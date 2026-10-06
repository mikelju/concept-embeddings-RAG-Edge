#!/usr/bin/env bash
# Phase 06 G-A2 pod session (plan increment 4; recipe docs/plans/fase-06-rivals/ga2_recipe.md):
# HippoRAG 2 at its pinned commit, in its own venv, with Llama-3.1-8B-Instruct served by vLLM
# and GritLM-7B loaded in-process, both on one 48 GB GPU (the normal flow needs both live).
# Run from the repository root at the frozen commit, with data/phase06/ga2/ uploaded, e.g.
#   HF_TOKEN=... POD_COST_PER_HR=0.49 nohup bash scripts/pod_ga2.sh > /workspace/pod_ga2.out 2>&1 &
# Progress: `STAGE_DONE <name>` (or STAGE_SKIP / STAGE_FAIL) lines in $LOG; a sampler line every
# 30 s in $SAMPLES (UTC time, last progress line, GPU memory used, cgroup memory).
# A non-empty C3 diff, a fact-filter error or a probe projection past the cut stops the session.
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN (read token with Llama-3.1 access); it is never printed}"
: "${POD_COST_PER_HR:?set POD_COST_PER_HR to the pod costPerHr}"
CUT_USD="${CUT_USD:-4.5}"            # spec: hard cut for G-A2, 1.5 x its projection
SPENT_S="${SPENT_S:-0}"              # pod seconds already used before this script started
PROBE_UNITS="${PROBE_UNITS:-500}"
PROBE_QUESTIONS="${PROBE_QUESTIONS:-50}"
VLLM_UTIL="${VLLM_UTIL:-0.5}"        # vLLM's share of GPU memory; GritLM-7B takes the rest
LOG="${LOG:-/workspace/pod_ga2.log}"
SAMPLES="${SAMPLES:-/workspace/pod_ga2.samples}"
PROGRESS="${PROGRESS:-/workspace/pod_ga2.progress}"
HIPPORAG_COMMIT=2bfd831417202b49cda9da7973e141a456e16872
LLAMA=meta-llama/Llama-3.1-8B-Instruct
LLAMA_REV=0e9e39f249a16976918f6564b8830bc894c89659
GRITLM=GritLM/GritLM-7B
GRITLM_REV=cb7f7ffb99c0c24ca2c325d798d7ef5c455d5339
HR=/workspace/HippoRAG
VENV=/workspace/hr-venv
PORT=6578
BUNDLE=data/phase06/ga2
OUT=$BUNDLE/pod
export HF_HOME=/workspace/hf POD_COST_PER_HR HF_TOKEN
export OPENAI_API_KEY=EMPTY  # the OpenAI client wants a key; vLLM ignores it

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

sampler() {
  while true; do
    echo "$(date -u +%FT%TZ) | $(tail -n 1 "$PROGRESS" 2> /dev/null | tr '\r' ' ' | tail -c 200) | gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2> /dev/null) | current $(cat /sys/fs/cgroup/memory.current 2> /dev/null) peak $(cat /sys/fs/cgroup/memory.peak 2> /dev/null)" >> "$SAMPLES"
    sleep 30
  done
}

sums() {
  (cd "$OUT" && find . -type f ! -name 'sha256sums.*' ! -name '*.tmp' -print0 | sort -z | xargs -0 -r sha256sum > sha256sums.tmp && mv sha256sums.tmp sha256sums.txt)
}

clone() {
  [ -d "$HR/.git" ] || git clone -q https://github.com/OSU-NLP-Group/HippoRAG.git "$HR"
  git -C "$HR" checkout -q "$HIPPORAG_COMMIT"
  [ "$(git -C "$HR" rev-parse HEAD)" = "$HIPPORAG_COMMIT" ]
}

venv() {  # HippoRAG's own pins (torch 2.5.1, transformers 4.45.2, vllm 0.6.6.post1, gritlm 1.0.2)
  uv venv -q --python 3.11 "$VENV"
  uv pip install -q --python "$VENV/bin/python" -e "$HR[vllm,gritlm]"
  uv pip freeze --python "$VENV/bin/python" > "$OUT/pip-freeze.txt"
  "$VENV/bin/python" -c 'import torch; x = torch.randn(2048, 2048, device="cuda"); print(torch.__version__, torch.cuda.get_device_name(0), (x @ x).sum().item())'
}

c3_diff() {  # spec C3: the diff against the pinned commit, recorded; this recipe changes nothing
  git -C "$HR" diff "$HIPPORAG_COMMIT" > "$OUT/c3-g-a2.diff"
  git -C "$HR" status --porcelain --untracked-files=all > "$OUT/c3-g-a2.status"
  [ ! -s "$OUT/c3-g-a2.diff" ]
}

models() {  # both models at their pinned revisions; GritLM is loaded by name, so its refs/main
  # is pointed at the pinned snapshot and the HippoRAG process runs with HF_HUB_OFFLINE=1
  "$VENV/bin/huggingface-cli" download "$LLAMA" --revision "$LLAMA_REV" --exclude 'original/*' > /dev/null
  "$VENV/bin/huggingface-cli" download "$GRITLM" --revision "$GRITLM_REV" > /dev/null
  mkdir -p "$HF_HOME/hub/models--GritLM--GritLM-7B/refs"
  echo -n "$GRITLM_REV" > "$HF_HOME/hub/models--GritLM--GritLM-7B/refs/main"
}

serve() {  # vLLM's OpenAI-compatible server, default sampling; only memory and length are set
  nohup "$VENV/bin/vllm" serve "$LLAMA" --revision "$LLAMA_REV" --tokenizer-revision "$LLAMA_REV" \
    --port "$PORT" --gpu-memory-utilization "$VLLM_UTIL" --max-model-len 8192 \
    > /workspace/vllm.out 2>&1 &
  echo $! > /workspace/vllm.pid
  for _ in $(seq 1 90); do
    curl -sf "http://localhost:$PORT/v1/models" > /dev/null && return 0
    sleep 10
  done
  return 1
}

run() {  # run <save dir> <out dir> [extra args]
  local save=$1 out=$2
  shift 2
  HF_HUB_OFFLINE=1 "$VENV/bin/python" src/edge_rag/pod/hipporag_run.py --bundle "$BUNDLE" \
    --save-dir "$save" --out "$out" --llm-base-url "http://localhost:$PORT/v1" \
    --hipporag-dir "$HR" --diff "$OUT/c3-g-a2.diff" "$@" 2>&1 | tee -a "$PROGRESS"
  return "${PIPESTATUS[0]}"
}

probe_gate() {  # linear projection from the probe; stops when it passes the cut (deviation 02.1)
  python3 - "$OUT/probe/hipporag.manifest.json" "$POD_COST_PER_HR" "$CUT_USD" "$((SPENT_S + SECONDS))" << 'EOF' | tee -a "$PROGRESS"
import json, sys
m = json.load(open(sys.argv[1])); rate, cut, spent = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
s = m["seconds"]; n = m["scope"]
index = s["index"] / n["units"] * 27989; query = s["retrieve"] / n["questions"] * 2255
hours = (spent + index + query + 900) / 3600  # + 15 min for writing and download
usd = hours * rate
print(f"PROBE_PACE index_s {index:.0f} retrieve_s {query:.0f} total_h {hours:.2f} usd {usd:.2f} cut {cut} (linear projection)")
sys.exit(0 if usd <= cut else 1)
EOF
  return "${PIPESTATUS[0]}"
}

mkdir -p "$OUT"
sampler &
SAMPLER=$!
trap 'kill "$SAMPLER" 2> /dev/null || true; [ -f /workspace/vllm.pid ] && kill "$(cat /workspace/vllm.pid)" 2> /dev/null || true' EXIT

(cd "$BUNDLE" && sha256sum -c bundle.sha256)
stage clone "" clone
stage venv "$OUT/pip-freeze.txt" venv
stage c3-diff "" c3_diff
stage models "" models
stage serve "" serve
stage probe "$OUT/probe/hipporag.manifest.json" \
  run /workspace/hr_probe "$OUT/probe" --limit-units "$PROBE_UNITS" --limit-questions "$PROBE_QUESTIONS"
stage probe-gate "" probe_gate
# FULL_ARGS=--force-index-from-scratch only after a crash between encoding and the saved graph
# (HippoRAG then refuses to resume; its LLM cache still replays every finished OpenIE call).
# shellcheck disable=SC2086
stage full "$OUT/hipporag.manifest.json" run /workspace/hr_save "$OUT" ${FULL_ARGS:-}
sums
echo "STAGE_DONE all" >> "$LOG"
