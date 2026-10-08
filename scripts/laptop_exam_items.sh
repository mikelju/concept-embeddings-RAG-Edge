#!/usr/bin/env bash
# Laptop dry run of scripts/pod_exam.sh (plan task 6; CPU only, Windows ARM64, Git Bash): the pod
# items of scripts/pod_exam_items.sh with the laptop's torch index (CPU) and sentence-transformers
# layered for J-strong (the `pod` group resolves on Linux x86_64 only), no project dependency.
# Items that cannot run here are recorded `not run (laptop: <reason>)` through LAPTOP_NOT_RUN.
# Pod behaviour is unchanged: the pod never sources this file. Example (repository root):
#   ITEMS_LIB=scripts/laptop_exam_items.sh SIMULATE_DELETE=1 SPLIT=dev \
#     SPLIT_DIR=data/phase07/dryrun/sample/split OUT=data/phase07/dryrun/sample/pod \
#     LOG=data/phase07/dryrun/sample/pod.log SAMPLES=... PROGRESS=... \
#     POD_COST_PER_HR=1 HARD_CUT_USD=0.5 GLINER_MODEL_DIR=<gliner folder> bash scripts/pod_exam.sh
# shellcheck source=scripts/pod_exam_items.sh
source scripts/pod_exam_items.sh

TORCH_INDEX="${LAPTOP_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
LAPTOP_NOT_RUN="${LAPTOP_NOT_RUN-setup: no CUDA; g-l: no CUDA, colbert_env resolves on Linux x86_64 only; c3: pod-only inputs (old Phase 17 pairs) and checks; j-strong: needs the G-L list; g-r2: needs the G-L list; j-rrf3: needs the G-L list; g-a1: no CUDA/vLLM}"

main_py() {
  uv run --frozen --with sentence-transformers==6.0.1 --extra-index-url "$TORCH_INDEX" \
    --index-strategy unsafe-best-match python "$@"
}

# `qasper-gliner` reads data/phase07/$SPLIT, not $SPLIT_DIR: on a sample, LAPTOP_PAPERS=N keeps
# it to the first N papers, the sample of SPLIT_DIR.
item_gliner() {
  local limit=()
  [ -n "$1" ] && limit=(--limit-units "$1")
  [ -n "${LAPTOP_PAPERS:-}" ] && limit+=(--papers "$LAPTOP_PAPERS")
  gliner_py -m edge_rag.cli qasper-gliner "$SPLIT" --out "$2" --model-dir "$GLINER_MODEL_DIR" \
    "${limit[@]}"
}
