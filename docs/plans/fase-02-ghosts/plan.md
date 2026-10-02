# Phase 02 - Ghosts on the terrain: plan

Status: in progress (paused 2026-10-02 at increment 4 by the author)
Spec: `spec.md` (approved 2026-10-02 under delegation)

## Decisions
Taken by the agent under the author's delegation.
- D1. One pod session runs G-L, then G-R, then G-A1, because G-R and G-A1 need G-L's index; setup is paid once. Reason: the forecast's setup allowances assume separate sessions.
- D2. PyLate's PLAID index instead of the `colbert-ai` library. Reason: the model card names both as compatible; PyLate installs from wheels on the pod, `colbert-ai` compiles CUDA extensions at run time.
- D3. Greedy decoding for G-A1, where `infer.py` samples at temperature 0.7. Reason: a ghost must be repeatable; the fidelity check (C4) runs both in greedy.
- D4. Pod code lives in `src/edge_rag/pod/` and runs under `uv run` from the frozen commit; heavy dependencies (pylate, vllm) go in an optional dependency group `pod`, so the laptop lock stays unchanged for the default group.
- D5. G-L runs in its own locked environment `src/edge_rag/pod/colbert_env` (PyLate 1.6.0 needs transformers <=5.3.0 and fast-plaid builds for torch 2.11, against the harness pins torch 2.13.0 and transformers 5.16.1); G-A1 reaches G-L's index through a local `colbert --serve` endpoint with Search-R1's `/retrieve` API. Dependencies added under the delegation.
- D6. numpy is pinned 2.3.5 on linux x86_64 only (vllm 0.30.0 needs <2.5, as in the old lock); the laptop keeps 2.5.2 and every figure is scored on the laptop.
- D7. A corpus larger than one index chunk (HotpotQA, chunks of 500,000 units) builds its PLAID centroids from a first chunk that is every `stride`-th unit of the whole corpus, then adds the rest through fast-plaid's update; the manifest records the stride. Reason: centroids from the first 500,000 units in file order would not span the corpus. This is not PyLate's single-call build and can move C2; it is stated with the result.
- D8. `colbert_env` takes `torch==2.11.0+cu126` from the PyTorch cu126 index (as the harness does), not PyPI's CUDA 13 build. Reason: F2; the pod driver is CUDA 12.8. The `fast-plaid==1.4.6.2110` linux wheel links only `libtorch`, `libtorch_cpu`, `libtorch_python` and `libc10` (read from the wheel, 2026-10-02), so it is indifferent to torch's CUDA build; the relock changed only the `nvidia-*`/CUDA packages (cu13 to cu12), PyLate and transformers are unchanged.

## Increments
| # | Increment | Criteria | Where | Status | Evidence |
|---|---|---|---|---|---|
| 1 | `edge-rag score` over ranking files; Phase 01 runs export depth-100 rankings; tests | C1, C8 | laptop | done | `uv run edge-rag reproduce --set <set>` (gate PASS on all three, writes `data/phase02/rankings/<set>/<system>.jsonl.gz` with `manifest.json`), then `uv run edge-rag score --set <set> --rankings data/phase02/rankings/<set>/<system>.jsonl.gz`: FS@2,048 P10-A/P10-B/P10-C/P14 = 4,125/4,536/4,801/5,224 (hotpotqa-dev), 436/524/669/761 (musique), 338/587/522/513 (multihop-rag), equal to the Phase 01 gate; tests in `tests/test_scoring.py`; `npm run check` exit 0 |
| 2 | Old references: J(P10-B) and union pool from old Phase 17 files, digest-pinned, scored | C1 | laptop | done | `uv run edge-rag old-refs --set <set>` then `uv run edge-rag score --set <set> --rankings data/phase02/rankings/<set>/j-<p10b,union>.jsonl.gz`: FS@2,048 J(P10-B) 5,198 / 704 / 822 and union 5,922 / 809 / 846 (hotpotqa-dev / musique / multihop-rag), all equal to the bar; inputs pinned to the sha256 recorded in old `phase17/outcome.json`, matched in place 2026-10-02 |
| 3 | Pod code: G-L encode, PLAID index, depth-100 search, timing manifest; G-R rerank with the old J-strong settings and the 300-pair check; Search-R1 vLLM driver, retrieval server over G-L, `infer.py` greedy comparison; one `pod_run.sh` with stage markers; probes on 1,000 units | C3, C4, C5 | laptop, then pod | done (code; GPU parts unrun) | commits bd5067e, ee70616; pure-Python tests pass; PyLate, fast-plaid and vLLM calls verified only on the pod (increment 4) |
| 4 | Pod session: G-L (3 sets), G-R (3 sets), C3 check, C4 check, G-A1 (MuSiQue, MultiHop-RAG); download with digests; terminate | C2-C5, C7 | pod | stopped (author's decision), not run | recipe `pod_recipe.md` (commit 96f9ead); session 1 pod `bf23ln5vq5f7ac` at commit 96f9ead: old inputs uploaded, both environments synced, `pod` group kernel check passed (torch 2.13.0+cu126 on the RTX 4090), `colbert_env` kernel check failed (F2); stopped on the author's order relayed by the launching session, pod terminated, nothing downloaded (no outputs beyond the sync log) |
| 5 | Results table and `results.md` from code; paired tests | C6 | laptop | pending | |
| 6 | Optional items under the money rule | C7 | pod | pending | |
| 7 | Validation: `npm run check`, adversarial review, delivery | C8 | laptop | pending | |

## Money
| Date | Item | Time x rate (derived) | Balance delta (measured) |
|---|---|---|---|
| 2026-10-02 | balance before the phase: 29.22 USD (measured, API) | | |
| 2026-10-02 | pod `bf23ln5vq5f7ac`, 1x RTX 4090 Secure Cloud, `costPerHr` 0.74 (measured, API); created 18:07:04Z, terminated 18:11:07Z (laptop clock, UTC) | 4.05 min x 0.74 USD/h = 0.05 USD (derived) | to be measured later (billing lags; balance read right after termination still 29.22 USD, measured); `myself { pods }` empty after `podTerminate` (measured) |
| 2026-10-02 | balance before session 2: 29.17 USD, `currentSpendPerHr` 0, no pods (measured, API) | | 29.22 - 29.17 = 0.05 USD for session 1 (derived from two measured balances) |

## Findings
- F1. Gold share at 5 (per-question fraction of gold in the top 5, HippoRAG's convention) equals Recall@5 by definition here; it is kept as the bridge label to the literature and adds no separate number.
- F2. `colbert_env` resolves `torch==2.11.0` from PyPI, a CUDA 13 build (it pulls `nvidia-*` cu13 wheels); the pod's driver 570.195.03 (CUDA 12.8) refuses it: `RuntimeError: The NVIDIA driver on your system is too old (found version 12080)`. Next step before G-L can run: route `colbert_env`'s torch to a CUDA 12.x PyTorch index (as the harness does with `pytorch-cu126`) and check that the `fast-plaid==1.4.6.2110` wheel loads against it, or request a pod with a CUDA 13 driver; then rerun the `sync` stage. Not attempted in session 1. Fixed on the laptop by D8; checked on the pod in session 2.
