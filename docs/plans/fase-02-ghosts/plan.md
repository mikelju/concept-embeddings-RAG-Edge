# Phase 02 - Ghosts on the terrain: plan

Status: in progress
Spec: `spec.md` (approved 2026-10-02 under delegation)

## Decisions
Taken by the agent under the author's delegation.
- D1. One pod session runs G-L, then G-R, then G-A1, because G-R and G-A1 need G-L's index; setup is paid once. Reason: the forecast's setup allowances assume separate sessions.
- D2. PyLate's PLAID index instead of the `colbert-ai` library. Reason: the model card names both as compatible; PyLate installs from wheels on the pod, `colbert-ai` compiles CUDA extensions at run time.
- D3. Greedy decoding for G-A1, where `infer.py` samples at temperature 0.7. Reason: a ghost must be repeatable; the fidelity check (C4) runs both in greedy.
- D4. Pod code lives in `src/edge_rag/pod/` and runs under `uv run` from the frozen commit; heavy dependencies (pylate, vllm) go in an optional dependency group `pod`, so the laptop lock stays unchanged for the default group.

## Increments
| # | Increment | Criteria | Where | Status | Evidence |
|---|---|---|---|---|---|
| 1 | `edge-rag score` over ranking files; Phase 01 runs export depth-100 rankings; tests | C1, C8 | laptop | done | `uv run edge-rag reproduce --set <set>` (gate PASS on all three, writes `data/phase02/rankings/<set>/<system>.jsonl.gz` with `manifest.json`), then `uv run edge-rag score --set <set> --rankings data/phase02/rankings/<set>/<system>.jsonl.gz`: FS@2,048 P10-A/P10-B/P10-C/P14 = 4,125/4,536/4,801/5,224 (hotpotqa-dev), 436/524/669/761 (musique), 338/587/522/513 (multihop-rag), equal to the Phase 01 gate; tests in `tests/test_scoring.py`; `npm run check` exit 0 |
| 2 | Old references: J(P10-B) and union pool from old Phase 17 files, digest-pinned, scored | C1 | laptop | done | `uv run edge-rag old-refs --set <set>` then `uv run edge-rag score --set <set> --rankings data/phase02/rankings/<set>/j-<p10b,union>.jsonl.gz`: FS@2,048 J(P10-B) 5,198 / 704 / 822 and union 5,922 / 809 / 846 (hotpotqa-dev / musique / multihop-rag), all equal to the bar; inputs pinned to the sha256 recorded in old `phase17/outcome.json`, matched in place 2026-10-02 |
| 3 | Pod code: G-L encode, PLAID index, depth-100 search, timing manifest; G-R rerank with the old J-strong settings and the 300-pair check; Search-R1 vLLM driver, retrieval server over G-L, `infer.py` greedy comparison; one `pod_run.sh` with stage markers; probes on 1,000 units | C3, C4, C5 | laptop, then pod | pending | |
| 4 | Pod session: G-L (3 sets), G-R (3 sets), C3 check, C4 check, G-A1 (MuSiQue, MultiHop-RAG); download with digests; terminate | C2-C5, C7 | pod | pending | |
| 5 | Results table and `results.md` from code; paired tests | C6 | laptop | pending | |
| 6 | Optional items under the money rule | C7 | pod | pending | |
| 7 | Validation: `npm run check`, adversarial review, delivery | C8 | laptop | pending | |

## Money
| Date | Item | Time x rate (derived) | Balance delta (measured) |
|---|---|---|---|
| 2026-10-02 | balance before the phase: 29.22 USD (measured, API) | | |

## Findings
None yet.
