# Phase 06 - Literature rivals never measured on the bench: plan

Status: in progress
Spec: `spec.md` (approved by the author, 2026-10-06; frozen)
Base: branch `fase-06-rivals` from `fase-05-hop` at 5b735e5

## Decisions
Taken by the agent under the author's delegation for this phase (spec, decision 6).
- D1. Phase 06 outputs live in `data/phase06/` (`pairs/<set>/`, `scores/<set>/`, `rankings/<set>/`, `checks/`, `logs/`, `subsample/`, `results.json`); `results.md` is generated into this folder, as in Phases 03-05.
- D2. G-R2 pair preparation reuses Phase 04's `judge-pairs` code path (`src/edge_rag/judge.py`: `pair_row`, `write_once`) over G-L's pinned depth-100 rankings; the pod scorer is a new `src/edge_rag/pod/` module beside `rerank.py`, and the rankings are assembled on the laptop, as Phase 04's `assemble` did.
- D3. G-A1 reuses the Phase 02 driver `src/edge_rag/pod/searchr1.py` and `scripts/pod_run.sh` unchanged; only the question list (the 1,000 qids) and the index path differ.
- D4. G-A2 code runs on the pod from a pinned HippoRAG commit; the adapter (our units in, a depth-100 ranking out) is the only code of ours on that path, and the diff against the pinned commit is saved for C3.
- D5. Money readings and pods follow `.agents/skills/remote-gpu/SKILL.md`, taken by the main session; `HF_TOKEN` is checked for presence only with `env | grep -o "^HF_TOKEN="` and never printed or written to a file.
- D6. A rival that the training-data check makes an upper reference is not run (spec, pending checks); the agent does not override that rule under the delegation.

## Increments
Each increment leaves the product working and covers named criteria.
This file is the durable state: a new session resumes from here and from Git.

| # | Increment | Criteria | Files | Status | Evidence |
|---|---|---|---|---|---|
| 1 | Licence and training-data checks for G-R2, G-A2 and G-A1 (model cards, papers, repositories, dataset docs), each answer yes, no or unknown per set with source and date, each rival labelled ghost or upper reference; HotpotQA 1,000-qid subsample written once from `random.Random(20261002).sample` over the sorted qids, sha256 recorded here, test recomputing it | C1, C2 | `plan.md`, `data/phase06/subsample/`, tests | pending | |
| 2 | G-R2 pair files prepared on the laptop for the three sets from G-L's pinned top-100 (reusing the Phase 04 pair code), scorer code (model card prompt, default instruction, yes/no logit) and assembly into depth-100 rankings, with local unit tests on a hand-built example | C3, C4 | `src/edge_rag/`, `src/edge_rag/pod/`, `scripts/`, tests | pending | |
| 3 | G-R2 pod run: fidelity on 300 pairs against the model card's reference code (within 1e-3), reversed-batch determinism (within 1e-3), then scoring the three sets; rankings and manifests written once | C3, C4, C8 | `data/phase06/`, `plan.md` | pending | |
| 4 | G-A2 pod run on MultiHop-RAG: pinned HippoRAG commit, Llama-3.1-8B-Instruct through vLLM, GritLM-7B; recorded diff; depth-100 rankings and manifest with LLM tokens; `not run` if `HF_TOKEN` is still absent after every other item | C3, C4, C8 | `data/phase06/`, `plan.md` | pending | |
| 5 | G-A1 pod run on the HotpotQA 1,000: G-L HotpotQA index rebuilt with the Phase 02 D17 settings (new, separately digested artifact), Phase 02 driver unchanged (`git diff` recorded), evidence lists and manifest | C3, C4, C8 | `data/phase06/`, `plan.md` | pending | |
| 6 | Phase 06 rows in the shared results module `src/edge_rag/phase_results.py` (rivals, G-L, G-R, G-A1, best so far, best own system; McNemar per comparison; literature bar per set and class); tests of the state and the bar on a hand-built example; generated `data/phase06/results.json` and `docs/plans/fase-06-rivals/results.md` | C5, C6 | `src/edge_rag/phase_results.py`, `src/edge_rag/cli.py`, tests, `results.md` | pending | |
| 7 | Close: C7 `git grep` and search of `data/phase06/` manifests and logs for `qasper`; C8 closing `clientBalance` reading at least 2 h after the last termination and the three money figures; C9 `npm run check`; adversarial review; delivery (push and PR) | C7, C8, C9 | `plan.md`, master plan | pending | |

Run order of the pod items: 3, 4, 5; items 4 and 5 swap if Llama-3.1 access is not ready when item 4 would start (spec, decision 4).

## Licence and training-data record (C1)
Filled by increment 1, before the first pod.

## Money
Rule (spec, decision 2): phase cap = `clientBalance` read before the first pod minus 1.0 USD safety buffer (derived; about 13.1 USD at the 2026-10-05 reading of 14.1147504042 USD).
An item starts only if the spend so far plus its re-projection at the offered rate stays within the cap; hard cut per item at 1.5 times its projection (G-R2 3.3 USD, G-A2 4.5 USD, G-A1 6.9 USD), never past the cap.
The exam (Phase 07) is unfunded after this phase; it needs a top-up decided by the author.
- Before the first pod: pending (UTC time, `clientBalance`, `myself { pods }`, phase cap derived from it).
- Per pod: pending (rate offered, re-projection, start and end UTC, time x rate).
- Close: pending (reading at least 2 h after the last termination; balance delta; invoice when the author copies it).

## Decisions taken during execution

## Deviations
| ID | Summary | Affects criteria | Status |
|---|---|---|---|

## Adversarial review
| Round | Backend | Range | Lenses | Findings | Status |
|---|---|---|---|---|---|

## Results by criterion
Pending: filled at the close, per criterion, with the command run, the observed result and the evidence.

## Candidate learnings
