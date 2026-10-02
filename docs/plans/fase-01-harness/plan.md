# Phase 01 - Harness and reproduction gate: plan and results

Status: ready locally
Spec: `spec.md` (frozen)
Base: branch `iniciar`, commit `beb78d2`; work branch `fase-01-harness`

## Decisions
Taken by the agent under the author's delegation (master plan approval, 2026-10-02).
- D1. Weights and alpha are pinned in `config.py` (0.5 / 0.5; 0.5 / 0.3 / 0.2; alpha 0.75); the two fit files are read only to check their recorded digests and that they hold the pinned values. Reason: one source per decision, and the fits are old artifacts.
- D2. Dense stays an exact per-query matrix-vector product, not a batched one. Reason: a batched product can change the float accumulation order and so a tie at a list boundary; speed is not a criterion.
- D3. BM25 is rebuilt from the corpus on every run and checked against the recorded index digest, as the old project did (the index was never stored).
- D4. Question vectors are served by qid instead of by question text. Reason: the old table refused two different vectors for one text, so the two lookups give the same vector; qid is the simpler key.
- D5. GLiNER extraction code is not copied in this phase. Reason: no run is needed (indexes are cached) and the configuration digest is recomputed from the recorded `extraction.json` configuration block with the old function's formula; the extractor is copied when a phase must extract.
- D6. Library versions that can move a number are pinned to the old lock's laptop versions: numpy 2.5.2, scipy 1.18.1, bm25s 0.3.11, torch 2.13.0 (+cpu here, cu126 on linux x86_64), transformers 5.16.1, sentence-transformers 6.0.1, tokenizers 0.23.1. These are the old project's dependencies, re-declared here (dependency added to this repository; authorized by the delegation).
- D7. The sentence-transformers backend (`SentenceTransformerBackend`) is copied for later phases but not exercised by the gate (the cached vectors are the inputs of record).
- D8. `npm ci` was run in the worktree so `npm run check` could find the pinned devDependencies (no new dependency).

## Increments
Each increment leaves the product working and covers concrete criteria.

- [x] 1. uv project, `edge_rag` package, torch source routing, ruff format from the first commit (C5) - evidence: `uv sync` installed `torch==2.13.0+cpu`; `uv run python -c "import numpy, scipy, torch"` printed `2.5.2 1.18.1 2.13.0+cpu`.
- [x] 2. Read-only data root, digests, loaders and caches (C2, C3) - evidence: `OLD_DATA_ROOT` read by `config.old_data_root()`; every write goes through `artifacts.guard_write`; tests `test_writes_under_the_old_data_root_are_refused`, `test_paths_cannot_escape_the_set_directory`, `test_a_mismatched_digest_is_refused`, `test_corpus_ids_and_order_are_verified`, `test_legacy_keys_reproduce_the_recorded_cache_names`, `test_caches_are_per_corpus` pass.
- [x] 3. Retrievers, fusion, hops, budget, metrics and exact McNemar (C4, C6) - evidence: 16 tests pass (`uv run pytest -q`), including the overflow case at n = 1,300 discordant against `scipy.stats.binomtest`.
- [x] 4. CLI `edge-rag reproduce` and the gate on MultiHop-RAG (C1) - evidence: below.
- [x] 5. Gate on MuSiQue (C1) - evidence: below.
- [x] 6. Gate on HotpotQA dev FullWiki (C1) - evidence: below.
- [x] 7. `npm run check` runs pytest, ruff and mypy (C5) - evidence: `npm run check` exit 0 ("Todos los casos correctos", 16 passed, "All checks passed!", "Success: no issues found in 12 source files").

## Deviations
| ID | Summary | Affects criteria | Status |
|---|---|---|---|

## Adversarial review
| Round | Backend | Range | Lenses | Findings | Status |
|---|---|---|---|---|---|
| - | - | - | - | not run in this session (the sdd-delivery review is pending before the PR) | pending |

## Results
Per criterion: the command or path run, the observed result, and where the evidence is.

C1, MultiHop-RAG: `uv run edge-rag reproduce --set multihop-rag`.
P10-A 338, P10-B 587, P10-C 522, P14 513 of 2,255: all match; `GATE PASS`.
Cross-check: 2,255 of 2,255 live fused lists equal the stored `rankings-*.jsonl.gz` lists, for each of the four systems.
23 digests checked; retrieval 8.4 s.

C1, MuSiQue: `uv run edge-rag reproduce --set musique`.
P10-A 436, P10-B 524, P10-C 669, P14 761 of 2,417: all match; `GATE PASS`.
Cross-check: 2,417 of 2,417 lists equal the stored ones for each system.
23 digests checked; retrieval 19.1 s.

C1, HotpotQA dev FullWiki: `uv run edge-rag reproduce --set hotpotqa-dev` (log `data/logs/reproduce-hotpotqa-dev.log`).
P10-A 4,125, P10-B 4,536, P10-C 4,801, P14 5,224 of 7,405: all match; `GATE PASS`.
The rebuilt BM25 index, the 5,233,329 passage vectors (file SHA-256 and float32 digest) and the entity index matched their recorded digests; 22 digests checked.
Timings on the laptop: sha256 30.2 s, corpus and questions 40.7 s, BM25 rebuild 217.4 s, vectors 68.3 s, entity index 16.9 s, retrieval of 7,405 questions 1,888.4 s; working set about 5.4 GB during the BM25 build (peak not recorded).
No stored-ranking cross-check here: the stored HotpotQA lists are component lists (`dev-lists.jsonl.gz`), not fused rankings, and the live counts already match.

C1, GLiNER configuration digest: `2f7864661b8ce7ff` recomputed from `extraction.json` of each set and checked in every run (`GLiNER configuration digest` entry of `digests_checked`).

C4: the paired p values the harness computes on MuSiQue (P10-B vs P10-A 2.06393982938988e-10; P10-C vs P10-B 1.3454696936403321e-22; P14 vs P10-C 1.741488902287888e-10) appear verbatim in the old `data/phase15/outcome.json` (grep count 3).

C6: each result JSON carries `recall_at_k` (k = 2, 5, 10, 20, 100) and `ndcg_at_10` per system; for example MuSiQue P14 Recall@10 0.5564, nDCG@10 0.5276.

Pending: the adversarial review and the delivery (push and PR) were not done in this session; peak memory was not instrumented.

Result files: `data/results/reproduce-<set>.json` (git-ignored), with counts, digests checked, timings and outcome digests.

## Candidate learnings
Only reusable lessons with a verbatim quote from the session; consolidated when the phase closes.
