# Phase 00 - State of the art: plan and results

Status: in progress
Spec: `spec.md` (frozen)
Base: branch `fase-00-state-of-the-art` from `iniciar`

## Increments

This file is the durable state: a new session resumes from here and from Git.

- [x] 1. Cheap-and-fast survey: records in `work/ledger-cheap.json`, notes in `work/notes-cheap.md` (C1-C3) - evidence: 128 records, all `verified: true` (script count); the subagent reported 21 self-reported, the file marks 15 `self_reported: true` (script count, 2026-10-02)
- [x] 2. Expensive-and-slow survey: records in `work/ledger-expensive.json`, notes in `work/notes-expensive.md` (C1-C3) - evidence: 156 records, 154 `verified: true`; the two unverified are R1-Searcher judge scores with an unreadable table header (script count)
- [x] 3. Benchmarks and metrics: per benchmark, its settings, metrics, leaderboards and best figures; exam candidates in detail, in `work/benchmarks.md` (C4) - evidence: `work/benchmarks.md` written (8 benchmarks, metric table, exam comparison recommending QASPER); MuSiQue and QASPER leaderboards not read
- [x] 4. Merge into `ledger.json` with a validation script `work/check_ledger.py` (C1, C3) - evidence: `python work/check_ledger.py` on 2026-10-02 printed 284 records, 0 problems (unique ids, all 25 keys present, every null field explained in `why_missing`, every unverified record with `why_unverified`), verified True 282 / False 2, self_reported True 15; no mechanical fix was needed
- [x] 5. `survey.md`: metric definitions, coverage, quality-cost maps, ghosts, recommendations, RunPod forecast (C2, C4-C8) - evidence: `survey.md` written; the script printed 81 ids cited, 81 in the ledger, 81 verified (100.0 percent; 4 self-reported)
- [ ] 6. Record the decisions in the master plan (C7); adversarial review; Spanish summary for the author - evidence: seven decision rows added to `../0_plan_maestro.md` on 2026-10-02 (six open decisions plus the RunPod forecast), its open-decisions list emptied, Phase 00 set to ready locally; adversarial review and Spanish summary pending

## Deviations
| ID | Summary | Affects criteria | Status |
|---|---|---|---|

## Adversarial review
| Round | Backend | Range | Lenses | Findings | Status |
|---|---|---|---|---|---|
| 1 | local | `iniciar..fase-00-state-of-the-art` | evidence; correctness and scope; security skipped (the only code is a stdlib script reading local JSON) | 2 important + 8 important + 7 minor: E1 confirmed, H4 split by metric and k; E2 confirmed, Table 5 rows added (closed API as context, 4 records), 2,255 now read in the paper; E3 confirmed, NQ + HotpotQA wording; E4 confirmed, the 66.7 vs 67.5 source is stated; E5 confirmed, RankLLaMA-13B over the A bound; E6 confirmed, the share is flag consistency, the reviewer's 12 re-read figures recorded; E7 confirmed (Table 3 reproduced HippoRAG 90.4), record added and wording fixed; R1 confirmed (answerai-colbert-small-v1 trains on a mix with HotpotQA), H1 labelled per row, G-L kept under a stated training rule; R2 confirmed, one rule: general-purpose models may include a terrain train split, single-benchmark fits (MDR) are references; R3 plausible, G-R kept (measured on our harness) and Qwen3-Reranker-0.6B added as optional G-R2; R4 confirmed, G-R is J-strong over G-L top-100, union pool only an old reference; R5 confirmed, EM fidelity claim dropped, driver-vs-script check instead; R6 confirmed (infer.py is one-at-a-time transformers), batched driver required, G-A1 re-projected; R7 confirmed, PLAID 2-bit about 13 GB, G-L 3.9 GPU-h; R8 confirmed in part (55.5 % is published in the QASPER paper, not computed by us), test-label count removed, multi-annotator gold rule added; M1 confirmed, licence rule stated, G-A2 encoder GritLM-7B (Apache-2.0); M2 confirmed, candidates and exam core ring-fenced, LLM graph ghost last, forecast 24.6 USD | fixed (all 17; none discarded) |

## Results

- C1 met: `ledger.json` has 284 records, one per (system, benchmark, setting, metric) figure, each with a `source` locator; the script finds zero required fields empty without a `why_missing` note (measured).
  Gaps: `licence` is null in 256 records (explained, mostly "not checked"); `code_url` and `weights_public` partly come from known repository names and were not opened this phase.
- C2 met: every Scope family has at least one record (`survey.md` section 1); graph without an LLM has only LinearRAG, with answer-accuracy figures in an unknown setting.
- C3 met: 287 of 289 records verified ledger-wide, and 87 of 87 figures cited in the maps and ghost list (100.0 percent, measured by the script, after review round 1); 15 records are self-reported model-card figures, labelled.
  The share measures the consistency of flags set by the surveyors; the round-1 evidence reviewer independently re-read 12 key figures and all matched.
- C4 met: `survey.md` section 2 defines every metric string in the ledger with unit, setting and relation to Full Support; no table mixes metrics or settings.
  Round 1 split H4 by metric and k and labelled every H1 row in-domain, zero-shot or not stated for HotpotQA train.
  Gap: no published system reports Full Support at any budget, and HippoRAG's recall@5 definition is assumed, not verified.
- C5 met: maps for HotpotQA, MuSiQue, 2Wiki and MultiHop-RAG (`survey.md` section 4), one table per (setting, metric); costs reported where the sources give them, otherwise labelled projection.
  Gaps: no full-corpus retrieval figure for MuSiQue, no class-A figure on MultiHop-RAG, and leaderboards for MuSiQue and QASPER were not read.
- C6 met: ghosts G-L answerai-colbert-small-v1 (Apache-2.0, in-domain on HotpotQA, PLAID 2-bit), G-R bge-reranker-v2-m3 (J-strong) over G-L's top-100, optional G-R2 Qwen3-Reranker-0.6B, G-A1 Search-R1 7B through a batched driver and G-A2 HippoRAG 2 with Llama-3.1-8B and GritLM-7B, each with figure, reason and projected cost under stated training and licence rules (`survey.md` section 5, revised in round 1).
  Gaps: the Search-R1 and HippoRAG licences were not opened; G-A1 reproduces no published figure; G-A2 deviates from its published 70B recipe and is a reduced reproduction.
- C7 met: recommendations in `survey.md` section 6, recorded as decisions in the master plan under the delegation; round 1 revised the metrics (EM as context), cost classes (class R wording, training rule), heavy class (ghost recipes), exam (multi-annotator gold rule, no test-label statistics) and forecast rows.
- C8 met: forecast per phase 01-06, about 24.6 USD in total (projection, revised in round 1), inside the authorized 25 USD, with the candidates' and the exam core's money ring-fenced and a priority and drop order (`survey.md` section 7).

## Candidate learnings
