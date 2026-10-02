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

## Results

- C1 met: `ledger.json` has 284 records, one per (system, benchmark, setting, metric) figure, each with a `source` locator; the script finds zero required fields empty without a `why_missing` note (measured).
  Gaps: `licence` is null in 256 records (explained, mostly "not checked"); `code_url` and `weights_public` partly come from known repository names and were not opened this phase.
- C2 met: every Scope family has at least one record (`survey.md` section 1); graph without an LLM has only LinearRAG, with answer-accuracy figures in an unknown setting.
- C3 met: 282 of 284 records verified ledger-wide, and 81 of 81 figures cited in the maps and ghost list (100.0 percent, measured by the script); 15 records are self-reported model-card figures, labelled.
- C4 met: `survey.md` section 2 defines every metric string in the ledger with unit, setting and relation to Full Support; no table mixes metrics or settings.
  Gap: no published system reports Full Support at any budget, and HippoRAG's recall@5 definition is assumed, not verified.
- C5 met: maps for HotpotQA, MuSiQue, 2Wiki and MultiHop-RAG (`survey.md` section 4), one table per (setting, metric); costs reported where the sources give them, otherwise labelled projection.
  Gaps: no full-corpus retrieval figure for MuSiQue, no class-A figure on MultiHop-RAG, and leaderboards for MuSiQue and QASPER were not read.
- C6 met: ghosts G-L answerai-colbert-small-v1, G-R bge-reranker-v2-m3 (J-strong), G-A1 Search-R1 7B and G-A2 HippoRAG 2 with an 8B model, each with figure, reason and projected cost (`survey.md` section 5).
  Gaps: the G-L model card and licence, and the Search-R1 and HippoRAG licences, were not opened; G-A2 deviates from its published 70B recipe and is a reduced reproduction.
- C7 met: recommendations in `survey.md` section 6, recorded as decisions in the master plan under the delegation.
- C8 met: forecast per phase 01-06, about 21.6 USD in total (projection), inside the authorized 25 USD, with a priority and drop order (`survey.md` section 7).

## Candidate learnings
