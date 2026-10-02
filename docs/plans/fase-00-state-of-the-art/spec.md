# Phase 00 - State of the art: specification

Status: approved (frozen)
Approved: 2026-10-02, by the agent under the author's delegation (master plan, strategic decisions)
Master plan: `../0_plan_maestro.md`

## Goal

A verified map of where multi-evidence retrieval and RAG stand today, in quality and in cost, so that every later phase is measured against the literature's best and its results are comparable with it.
It is needed first because the old line's lesson is that a weak control wastes phases (charter section 1).

## Scope

In:
- Published retrieval and RAG strategies (papers and public repositories, up to 2026-10) evaluated on multi-evidence or retrieval benchmarks:
  HotpotQA (distractor and FullWiki), MuSiQue, 2WikiMultiHopQA, MultiHop-RAG, BEIR subsets, and the exam candidates QASPER and LegalBench-RAG.
- Two budgets. Cheap and fast: dense models of several sizes, hybrid lexical + dense, learned sparse (SPLADE family), late interaction (ColBERT family, BGE-M3), zero-shot cross-encoder rerankers, weight-free fusion, graph methods without an LLM.
  Expensive and slow: iterative retrieval with an LLM (IRCoT and successors), agentic RAG and search agents, LLM rerankers, LLM-built graphs (GraphRAG, HippoRAG and successors).
- For each system: metric and unit, retrieval depth or reading budget, corpus setting (distractor, pooled, full), reported figures, latency and cost where reported, offline cost, code and weights availability and licence.
- Inputs already in hand: the old project's `docs/refs/bibliografia.md`, `docs/plans/corpora_survey_2026-09-30.md`, `docs/plans/phase_16/16.pre_spec_notes.md`, `docs/paper/tables/published-*.md` (55 verified rows).

Out:
- Any code, download of models or data, or measurement.
- Systems trained on the exam candidates' test data, and closed API-only systems except as labelled context.

## Acceptance criteria

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | `ledger.json` lists every system found, one record per (system, benchmark, setting, metric) figure, with the fields in Scope and a `source` locator (paper and table, figure or section; or repository file) | the file parses; a script counts records with any required field empty: zero, or the field is `null` with a `why_missing` note |
| C2 | Every family named in Scope has at least one record, or a sentence in `survey.md` saying why none qualifies | family coverage table in `survey.md` |
| C3 | Figures are checked against their source: each record carries `verified: true` (read in the source this phase) or `verified: false` with the reason; at least 80 % of the figures used by C5 and C6 are `verified: true` | count in `survey.md`, by script |
| C4 | `survey.md` defines every metric the ledger uses, its unit and setting, and how it relates to Full Support at a token budget; it never compares figures across metrics or settings | read; each metric of the ledger has a definition row |
| C5 | A quality-cost map per terrain benchmark: for each system with figures in a comparable setting, its best figure beside its cost class and reported or estimated cost (estimates labelled projection) | tables in `survey.md` |
| C6 | A ghost shortlist: at least one per cost class, each with public code and weights, its published figure, why it is the strongest reproducible choice in its class, and a projected GPU time and cost to run it on the terrain (labelled projection) | section in `survey.md` |
| C7 | A recommendation, with its reason, for each open decision of the master plan (metrics, reading budget, cost classes, LLM in the heavy class, offline LLM work in the light class, exam corpus), recorded as a decision in the master plan under the delegation | master plan decisions table |
| C8 | A RunPod spending forecast for Phases 01-06, per phase, labelled projection, with a recommended cap | section in `survey.md` |

## Constraints and risks

- Web sources can be wrong, retracted or ambiguous about setting; a figure whose setting cannot be determined is kept with `setting: unknown` and used in no comparison.
- Papers report recall at k passages, EM/F1 of answers, or nDCG; none reports Full Support over whole paragraphs at a token budget (old handover fact 16). Comparability is the core risk: C4 forbids cross-metric comparisons.
- Self-reported figures from repositories are labelled as such.
- No money is spent in this phase.

## Open decisions

None: the author delegated them (master plan, 2026-10-02); C7 records what is decided.
