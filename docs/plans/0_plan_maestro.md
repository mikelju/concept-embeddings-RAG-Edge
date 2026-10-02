# Master plan - concept-embeddings-RAG-Edge

Status: approved
Approved by the author: 2026-10-02

## Vision

The goal is a cheap, fast way to find the evidence a question needs, in any text corpus, better than what is used today.
The project runs a tournament of retrieval strategies on a shared harness.
Every candidate is scored on quality and cost together and measured against the literature's best recipes, reproduced here.
Quality is the share of questions whose full evidence enters a fixed reading budget; cost is time and money per question, offline and online separately.
The project succeeds if a strategy sits on the quality-cost frontier against the ghosts on the terrain, and keeps its place in a single run on a locked exam corpus.

Starting brief: `docs/refs/successor_project_charter.md` (decisions) and `docs/refs/successor_project_handover.md` (state, numbers, know-how).

## Out of scope

- Plans, diagrams and images, until a labelled set for them exists.
- Numeric aggregates over tables: retrieval finds the table, code computes.
- Set queries that need many passages from all over a corpus (QUEST, GlobalQA): a separate question with its own spec, if the author raises it.
- The old project's paper, its venue process, and the concept-space line.

## Phases

| Phase | Goal in one sentence | Depends on | Status |
|---|---|---|---|
| 00 | Map the state of the art: which retrieval and RAG strategies are published (papers and GitHub), on which corpora, with which metrics, at which cost, and propose the ghosts | - | pending |
| 01 | Build the harness from the old project's code and pass the reproduction gate: every recorded figure reproduced exactly | - | ready locally |
| 02 | Reproduce the ghosts chosen in 00 on the terrain, with cost columns, beside the four old systems and the Phase 17 union pool under J-strong | 00, 01 | pending |
| 03 | First candidate, chosen after 02 from the charter's candidates or the pool of ideas | 02 | pending |
| 04 | Second candidate | 02 | pending |
| 05 | Third candidate | 02 | pending |
| 06 | Exam: one locked run of the chosen candidates and the ghosts on the exam corpus, under a rule frozen before it opens | 03-05 | pending |
| 07 | Game: a tower-defense page that reads the measured results as aggregate performance, never computing anything | 06 | pending |

Statuses: pending, spec in review, spec approved, in progress, blocked, ready locally, integrated.
Only the author adds, removes or reorders phases; the agent proposes.
Phases 03-05 are placeholders: their number and content are decided after Phase 02.

### What Phase 00 delivers

- A ledger of published systems: method, cost class, corpora, metric and unit, reading budget or k, reported figures with source (table or page), latency and cost where reported, code and weights availability.
- Two budgets covered: cheap and fast (dense models of several sizes, hybrid lexical + dense, learned sparse, late interaction, zero-shot rerankers, weight-free fusion, graph methods without an LLM) and expensive and slow (iterative retrieval with an LLM, agentic RAG, LLM rerankers, LLM-built graphs).
- A map of quality against cost, and a shortlist of ghosts, one per cost class, with a reason for each.
- A recommendation, for the author to decide, on the open questions below: metrics comparable with the literature, reading budget, cost classes and the exam corpus.
- No code and no measurement; every figure cited from its source and checked against it.

## Global criteria

- Research protocol in `.agents/skills/research-protocol/SKILL.md`: terrain open for choosing, exam locked and run once, nothing fitted on the corpus a strategy is used on.
- Every result carries its cost beside its score, offline and online, time and money, each value labelled measured, derived, exploratory, interpretation or projection.
- Every new system is compared with the ghost of its class and with the project's best system so far, on the same questions, with an exact paired test.
- The old project's artifacts are read in place, read-only, checked against their digests; every new artifact is write-once, digested and in a non-executing format.
- Money is spent only with the author's prior approval and a stated cap; processing a corpus with an LLM must stay far below thousands of euros.
- Code, docs and results in English; the author gets a Spanish summary at each approval and full translations on request.
- The repository is public: no credentials, raw corpora or `data/` artifacts are committed.

## Strategic decisions

| Date | Decision | Reason | Alternative discarded |
|---|---|---|---|
| 2026-09-30 | A new repository for a broader question: quality at a given cost, in any text corpus | the old question was answered within bounds; the new one needs different rules (unit as a variable, rerankers and learned models admitted) | extending the old line |
| 2026-09-30 | Start from the literature's best, reproduced, as the bar | the old line compared against a weak control for 16 phases | Dense + BM25 as the control |
| 2026-09-30 | Terrain HotpotQA, MuSiQue, MultiHop-RAG; exam QASPER or LegalBench-RAG, run once | choose on open corpora, test on one nobody looked at | choosing on the exam |
| 2026-09-30 | A tower-defense game, waves by difficulty, built last from measured results | aggregate reading of the tournament | a rally; per-question mechanics |
| 2026-10-02 | SDD Lite as the working framework, with the old `research-protocol` and `remote-gpu` knowledge ported as skills | one flow, one guard; keep the domain know-how | copying the old commands, guard and templates |
| 2026-10-02 | Public repository `mikelju/concept-embeddings-RAG-Edge` | the author's choice | private |
| 2026-10-02 | Docs and code in English; Spanish translations on request, not mirrored | a mirror doubles every edit and drifts | a Spanish mirror of every document |
| 2026-10-02 | The author delegates every further decision, including spec approvals and the open decisions below, to the agent; the only limits are the token quota and the RunPod budget, which the author authorizes up front; merging PRs stays the author's | a test of fully autonomous execution under SDD Lite | stopping at each spec for approval |
| 2026-10-02 | Phases 00 and 01 run in parallel | 01 depends only on the old project's code | running them in sequence |
| 2026-10-02 | The reading budget, the cost classes, an LLM in the heavy class's online loop, offline LLM work in the light class and the exam corpus are decided after Phase 00 | results must be comparable with the state of the art first | deciding them now |

## Open decisions (after Phase 00)

1. Metrics: Full Support at a token budget, plus which literature metrics (for example Recall@k, end-to-end answer accuracy) make results comparable.
2. Reading budget of record: 2,048 tokens, or a small set.
3. Cost classes and their bounds.
4. Whether the heavy class admits an LLM in the online loop (depends on budget); agentic RAG is needed as a ghost either way.
5. Whether offline LLM work is allowed in the light class (default no; depends on budget).
6. The exam corpus: QASPER or LegalBench-RAG (provisional recommendation: LegalBench-RAG, for its published whole-corpus retrieval figures).

## Milestones

- After 00: a map of the field and its costs, useful on its own.
- After 02: honest baselines, the ghosts beside the old systems, on the terrain.
- After 06: the tournament's answer, with the exam result, ready to write up.
- After 07: the game.
