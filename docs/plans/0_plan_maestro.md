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
| 00 | Map the state of the art: which retrieval and RAG strategies are published (papers and GitHub), on which corpora, with which metrics, at which cost, and propose the ghosts | - | integrated |
| 01 | Build the harness from the old project's code and pass the reproduction gate: every recorded figure reproduced exactly | - | integrated |
| 02 | Reproduce the ghosts chosen in 00 on the terrain, with cost columns, beside the four old systems and the Phase 17 union pool under J-strong | 00, 01 | ready locally |
| 03 | First candidate: weight-free fusion with source diversity (charter candidate 3, light class), on the laptop | 02 | ready locally |
| 04 | Second candidate: judge over a pooled bag (charter candidate 1, rerank class), G-L in the pool | 02 | ready locally |
| 05 | Third candidate: multi-seed convergent hop with a refined query (charter candidate 2, light class) | 02 | pending |
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
| 2026-10-02 | RunPod spending cap: 25 USD in total for the project so far, per-pod session cap 15 USD; any spend beyond it needs the author | the author authorizes money up front; the forecast was about 25-55 USD | the recommended 60 USD cap |
| 2026-10-02 | Phases 00 and 01 run in parallel | 01 depends only on the old project's code | running them in sequence |
| 2026-10-02 | The reading budget, the cost classes, an LLM in the heavy class's online loop, offline LLM work in the light class and the exam corpus are decided after Phase 00 | results must be comparable with the state of the art first | deciding them now |
| 2026-10-02 | Metrics: Full Support @2,048 of record; beside it, from the same rankings, all-gold at k units (FS@2, @5, @20), per-question gold share at 5, and nDCG@10 plus Recall@100 on HotpotQA dev; answer EM only for the agentic ghost, as context, not a fidelity check (its corpus, retriever and questions differ from the paper's; revised in Phase 00 review round 1); no reader in the tournament (Phase 00 `survey.md` section 6) | decided by the agent under delegation; computable at no cost from depth-100 rankings; HotpotQA dev with the 5.23M corpus is BEIR-HotpotQA's exact setting, the one place our figures meet the literature's directly | answer EM/F1 for every system, which needs a reader and its cost |
| 2026-10-02 | Reading budget: 2,048 tokens of record, 1,024 and 4,096 reported beside it | decided by the agent under delegation; 2,048 is the old line's budget and MultiHop-RAG's reader budget; the other two are free from the same rankings | a single budget |
| 2026-10-02 | Cost classes: L light (no generative LLM, models up to 1B, online up to 0.1 s per question on one RTX 4090 or 2 s on the laptop CPU, offline up to 2 GPU-h per million units); R rerank (L plus zero-shot cross-encoders up to 1B over at most the top-100, online up to 1 s per question, no text generation; a decoder up to 1B used as a pointwise scorer counts as a cross-encoder); zero-shot means general-purpose models whose training mix may include a terrain train split, never our dev questions or the exam's data (training rule, Phase 00 `survey.md` section 5); A LLM (open models up to 8B on one 24 GB GPU, online up to 30 s per question, offline up to 5 GPU-h per corpus) | decided by the agent under delegation; the bounds hold the measured J-strong (0.45 to 0.54 s) in R and GLiNER and the Entity Hop in L, and keep each class-A run inside the 25 USD cap | two classes, cheap and expensive, without bounds |
| 2026-10-02 | The heavy class A admits an open LLM of up to 8B in the online loop, no closed APIs, run on MuSiQue and MultiHop-RAG in full and on a preregistered 1,000-question HotpotQA dev subsample; ghosts Search-R1 (agentic, run through a batched vLLM driver checked against the repository script) and HippoRAG 2 with Llama-3.1-8B and GritLM-7B (LLM-built graph, MultiHop-RAG only, run last); ghost weights need a licence without a non-commercial restriction | decided by the agent under delegation; an agentic ghost is needed either way, and an open 7B agent costs about 2.0 USD on MuSiQue and MultiHop-RAG, 4,672 questions, with a batched driver (projection; the repository's one-at-a-time script would cost about 17.5 USD for 5,672 questions, those two plus the 1,000-question subsample, projection) | no LLM online; 70B models or closed APIs |
| 2026-10-02 | No offline generative LLM work in the light class; encoder models up to 1B (GLiNER, encoders) are allowed offline; LLM graph building and LLM document expansion belong to class A | decided by the agent under delegation; the light class must stay cheap on any corpus; LLM graph building costs about 2 USD per 28,000 units here (projection) and grows with the corpus | allowing offline LLM work in the light class |
| 2026-10-02 | Exam corpus: QASPER, pooled across papers declared as a new setting, plus a within-paper control; before the exam is opened its spec fixes, from the paper and the train and dev splits only, the gold rule for questions with several annotators, the unit rule for tables and figures and the paragraph-count estimate; no statistic is computed from test labels before the exam runs | decided by the agent under delegation; native paragraph gold and, per the QASPER paper itself (a published dataset-level figure, not computed by us), 55.5 % of answerable text-evidence questions have multi-paragraph evidence; LegalBench-RAG is single-document with 500-character span gold, so Full Support collapses to single-evidence recall | LegalBench-RAG (the earlier provisional recommendation) |
| 2026-10-02 | RunPod forecast for Phases 01-06: about 24.5 USD (projection, revised in Phase 00 review rounds 1 and 2), soft caps per phase 0.5 / 11.5 (7.9 core, 3.6 optional) / 2.5 / 2.5 / 2.5 / 5.0 (2.4 core, 2.6 optional with G-R2 and G-A2) USD; the candidates' 7.5 USD and the exam core's 2.4 USD ring-fenced, no ghost may spend them, LLM-built graph ghost last; inside the authorized 25 USD total and 15 USD per pod; priority and drop order in Phase 00 `survey.md` section 7 | decided by the agent under delegation; fits the author's cap with 0.5 USD unallocated after correcting the agent's throughput and the ColBERT index cost | running HippoRAG 2 on MuSiQue or NV-Embed-v2 on FullWiki inside the cap |
| 2026-10-03 | Phases 03-05 are charter candidates 3, 1 and 2, in that order | taken by the author on the agent's proposal after Phase 02: the zero-cost laptop candidate first protects the money left (10.22 USD inside the 25 USD authorization against the 9.9 USD reserve, deviation 02.1); the pooled judge, the strongest old system (j-union) on two of three sets, gets the reserve's pod money | the pooled judge first |
| 2026-10-04 | Phase 03 sends no entrant to the exam: F3 advances against G-L but has no win and two losses against RRF3, and loses to the best light-class system on all three sets (written by code, `fase-03-fusion/results.md`); RRF3's own verdict is context only, never an entrant; C7 met on the author's billing-explorer reading (deviation 03.1) | the selection rule frozen in the Phase 03 spec, applied by the outcome code to the run files | entering F3 or promoting RRF3 after seeing the figures, which would choose by looking at the results |
| 2026-10-04 | Phase 04 sends no entrant to the exam: `j-rrf4` advances against G-R but loses to `j-rrf3` on MultiHop-RAG (written by code, `fase-04-judge/results.md`); `j-rrf3`'s own verdict is context only, never an entrant; no candidate has entered the exam so far, and Phase 05 stays as planned; Phase 04 spent 0.33 USD (author's invoice), leaving 9.89 USD inside the 25 USD authorization (derived; the C9 closing `clientBalance` reading agrees, delta 0.33 USD) | the selection rule frozen in the Phase 04 spec, applied by the outcome code to the run files | entering `j-rrf4` or promoting `j-rrf3` after seeing the figures, which would choose by looking at the results |

## Open decisions (after Phase 00)

None: the six items listed here until 2026-10-02 (metrics, reading budget, cost classes, LLM in the heavy class, offline LLM work in the light class, exam corpus) are decided in the table above.

## Milestones

- After 00: a map of the field and its costs, useful on its own.
- After 02: honest baselines, the ghosts beside the old systems, on the terrain.
- After 06: the tournament's answer, with the exam result, ready to write up.
- After 07: the game.
