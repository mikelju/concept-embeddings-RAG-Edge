# Phase 07 - Exam on QASPER: specification

Status: draft, for the author's approval (not frozen)
Master plan: `../0_plan_maestro.md` (exam corpus decision of 2026-10-02; author decision D13 (c) of 2026-10-06 in `../fase-06-rivals/plan.md`).
Starting point: Phase 06 `results.md` (literature bar on the terrain), Phase 02 to 06 plans for measured paces and costs.

## Objective
Run once, on a corpus nobody looked at while choosing, the project's best own systems and the literature ghosts, under a rule frozen before the corpus is opened.
This is the tournament's answer: does the best own system of each cost class keep its place against the strongest ghost of that class outside the terrain.
Said first and plainly: no candidate of Phases 03-05 advanced, so no system enters on its own claim.
The own systems below are named by the author from the terrain figures (Phase 06 `results.md`), which is choosing on the terrain, as the protocol allows; nothing is chosen from the exam.
The light-class bar is weak: G-L is weaker than the light systems already on disk, and MDR, the strongest light-class literature recipe, is not reproduced (Phase 05, Phase 06 `results.md`).
The class A bar stands on G-A1 alone: G-A2 (HippoRAG 2) does not fit the money (Phase 06 probe projected 7.10 USD on 27,989 MultiHop-RAG units, measured projection, deviation 06.3).

## Question
On QASPER test, pooled across papers, what Full Support @2,048 do the entrants and the ghosts reach, at what offline and online cost, and, paired on the same questions, does each class's own entrant win, tie or lose against the strongest measured ghost of its class or a cheaper one?

## The exam corpus, from the paper only
Figures in this section are cited from the QASPER paper (Dasigi et al., NAACL 2021) and still need to be checked against its text before freezing; none was computed by us.
- 5,049 questions over 1,585 NLP papers; split by paper: train 888 papers and 2,593 questions, dev 281 and 1,005, test 416 and 1,451.
- Evidence is a set of paragraphs, or table and figure captions (strings prefixed `FLOAT SELECTED:`), chosen per annotator; dev and test questions may have several annotators.
- 55.5 % of answerable text-evidence questions have multi-paragraph evidence (paper figure, master plan 2026-10-02), so Full Support does not collapse to recall at the budget (research protocol, known traps).
- Answer types: extractive, abstractive, yes/no, unanswerable.
- The release ships train and dev in one archive and test with the evaluator in another, so everything this spec needs before the exam opens comes from the train-dev archive alone (to be confirmed when downloading, with the author's permission, D7).

## What stays fixed
- Code and settings of every system as run on the terrain (Phase 02 to 06 commits), with no constant changed: RRF k = 60, depth 100, J-strong `BAAI/bge-reranker-v2-m3` at the pinned revision and G-R settings, G-R2 `Qwen/Qwen3-Reranker-0.6B` at its pinned revision with the Phase 06 D10 memory fix, G-A1 with the Phase 02 driver, checkpoint, prompt and decoding, G-L `answerai-colbert-small-v1` with the Phase 02 D17 build settings, GLiNER and the entity hop as in Phase 04.
- Metrics as in Phases 02-06: FS @1,024 / @2,048 (of record) / @4,096, FS@2/5/20 units, gold share at 5; token counts with the terrain's tokenizer, in batches (known trap).
- Paired test: exact McNemar on per-question FS@2,048, two-sided, alpha 0.05 per comparison, no correction (stated as a limitation).
- Nothing is fitted on QASPER; no system, constant, unit rule or gold rule is changed after the test archive is opened.

## Rules fixed before the exam opens (from the paper and train/dev only)
- **Units**: one unit per full-text paragraph of a paper, title "paper title - section name"; one unit per table or figure, text its caption as released (the release has captions, not table contents), with the `FLOAT SELECTED:` prefix kept so evidence strings match.
  Numeric content of tables stays out of scope (master plan).
- **Gold mapping**: an evidence string maps to a unit by exact match after whitespace normalization; the match rate is measured on train and dev before freezing (C2), and one example is checked by hand (research protocol).
- **Gold rule with several annotators**: a question is in scope when at least one annotator marks it answerable with a non-empty evidence set that maps fully to units; Full Support holds when the context covers every evidence unit of at least one such annotator (recommended, D3; QASPER's own evaluator also takes the best-matching annotator).
- **Paragraph-count estimate**: mean units per paper on train plus dev (laptop count, C2) times 416 test papers gives the pooled corpus size used to size the pods and fix the cost projection; until counted, this draft uses 150 units per paper, 62,400 units, as an upper projection (not measured).
- **Settings**: pooled, every question searched against all units of the 416 test papers, declared a new setting (QASPER is natively within one paper); within-paper control, every question searched only in its own paper's units, for BM25, Dense and J-strong over all of that paper's units (recommended, D4), reported beside the pooled figures, never deciding the verdict.
- **Metric of record**: FS@2,048 in the pooled setting; single- and multi-evidence questions reported as a breakdown, computed at scoring time.
- **Baseline rule**: the literature control is `j-rrf3` (hybrid RRF of Dense, BM25 and G-L, then a cross-encoder over its top 100), the recipe reproduced in Phase 04; no published figure exists for pooled QASPER at a token budget, and the native QASPER evidence figure (Evidence-F1) is another metric, context only.

## Line-up (open: D1; recommendation)
The own systems are named from the terrain figures; the ghosts are those already reproduced on this harness.

| System | Role | Class | Terrain reason (Phase 06 `results.md`, measured) | Where it runs | Marginal cost (projection) |
|---|---|---|---|---|---:|
| `j-rrf4` | own entrant | R (and A by "or cheaper") | best own system on HotpotQA (5,950) and MuSiQue (831) | laptop + 4090 | 0.11 USD judge + 0.08 USD GLiNER |
| `rrf4` | own entrant | L | best own light system on HotpotQA (5,291); weight-free | laptop + 4090 (G-L, GLiNER shared) | 0 beyond shared |
| `j-rrf3` | literature control, own context | R | best own system on MultiHop-RAG (873) | laptop + 4090 | 0.11 USD |
| `p10-b` | context | L | best own light system on MultiHop-RAG (587); 0.5 / 0.5 Dense and BM25 | laptop only | 0 |
| G-L | ghost | L | class L bar on every set | 4090 | in the pod's fixed 0.30 USD |
| G-R | ghost | R | class R bar on HotpotQA | 4090 | 0.11 USD |
| G-R2 | ghost | R | class R bar on MuSiQue and MultiHop-RAG | 4090 | 0.49 USD |
| G-A1 | ghost | A | class A bar; best system on MuSiQue (1,101) | A40 (recommended, D2) or A100 | 0.89 USD on A40, 1.55 USD on A100 |

`p14` is left out (recommended, D1): its relevance hop mixes with an alpha fitted on HotpotQA (Phase 04 DJ3), and fitted weights do not travel between kinds of text (research protocol).
G-A2 is left out: it does not fit the money (above).
Contamination check before the archive is opened (C3), under the training rule and Phase 06 E1: QASPER questions, answers or annotations in the training mix of BGE-small, answerai-colbert-small-v1, bge-reranker-v2-m3, Qwen3-Reranker-0.6B, GLiNER and Search-R1 make that system an upper reference; S2ORC paper text possibly seen is labelled, not excluding.

## Locked single run and verdict rule (frozen on approval)
- Before opening: spec frozen; train/dev counts and the QASPER loader tested on train/dev only; one frozen commit for every run, its hash in `plan.md`.
- Opening: download the test archive once, record its sha256; the loader writes units, questions and a separate gold file with their digests; no statistic is computed from the gold file until every ranking is written and digested.
- Runs: laptop items, then the 4090 pod, then G-A1 if its gate passes (money below).
  A technical rerun with identical settings is allowed only before scoring and only for a crash that wrote no ranking (Phase 06 deviation 06.4), recorded as a deviation; nothing is rerun after scoring.
- Scoring: one call of the outcome code reads the gold file, writes `results.json` and `results.md`; states and claims are written by code.
- Verdict per class c in L, R, A: the bar is the strongest measured ghost of class c or a cheaper class by pooled FS@2,048; the own entrant of class c or cheaper (L `rrf4`; R and A `j-rrf4`) is paired against it with exact McNemar.
  State `win` when wins > losses and p < 0.05, `loss` when losses > wins and p < 0.05, `tie` otherwise, `not run` when the bar's ghost was not run.
  "Keeps its place on the exam in class c" is written only where the state is `win` (recommended, D5); `tie` is reported as not losing, never as a win.
- Context, not verdict: `j-rrf4` against `j-rrf3` (does the hop help outside the terrain), every own system against every ghost, the within-paper control.

## Cost estimate
Every USD is time x rate at the prices read on 2026-10-06T14:10:40Z through the RunPod API (`gpuTypes`, Secure Cloud, measured): RTX 4090 0.74, A40 0.49, RTX A6000 0.53, A100-SXM4-80GB 1.59, A100 80GB PCIe 1.59 USD/h.
Sizes: 1,451 test questions (paper figure, an upper bound for the in-scope count); 62,400 units (projection, replaced by the train/dev estimate).
Laptop CPU (0 USD by assumption, as in earlier phases): BM25 build and search, Dense BGE-small corpus encoding (projection: under 1 h), question encoding, entity hop, RRF, within-paper BM25 and Dense, scoring.
GLiNER stays on the pod: at the laptop's measured 0.558 paragraphs/s, 62,400 units would take about 31 h (derived), against 6.4 min on a 4090.

| Item | Basis | Label | h | USD |
|---|---|---|---:|---:|
| 4090 setup, uploads, model downloads | Phase 06 pods: `uv sync` and CUDA check 4-5 min (measured), plus downloads | projection | 0.25 | 0.19 |
| GLiNER over 62,400 units | 162.178 paragraphs/s on a 4090 (measured, old project) | derived from a projected size | 0.11 | 0.08 |
| G-L encode, index and search | 20.06 s for 27,989 units on a 4090 (measured, Phase 02), scaled; search 0.1 s per question (projection) | projection | 0.05 | 0.04 |
| C3 checks (fidelity and determinism) | Phase 06 C3 took 23-25 s (measured) | projection | 0.05 | 0.04 |
| J-strong, pooled: G-R, `j-rrf3`, `j-rrf4` (300 pairs per question, overlap not removed) | 268 pairs/s, G-R on MultiHop-RAG, 4090 (measured, Phase 02) | derived | 0.45 | 0.33 |
| J-strong, within-paper control (up to 150 pairs per question) | same pace | derived from a projected size | 0.23 | 0.17 |
| G-R2, pooled | 1.634 s per question, MuSiQue, 4090 (measured, Phase 06) | derived | 0.66 | 0.49 |
| 4090 download | | projection | 0.05 | 0.04 |
| **4090 pod, base** | sum | derived | 1.85 | 1.37 |
| G-A1 on A100: setup 21 min and server start 7.5 min (measured, Phase 06 G-A1 pod), 1,451 questions at 1.100 s (measured, HotpotQA), download | | derived | 0.97 | 1.55 |
| G-A1 on A40: same setup, generation 2.9 times slower (memory bandwidth ratio, interpretation) | | projection | 1.81 | 0.89 |

Each item's projection is its base + 25 % (Phase 06 contingency); its hard cut is 1.5 times its projection.
G-A1's 1.100 s per question is the slowest measured pace (HotpotQA, beside a 37 GB index on the same GPU); MuSiQue and MultiHop-RAG ran at about 0.38 and 0.26 s per question in the turn loop (derived, Phase 02 F7).

### Which sets fit the cap
Cap = 6.4847 USD balance (measured 2026-10-06T14:02:25Z) - 1.0 USD buffer = 5.48 USD (derived); it is re-fixed from the `clientBalance` read before the first pod.
Billing tail: 0.14 USD beyond time x rate (measured gap, Phase 06 C8), added to every worst case.

| Set | Projection | Margin at projection | Sum of hard cuts + tail | Margin at cuts |
|---|---:|---:|---:|---:|
| A. Recommended: everything, G-A1 on A40 | 2.82 | 2.67 | 4.37 | 1.11 |
| B. Everything, G-A1 on A100 (measured hardware) | 3.64 | 1.85 | 5.60 | -0.12 |
| C. No G-A1 (4090 pod only) | 1.71 | 3.78 | 2.70 | 2.78 |
| D. No G-A1, no G-R2 | 1.10 | 4.39 | 1.79 | 3.69 |
| E. Minimal: no G-A1, G-R2, within-paper control or hop systems | 0.65 | 4.83 | 1.12 | 4.36 |

All figures derived from the item table.
Stress case from Phase 06's failure modes (judges and G-R2 2.7 times slower, as G-R2's pace was against its assumption in deviation 06.2; one out-of-memory restart repeating each pod's setup, as in 06.1 and 06.4; the 0.14 USD tail): 4090 pod 3.23 USD, G-A1 on A100 3.50 USD, 6.87 USD in all, past the cap; without G-A1 3.37 USD, 2.11 USD under it (derived).
So the 4090 pod runs first and G-A1 runs only when the 4090 pod's time x rate plus 0.14 USD plus G-A1's hard cut stays within the cap (on A40: 4090 spend at most 3.67 USD, above its own 2.56 USD cut, so it always passes; on A100: at most 2.44 USD).

## Money and stopping rule
- Authorization: the remaining balance, no top-up (D13 (c)); the exam runs only if the projection of the approved set fits with margin.
- Before each pod: read `clientBalance` and `myself { pods }`, record the offered rate, re-project the item; close with a reading at least 2 h after the last termination.
- Hard cut per item at 1.5 times its projection, never past the cap; a probe on the first 50 questions of each GPU stage projects the full stage, and a stage that projects past its cut stops and is recorded `not run`.
- Out of GPU memory once: one rerun with identical settings and a memory-only fix (06.4), or one resize to the next tier if the re-projection fits the cut; otherwise `not run`.
- Spend recorded as time x rate (derived), balance delta (measured) and invoice when the author copies it.

## Scope
In, in order: train/dev counts and rules (laptop); contamination check; QASPER loader and units; test archive opened once; laptop rankings; 4090 pod; G-A1 pod if gated in; scoring and results by code.
Out: any new candidate or setting; G-A2; `p14`; a reader or answer metric; numeric table content; any rerun after scoring.

## Acceptance criteria
Frozen on approval.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | The paper figures above are checked against the paper text and corrected visibly before freezing | quotes with page or table in `plan.md` |
| C2 | Before the test archive is downloaded: mean units per paper, evidence match rate and annotator counts measured on train and dev only, one example checked by hand, and the cost projection re-fixed from them | `plan.md` record, its commit earlier than the test archive's download time |
| C3 | Before opening: licence and QASPER training-data status of every system's models recorded with source and date; each labelled ghost, own or upper reference | `plan.md` |
| C4 | The test archive is downloaded once; its sha256, the units, questions and gold files and their digests are recorded; no gold-derived figure exists before every ranking is digested | manifests and timestamps under `data/phase07/` |
| C5 | Every system in the approved line-up has a depth-100 ranking (G-A1: evidence lists) for every in-scope question, pooled, and the within-paper control for its systems, written once, with manifests pinning code commit, models and revisions, hardware, `costPerHr` and seconds | manifests, digests in `plan.md` |
| C6 | `results.json` and `results.md`, written by code in one scoring call, give every metric of record with a label, offline and online cost per system, exact McNemar per comparison, and the per-class verdict under the frozen rule | tests on a hand-built example for the verdict; the generated page |
| C7 | Money: spend at most the cap, no item past its hard cut, `myself { pods }` empty at the close, three labelled spend figures | readings in `plan.md` |
| C8 | `npm run check` passes | exit 0 |

## Assumptions
- QASPER paragraphs are close to MultiHop-RAG's units in length, so the judges' measured paces hold (assumed; the stress case covers 2.7 times slower).
- Search-R1 and its 7B model fit on a 48 GB A40 beside a small G-L index (assumed; the A100 run held a 37 GB index beside the model).
- GLiNER's entity types, built for Wikipedia and news, find usable entities in NLP papers (interpretation; the hop may add little, which the context comparison shows).

## Risks
- A single run can fail for a technical reason; the rerun rule above allows one identical rerun before scoring, nothing after.
- The terrain wins on HotpotQA carry in-domain caveats (Phase 06); the exam is the first test without them.
- G-A1 on an untested GPU may be slower than projected; its cut and probe stop it.

## Anti-goals
- No new candidate, constant or variant; no look at any test figure before scoring; no reader.

## Open decisions (for the author; recommendation first)
1. **D1 Line-up**: own entrants `j-rrf4` (R) and `rrf4` (L), `j-rrf3` as control, `p10-b` as zero-cost context; ghosts G-L, G-R, G-R2, G-A1; `p14` out.
   Cheaper: drop G-A1 (set C, class A not run) or also G-R2 (set D).
2. **D2 G-A1 hardware**: A40 48 GB at 0.49 USD/h (set A, fits even at the hard cuts); alternative A100 80 GB, measured with G-A1 but past the cap at the cuts, so gated (set B).
3. **D3 Gold rule**: any annotator's full evidence set (recommended); alternatives: union of annotators, or first annotator.
4. **D4 Within-paper control**: BM25, Dense and J-strong over all of the paper's units; alternative: no control (saves 0.17 USD).
5. **D5 Claim rule**: "keeps its place" only on `win`; alternative: also on `tie`.
6. **D6 Money order**: 4090 pod first, G-A1 last and gated as above.
7. **D7 Downloads**: the train-dev archive now (for C2) and the test archive after freezing, both from the official QASPER release, sizes stated when asking.
