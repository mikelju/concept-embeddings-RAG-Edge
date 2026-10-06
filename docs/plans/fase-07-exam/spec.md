# Phase 07 - Exam on QASPER: specification

Status: draft, for the author's approval (not frozen); decisions D1-D7 resolved by the author in chat on 2026-10-06; train/dev figures measured 2026-10-06
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
Sources: the QASPER paper (Dasigi et al., NAACL 2021), the dataset card of the Hugging Face dataset `allenai/qasper` (repository sha `fdc9d82`, read 2026-10-06), and the train-dev archive (D7).
Test figures are published figures, never computed by us; train and dev figures are measured on the archive.
- Archive: `qasper-train-dev-v0.3.tgz` from the AllenAI S3 bucket `qasper-dataset`, the `_URL_TRAIN_DEV` of the official loader `qasper.py` in `allenai/qasper`; downloaded 2026-10-06T16:34:19Z, sha256 `a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a` (measured), stored under `data/phase07/train-dev/` (gitignored).
  It holds only `qasper-train-v0.3.json` (sha256 `9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4`), `qasper-dev-v0.3.json` (sha256 `2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93`) and a README, which confirms that test ships in a separate archive (measured).
  Nothing from the test archive was downloaded, opened or computed.
- Totals: 5,049 questions over 1,585 NLP papers (dataset card).
- Train: 888 papers, 2,593 questions; dev: 281 papers, 1,005 questions (dataset card table, and measured on the archive: equal).
- Test: 416 papers, 1,451 questions (derived from the card's totals minus its train and dev rows; the paper's own split table is still to be quoted under C1).
- Annotators per question (measured): train 1.03 on average (2,511 with one, 82 with two); dev 1.76 (261 with one, 729 with two, 15 with three).
  Test is annotated like dev (paper, to be quoted under C1), so dev is the closer guide to test.
- Evidence is a set of paragraphs, or table and figure captions (strings prefixed `FLOAT SELECTED:`), chosen per annotator.
- Multi-evidence share, measured under the unit rule and the gold rule below (a question counts as multi-evidence when every annotator set that maps fully has two or more units): train 734 of 2,172 in-scope questions (33.8 %), dev 210 of 901 (23.3 %).
  The draft's 55.5 % (paper figure via the master plan, 2026-10-02) is not what these rules give; it is still to be found and quoted under C1, and the measured shares stand for this spec.
  Full Support therefore differs from recall at the budget on about a quarter to a third of the questions, less than the draft assumed (interpretation).
- Answer types: extractive, abstractive, yes/no, unanswerable.

## What stays fixed
- Code and settings of every system as run on the terrain (Phase 02 to 06 commits), with no constant changed: RRF k = 60, depth 100, J-strong `BAAI/bge-reranker-v2-m3` at the pinned revision and G-R settings, G-R2 `Qwen/Qwen3-Reranker-0.6B` at its pinned revision with the Phase 06 D10 memory fix, G-A1 with the Phase 02 driver, checkpoint, prompt and decoding, G-L `answerai-colbert-small-v1` with the Phase 02 D17 build settings, GLiNER and the entity hop as in Phase 04.
- Metrics as in Phases 02-06: FS @1,024 / @2,048 (of record) / @4,096, FS@2/5/20 units, gold share at 5; token counts with the terrain's tokenizer, in batches (known trap).
- Paired test: exact McNemar on per-question FS@2,048, two-sided, alpha 0.05 per comparison, no correction (stated as a limitation).
- Nothing is fitted on QASPER; no system, constant, unit rule or gold rule is changed after the test archive is opened.

## Rules fixed before the exam opens (from the paper and train/dev only)
- **Units**: one unit per full-text paragraph of a paper, title "paper title - section name"; one unit per table or figure, text its caption as released (the release has captions, not table contents), with the `FLOAT SELECTED:` prefix kept so evidence strings match.
  Numeric content of tables stays out of scope (master plan).
- **Gold mapping**: an evidence string maps to a unit by exact match after whitespace normalization.
  Measured on train and dev with a first count script kept outside the repo (2026-10-06): train 4,062 of 4,209 evidence strings match (96.5 %), dev 2,679 of 2,808 (95.4 %); every `FLOAT SELECTED:` string matches (386 and 253).
  The non-matching strings seen are section headings (for example `Datasets`), not paragraphs, so they are not units (measured on four dev cases, interpretation for the rest); an annotator set with such a string is not fully mapped (76 sets on train, 52 on dev).
  One example checked by hand: dev paper 1912.01214, question "what are the pivot-based baselines?", two evidence strings, both full paragraphs of section "Experiments ::: Main Results".
  C2 repeats these counts with the tested loader before the test archive is downloaded.
- **Gold rule with several annotators**: a question is in scope when at least one annotator marks it answerable with a non-empty evidence set that maps fully to units; Full Support holds when the context covers every evidence unit of at least one such annotator (decided by the author, D3, 2026-10-06; QASPER's own evaluator also takes the best-matching annotator).
  In scope under this rule (measured): train 2,172 of 2,593 questions (83.8 %), dev 901 of 1,005 (89.7 %); so the 1,451 test questions are an upper bound for the in-scope count.
- **Paragraph-count estimate** (measured on train plus dev, 2026-10-06): 68,634 units over 1,169 papers, 58.71 units per paper (train 60.18, mean paragraphs 52.80 plus captions 7.38; dev 54.07); paragraphs average 70-73 words.
  Projected exam corpus: 58.71 x 416 test papers = 24,424 units (derived), replacing the draft's 62,400 (150 units per paper, a guess).
  Per question, the mean units of its own paper is 58.9 (measured, question-weighted), used for the within-paper control.
- **Settings**: pooled, every question searched against all units of the 416 test papers, declared a new setting (QASPER is natively within one paper); within-paper control, every question searched only in its own paper's units, for BM25, Dense and J-strong over all of that paper's units (decided by the author, D4, 2026-10-06), reported beside the pooled figures, never deciding the verdict.
- **Metric of record**: FS@2,048 in the pooled setting; single- and multi-evidence questions reported as a breakdown, computed at scoring time.
- **Baseline rule**: the literature control is `j-rrf3` (hybrid RRF of Dense, BM25 and G-L, then a cross-encoder over its top 100), the recipe reproduced in Phase 04; no published figure exists for pooled QASPER at a token budget, and the native QASPER evidence figure (Evidence-F1) is another metric, context only.

## Line-up (decided by the author, D1 option B, 2026-10-06)
The own systems are named from the terrain figures; the ghosts are those already reproduced on this harness.
Own entrants `j-rrf4` (R and A) and `rrf4` (L); control `j-rrf3`; ghosts G-L, G-R, G-R2 and G-A1; `p10-b` and `p14` are context rows only, in the table and never in the verdict.

| System | Role | Class | Terrain reason (Phase 06 `results.md`, measured) | Where it runs | Marginal cost (projection) |
|---|---|---|---|---|---:|
| `j-rrf4` | own entrant | R (and A by "or cheaper") | best own system on HotpotQA (5,950) and MuSiQue (831) | laptop + 4090 | 0.11 USD judge + 0.03 USD GLiNER |
| `rrf4` | own entrant | L | best own light system on HotpotQA (5,291); weight-free | laptop + 4090 (G-L, GLiNER shared) | 0 beyond shared |
| `j-rrf3` | literature control, own context | R | best own system on MultiHop-RAG (873) | laptop + 4090 | 0.11 USD |
| `p10-b` | context | L | best own light system on MultiHop-RAG (587); 0.5 / 0.5 Dense and BM25 | laptop only | 0 |
| `p14` | context (author's request) | L | relevance hop mixed with an alpha fitted on HotpotQA (Phase 04 DJ3) | laptop (hop and GLiNER entities shared) | about 0 |
| G-L | ghost | L | class L bar on every set | 4090 | in the pod's fixed 0.30 USD |
| G-R | ghost | R | class R bar on HotpotQA | 4090 | 0.11 USD |
| G-R2 | ghost | R | class R bar on MuSiQue and MultiHop-RAG | 4090 | 0.49 USD |
| G-A1 | ghost | A | class A bar; best system on MuSiQue (1,101) | A100 80 GB (D2), last and gated (D6) | 1.54 USD |

`p14` is added at the author's request (D1, 2026-10-06) as a context row only: its alpha was fitted on HotpotQA (Phase 04 DJ3), and fitted weights do not travel between kinds of text (research protocol), so the protocol keeps it out of the verdict.
Its marginal cost is about 0 USD on the laptop, sharing the entity hop and the pod's GLiNER entities (projection).
G-A2 is left out: it does not fit the money (above).
Contamination check before the archive is opened (C3), under the training rule and Phase 06 E1: QASPER questions, answers or annotations in the training mix of BGE-small, answerai-colbert-small-v1, bge-reranker-v2-m3, Qwen3-Reranker-0.6B, GLiNER and Search-R1 make that system an upper reference; S2ORC paper text possibly seen is labelled, not excluding.

## Locked single run and verdict rule (frozen on approval)
- Before opening: spec frozen; train/dev counts and the QASPER loader tested on train/dev only; one frozen commit for every run, its hash in `plan.md`.
- Opening: download the test archive once, record its sha256; the loader writes units, questions and a separate gold file with their digests; no statistic is computed from the gold file until every ranking is written and digested.
- Runs: laptop items, then the 4090 pod, then G-A1 if its gate passes (D6, money above).
  A technical rerun with identical settings is allowed only before scoring and only for a crash that wrote no ranking (Phase 06 deviation 06.4), recorded as a deviation; nothing is rerun after scoring.
- Scoring: one call of the outcome code reads the gold file, writes `results.json` and `results.md`; states and claims are written by code.
- Verdict per class c in L, R, A: the bar is the strongest measured ghost of class c or a cheaper class by pooled FS@2,048; the own entrant of class c or cheaper (L `rrf4`; R and A `j-rrf4`) is paired against it with exact McNemar.
  State `win` when wins > losses and p < 0.05, `loss` when losses > wins and p < 0.05, `tie` otherwise, `not run` when the bar's ghost was not run.
  "Keeps its place on the exam in class c" is written only where the state is `win` (decided by the author, D5, 2026-10-06); `tie` is reported as not losing, never as a win.
- Context, not verdict: `j-rrf4` against `j-rrf3` (does the hop help outside the terrain), every own system against every ghost, the within-paper control.

## Cost estimate
Every USD is time x rate at the prices read on 2026-10-06T14:10:40Z through the RunPod API (`gpuTypes`, Secure Cloud, measured): RTX 4090 0.74, A40 0.49, RTX A6000 0.53, A100-SXM4-80GB 1.59, A100 80GB PCIe 1.59 USD/h.
Sizes: 1,451 test questions (published figure, an upper bound for the in-scope count); 24,424 units (derived: 58.71 units per paper measured on train and dev, times 416 test papers).
Laptop CPU (0 USD by assumption, as in earlier phases): BM25 build and search, Dense BGE-small corpus encoding (projection: under 1 h), question encoding, entity hop, RRF, `p10-b` and `p14`, within-paper BM25 and Dense, scoring.
GLiNER stays on the pod: at the laptop's measured 0.558 paragraphs/s, 24,424 units would take about 12.2 h (derived), against 2.5 min on a 4090.

| Item | Basis | Label | h | USD |
|---|---|---|---:|---:|
| 4090 setup, uploads, model downloads | Phase 06 pods: `uv sync` and CUDA check 4-5 min (measured), plus downloads | projection | 0.25 | 0.19 |
| GLiNER over 24,424 units | 162.178 paragraphs/s on a 4090 (measured, old project) | derived | 0.04 | 0.03 |
| G-L encode, index and search | 20.06 s for 27,989 units on a 4090 (measured, Phase 02), scaled to 17.5 s; search 0.1 s per question (projection) | projection | 0.05 | 0.03 |
| C3 checks (fidelity and determinism) | Phase 06 C3 took 23-25 s (measured) | projection | 0.05 | 0.04 |
| J-strong, pooled: G-R, `j-rrf3`, `j-rrf4` (300 pairs per question, overlap not removed) | 268 pairs/s, G-R on MultiHop-RAG, 4090 (measured, Phase 02) | derived | 0.45 | 0.33 |
| J-strong, within-paper control (58.9 pairs per question: the mean units of the question's paper, measured on train and dev) | same pace | derived | 0.09 | 0.07 |
| G-R2, pooled | 1.634 s per question, MuSiQue, 4090 (measured, Phase 06) | derived | 0.66 | 0.49 |
| 4090 download | | projection | 0.05 | 0.04 |
| **4090 pod, base** | sum | derived | 1.64 | 1.21 |
| G-A1 on A100 80 GB: setup 21 min and server start 7.5 min (measured, Phase 06 G-A1 pod), 1,451 questions at 1.100 s (measured, HotpotQA), download 0.05 h (projection) | | derived | 0.97 | 1.54 |

Each item's projection is its base + 25 % (Phase 06 contingency); its hard cut is 1.5 times its projection (derived): 4090 pod 1.51 projection, 2.27 cut; G-A1 1.92 projection, 2.89 cut.
Only the GLiNER, G-L and within-paper lines depend on the corpus size; the draft's 62,400-unit guess had them at 0.08, 0.04 and 0.17 USD.
G-A1's 1.100 s per question is the slowest measured pace (HotpotQA, beside a 37 GB index on the same GPU); MuSiQue and MultiHop-RAG ran at about 0.38 and 0.26 s per question in the turn loop (derived, Phase 02 F7).
G-A1 runs on an A100 80 GB, not an A40 (decided by the author, D2, 2026-10-06): Phase 06 measured about 65 GB of GPU memory for it (vLLM share 0.46 of 80 GB plus the retrieval server, deviation 06.4), and the A40 has 48 GB and is untested.

### Total for the decided set against the cap
Cap = 6.4847 USD balance (measured 2026-10-06T14:02:25Z) - 1.0 USD buffer = 5.48 USD (derived); it is re-fixed from the `clientBalance` read before the first pod.
Billing tail: 0.14 USD beyond time x rate (measured gap, Phase 06 C8), added to every worst case.

| Set | Projection | Margin at projection | Sum of hard cuts + tail | Margin at cuts |
|---|---:|---:|---:|---:|
| B. Decided (D1, D2): everything, G-A1 on A100 80 GB | 3.44 | 2.04 | 5.30 | 0.18 |
| C. Gate not passed: 4090 pod only, G-A1 `not run (cost gate)` | 1.51 | 3.97 | 2.41 | 3.07 |

All figures derived from the item table.
Stress case from Phase 06's failure modes (judges, G-R2 and G-A1's generation 2.7 times slower, as G-R2's pace was against its assumption in deviation 06.2; one out-of-memory restart repeating each pod's setup, as in 06.1 and 06.4; the 0.14 USD tail): 4090 pod 2.90 USD, G-A1 3.49 USD, 6.54 USD in all, 1.06 USD past the cap; without G-A1 3.04 USD, 2.44 USD under it (derived).
In that case the hard cuts stop each stage first (4090 at 2.27, G-A1 at 2.89), and the gate below keeps G-A1 out when the first stage overspends.

### Run order and the G-A1 gate (decided by the author, D6, 2026-10-06)
The 4090 pod runs first; G-A1 runs last and only through this gate.
After the 4090 pod is terminated, read `clientBalance`; G-A1 is launched only if that measured balance minus the 1.0 USD buffer covers G-A1's hard cut (1.5 x its projection, 2.89 USD at today's figures, re-projected before launch).
Otherwise G-A1 is recorded `not run (cost gate)` and class A's verdict is `not run`.
At today's figures the gate needs a balance of at least 3.89 USD, so a first stage of at most 2.60 USD balance delta from 6.4847 USD (derived); this is above the 4090 pod's 2.27 USD cut, so a 4090 stage inside its cut passes unless its billing shows more than 0.33 USD beyond its cut (derived).

## Money and stopping rule
- Authorization: the remaining balance, no top-up (D13 (c)); the exam runs only if the projection of the approved set fits with margin.
- Before each pod: read `clientBalance` and `myself { pods }`, record the offered rate, re-project the item; close with a reading at least 2 h after the last termination.
- Hard cut per item at 1.5 times its projection, never past the cap; a probe on the first 50 questions of each GPU stage projects the full stage, and a stage that projects past its cut stops and is recorded `not run`.
- Out of GPU memory once: one rerun with identical settings and a memory-only fix (06.4), or one resize to the next tier if the re-projection fits the cut; otherwise `not run`.
- Spend recorded as time x rate (derived), balance delta (measured) and invoice when the author copies it.

## Scope
In, in order: train/dev counts and rules (laptop); contamination check; QASPER loader and units; test archive opened once; laptop rankings; 4090 pod; G-A1 pod if gated in; scoring and results by code.
Out: any new candidate or setting; G-A2; `p14` or `p10-b` in the verdict (context rows only, D1); a reader or answer metric; numeric table content; any rerun after scoring.

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
- QASPER paragraphs (70-73 words on average, measured on train and dev) are close to MultiHop-RAG's units in length, so the judges' measured paces hold (assumed; the stress case covers 2.7 times slower).
- G-A1 on an A100 80 GB with a G-L index of 24,424 units fits in memory, as the Phase 06 run did with 27,989 MultiHop-RAG units (interpretation).
- GLiNER's entity types, built for Wikipedia and news, find usable entities in NLP papers (interpretation; the hop may add little, which the context comparison shows).

## Risks
- A single run can fail for a technical reason; the rerun rule above allows one identical rerun before scoring, nothing after.
- The terrain wins on HotpotQA carry in-domain caveats (Phase 06); the exam is the first test without them.
- G-A1 may be slower than projected on QASPER; its cut and probe stop it, and the gate keeps it out when the first stage overspends.

## Anti-goals
- No new candidate, constant or variant; no look at any test figure before scoring; no reader.

## Decisions (resolved by the author in chat, 2026-10-06)
1. **D1 Line-up**: option B - own entrants `j-rrf4` (R/A) and `rrf4` (L), control `j-rrf3`, ghosts G-L, G-R, G-R2 and G-A1; `p10-b` and `p14` are context rows only, never in the verdict; `p14` added at the author's request, its HotpotQA-fitted alpha keeping it out of the verdict.
2. **D2 G-A1 hardware**: A100 80 GB, not A40 - about 65 GB of GPU memory measured in Phase 06 (deviation 06.4); the A40 has 48 GB and is untested.
3. **D3 Gold rule**: any annotator's full evidence set, as recommended in the draft.
4. **D4 Within-paper control**: BM25, Dense and J-strong over all of the paper's units, as recommended in the draft (0.07 USD base after the train/dev count).
5. **D5 Claim rule**: "keeps its place" only on `win`, as recommended in the draft.
6. **D6 Money order**: 4090 pod first, G-A1 last and gated on the measured balance after the first stage minus the 1.0 USD buffer covering G-A1's hard cut; otherwise `not run (cost gate)`.
7. **D7 Downloads**: train and dev authorized now and downloaded from the official AllenAI source on 2026-10-06 (10.8 MB archive, above); the test split only after the spec is frozen; nothing is downloaded, opened or computed on test before then.
