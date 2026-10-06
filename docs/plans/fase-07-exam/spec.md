# Phase 07 - Exam on QASPER: specification

Status: draft, for the author's approval (not frozen); decisions D1-D7 resolved by the author in chat on 2026-10-06, except D6's 4.17 USD gate threshold; that threshold and D8 (pooled query names the paper) pending the author's approval at freeze; train/dev figures measured 2026-10-06
Master plan: `../0_plan_maestro.md` (exam corpus decision of 2026-10-02; author decision D13 (c) of 2026-10-06 in `../fase-06-rivals/plan.md`).
Starting point: Phase 06 `results.md` (literature bar on the terrain), Phase 02 to 06 plans for measured paces and costs.

## Objective
Run once, on a corpus nobody looked at while choosing, the project's best own systems and the literature ghosts, under a rule frozen before the corpus is opened.
This is the tournament's answer: does the best own system of each cost class keep its place against the strongest ghost of that class outside the terrain.
Said first and plainly: no candidate of Phases 03-05 advanced, so no system enters on its own claim.
The own systems below are named by the author from the terrain figures (Phase 06 `results.md`), which is choosing on the terrain, as the protocol allows; nothing is chosen from the exam.
The light-class bar is weak: G-L is weaker than the light systems already on disk, and MDR, the strongest light-class literature recipe, is not reproduced (Phase 05, Phase 06 `results.md`).
The class A bar stands on G-A1 alone: G-A2 (HippoRAG 2) does not fit the money (Phase 06 probe projected 7.10 USD on 27,989 MultiHop-RAG units, measured projection, deviation 06.3).
Two weaknesses of the exam itself, said plainly:
- Test has more annotators per question than dev (multiple references on 98 % of test questions against 74 % on dev, published), so under the any-annotator gold rule more test questions count as single-evidence than on dev (interpretation).
  Full Support on test then sits closer to recall at the budget than the dev figures suggest; this is the same for every system, but it weakens the exam as a test of the hop.
- The master plan chose QASPER on the paper's figure of 55.5 % multi-paragraph answers; on the same denominator the released train and dev give 32.3 % and 30.1 % (measured, below), and the cause of the gap is not known.

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
- Test: 416 papers, 1,451 questions (papers derived from the card's totals minus its train and dev rows; questions published, paper Section 4.1, "Data splits": "This resulted in 2,593, 1,005, and 1,451 questions in the three sets"; the paper has no split table and gives no paper counts per split).
- Annotators per question (measured): train 1.03 on average (2,511 with one, 82 with two); dev 1.76 (261 with one, 729 with two, 15 with three).
  Multiple references are published as "98% in test, and 74% in validation" (paper Section 4.1, published); dev's 744 of 1,005 with two or more (74.0 %, measured) matches, so test has more references per question than dev (interpretation: more chances of a single-unit annotator set under the gold rule).
- Evidence is a set of paragraphs, or table and figure captions (strings prefixed `FLOAT SELECTED:`), chosen per annotator.
- Multi-evidence share, measured under the unit rule and the gold rule below (a question counts as multi-evidence when every annotator set that maps fully has two or more units): train 734 of 2,172 in-scope questions (33.8 %), dev 210 of 901 (23.3 %).
  The paper's figure (Section 3, "Evidence types", published): "Among the answerable questions with text-only evidence", "55.5% of the answers have multi-paragraph evidence".
  Recomputed on that denominator (answers that are answerable with non-empty, text-only evidence), the released v0.3 train and dev give 640 of 1,981 (32.3 %) and 403 of 1,340 (30.1 %) (measured), so the gap is not a denominator difference; its cause is not known (perhaps an earlier annotation pool or data version, interpretation), and the measured shares stand for this spec.
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
  A unit's indexed text is `title. body`, as on the terrain (`src/edge_rag/corpus.py`), and its token count for the budget is taken over that whole text, title included, the same for every system; whether the terrain's inherited token counts included the title is not verified here and C2 records it.
- **Gold mapping**: an evidence string maps to a unit of the question's own paper only, by exact match after whitespace normalization; a string equal to a unit of another paper never maps.
  Measured on train and dev with a first count script kept outside the repo (2026-10-06): train 4,062 of 4,209 evidence strings match (96.5 %), dev 2,679 of 2,808 (95.4 %); every `FLOAT SELECTED:` string matches (386 and 253).
  The non-matching strings seen are section headings (for example `Datasets`), not paragraphs, so they are not units (measured on four dev cases, interpretation for the rest); an annotator set with such a string is not fully mapped (76 sets on train, 52 on dev).
  One example checked by hand: dev paper 1912.01214, question "what are the pivot-based baselines?", two evidence strings, both full paragraphs of section "Experiments ::: Main Results".
  C2 repeats these counts with the tested loader before the test archive is downloaded.
- **Gold rule with several annotators**: a question is in scope when at least one annotator marks it answerable with a non-empty evidence set that maps fully to units; Full Support holds when the context covers every evidence unit of at least one such annotator (decided by the author, D3, 2026-10-06; QASPER's own evaluator also takes the best-matching annotator).
  In scope under this rule (measured): train 2,172 of 2,593 questions (83.8 %), dev 901 of 1,005 (89.7 %); so the 1,451 test questions are an upper bound for the in-scope count.
  Every system ranks all 1,451 test questions; ranking never reads the gold file, and the in-scope filter is applied only at scoring, by the outcome code.
- **Paragraph-count estimate** (measured on train plus dev, 2026-10-06): 68,634 units over 1,169 papers, 58.71 units per paper (train 60.18, mean paragraphs 52.80 plus captions 7.38; dev 54.07); paragraphs average 70-73 words.
  Projected exam corpus: 58.71 x 416 test papers = 24,424 units (derived), replacing the draft's 62,400 (150 units per paper, a guess).
  Per question, the mean units of its own paper is 58.9 (measured, question-weighted), used for the within-paper control.
- **Settings**: pooled, every question searched against all units of the 416 test papers, declared a new setting (QASPER is natively within one paper); within-paper control, every question searched only in its own paper's units, for BM25, Dense and J-strong over all of that paper's units (decided by the author, D4, 2026-10-06), reported beside the pooled figures, never deciding the verdict.
- **Pooled query (D8, pending the author's approval at freeze)**: in the pooled setting every system's query is `<paper title>. <question>`, the same string for every system (G-A1 included), fixed before the exam opens; the within-paper control keeps the question alone.
  D8 makes the query the same only at the input: G-A1 writes its own search queries during its turns and may drop the title, so its retrieval queries can differ from the other systems' (interpretation).
  Reason: annotators saw only the title and abstract and wrote questions such as "What baselines do they use?", which name no paper; pooled over 416 papers such a question has no single right answer, every system may score near the floor and most pairs may tie.
  Naming the paper is the realistic setting of a user asking about a paper they know.
  Hand check (dev, read only, no score computed, 2026-10-06): the first question of each of the 15 dev papers with the lowest ids (1503.00841 to 1606.02891); 12 of 15 cannot be answered without knowing the paper (for example "What baseline model is used?", "How was this data collected?", "What evaluation metric do they use?"), and 3 name enough of their topic to point to it (medical Wikipedia jargon, LDA spam detection, conversation flow in debates) (measured by reading, a judgment per question).
  In all 15 the title is specific to its paper, and all 1,169 train and dev titles are distinct after lower-casing (measured), so the title disambiguates; it also matches the title prefix of every unit of its paper.
  Expected effect (interpretation): every system rises off the floor, lexical and dense retrieval first pick the paper and then the paragraph, the pooled setting moves toward the within-paper control, and the room left for the hop may shrink.
  Rejected alternative: the question alone, as QASPER writes it; rejected because most questions are ambiguous across papers, so the exam would measure guessing the paper rather than retrieving its evidence.
- **Metric of record**: FS@2,048 in the pooled setting; single- and multi-evidence questions reported as a breakdown, computed at scoring time.
- **Baseline rule**: the literature control is `j-rrf3` (hybrid RRF of Dense, BM25 and G-L, then a cross-encoder over its top 100), the recipe reproduced in Phase 04; no published figure exists for pooled QASPER at a token budget, and the native QASPER evidence figure (Evidence-F1) is another metric, context only.

## Line-up (decided by the author, D1 option B, 2026-10-06)
The own systems are named from the terrain figures; the ghosts are those already reproduced on this harness.
Own entrants `j-rrf4` (R and A) and `rrf4` (L); control `j-rrf3`; ghosts G-L, G-R, G-R2 and G-A1; `p10-b` and `p14` are context rows only, in the table and never in the verdict.

| System | Role | Class | Terrain reason (Phase 06 `results.md`, measured) | Where it runs | Marginal cost (projection) |
|---|---|---|---|---|---:|
| `j-rrf4` | own entrant | R (and A by "or cheaper") | best own system on HotpotQA (5,950) and MuSiQue (831); its HotpotQA lead carries the in-domain caveat (BGE-small fine-tuned on HotpotQA train), and it lost to `j-rrf3` on MultiHop-RAG, the set most like pooled QASPER (843 against 873, 21 wins, 51 losses, p 0.00053, Phase 04) | laptop + 4090 | 0.11 USD judge + 0.03 USD GLiNER |
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
An upper-reference label never changes the bar, the pairing or the state, which code writes as for any system; it is printed beside every row and comparison that uses the labelled model.
`bge-reranker-v2-m3` sits in G-R, `j-rrf3` and `j-rrf4`: if C3 labels it, all three carry the label, and the R and A comparisons that pair `j-rrf4` with G-R are marked "upper reference on both sides"; a `win` by a labelled entrant is written "keeps its place (upper reference: <model>)".

## Locked single run and verdict rule (frozen on approval)
- Before opening: spec frozen; train/dev counts and the QASPER loader tested on train/dev only; one frozen commit for every run, its hash in `plan.md`.
- Opening: download the test archive once, record its sha256; the loader writes units, questions and a separate gold file with their digests; no statistic is computed from the gold file until every ranking is written and digested.
- Runs: laptop items, then the 4090 pod, then G-A1 if its gate passes (D6, money above).
  A technical rerun with identical settings is allowed only before scoring and only for a crash that wrote no ranking (Phase 06 deviation 06.4), recorded as a deviation; nothing is rerun after scoring.
- Scoring: one call of the outcome code reads the gold file, writes `results.json` and `results.md`; states and claims are written by code.
- Verdict per class c in L, R, A: the bar is the strongest measured ghost of class c or a cheaper class by pooled FS@2,048; the own entrant is paired against it with exact McNemar.
  The entrants are hard-coded in the outcome code before the exam opens, L `rrf4`, R and A `j-rrf4`, and never chosen from exam figures; context rows (`j-rrf3`, `p10-b`, `p14`) never enter the verdict, even when they beat the entrant.
  Each R and A verdict line also prints, on the same line, the state of `j-rrf4` against `j-rrf3` (the literature control, which beat `j-rrf4` on MultiHop-RAG in Phase 04); this is report only and never changes the verdict.
  Bar tie-break, deterministic: higher FS@2,048, then the cheaper class (L before R before A), then the fixed order G-L, G-R, G-R2, G-A1.
  Partial runs: a system with fewer than 1,451 test rankings (for example an item stopped by its cut or skipped by its probe) is `not run`; it is never scored on the subset it wrote, and its files are kept only as a record.
  One rule for missing systems: class c's state is `not run` when its entrant is `not run` or when any line-up ghost of class c or a cheaper class is `not run`, because the project cannot claim to beat the literature in a class whose entrant or strongest ghost was never fully measured (Phase 06 author rationale); so with G-A1 not run, class A is `not run` and no cheaper ghost stands in for it.
  Otherwise the state is `win` when wins > losses and p < 0.05, `loss` when losses > wins and p < 0.05, `tie` otherwise.
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
Billing tail: 0.14 USD beyond time x rate (measured gap, Phase 06 C8), counted once per pod in every worst case.

| Set | Projection | Margin at projection | Sum of hard cuts + tails | Margin at cuts |
|---|---:|---:|---:|---:|
| B. Decided (D1, D2): everything, G-A1 on A100 80 GB | 3.44 | 2.04 | 5.44 (2.27 + 2.89 + 2 x 0.14) | 0.04 |
| C. Gate not passed: 4090 pod only, G-A1 `not run (cost gate)` | 1.51 | 3.97 | 2.41 (2.27 + 0.14) | 3.07 |

All figures derived from the item table.
Set B's margin at the cuts is thin, 0.04 USD; the gate below is what keeps the worst case inside the cap.
Stress case from Phase 06's failure modes (judges, G-R2 and G-A1's generation 2.7 times slower, as G-R2's pace was against its assumption in deviation 06.2; one out-of-memory restart repeating each pod's setup, as in 06.1 and 06.4; a 0.14 USD tail per pod): 4090 pod 3.04 USD, G-A1 3.63 USD, 6.68 USD in all (sum before rounding), 1.20 USD past the cap; without G-A1 3.04 USD, 2.44 USD under it (derived).
In that case the hard cuts stop each stage first (4090 at 2.27, G-A1 at 2.89), and the gate below keeps G-A1 out when the first stage overspends.

### Run order and the G-A1 gate (decided by the author, D6, 2026-10-06)
The 4090 pod runs first; G-A1 runs last and only through this gate.
After the 4090 pod is terminated, read `clientBalance`; G-A1 is launched only if that measured balance minus the 1.0 USD buffer, minus the 4090 pod's 0.14 USD billing tail (Phase 06 C8, which a reading taken right after termination may not yet show), minus G-A1's own 0.14 USD tail, covers G-A1's hard cut (1.5 x its projection, 2.89 USD at today's figures, re-projected before launch at the offered rate).
Otherwise G-A1 is recorded `not run (cost gate)` and class A's verdict is `not run` (verdict rule above).
At today's figures the gate needs a balance of at least 4.17 USD (1.0 + 0.14 + 0.14 + 2.89; G-A1's own tail in this sum pending the author's confirmation at freeze), so a first stage of at most 2.31 USD balance delta from 6.4847 USD (derived); this is 0.04 USD above the 4090 pod's 2.27 USD cut, so a 4090 stage that bills past its cut by more than 0.04 USD keeps G-A1 out (derived).

## Money and stopping rule
- Authorization: the remaining balance, no top-up (D13 (c)); the exam runs only if the projection of the approved set fits with margin.
- Before each pod: read `clientBalance` and `myself { pods }`, record the offered rate, re-project the item; close with a reading at least 2 h after the last termination.
- Hard cut per pod at 1.5 times its projection, never past the cap.
- Items inside a pod run in a fixed order, hard-coded before the exam opens: verdict items first, then the within-paper control and the context rows.
  4090 pod: setup, GLiNER, G-L, C3 checks, J-strong for G-R and `j-rrf4`, G-R2, then J-strong for `j-rrf3`, then the within-paper J-strong control, then download; the A100 pod runs G-A1 alone.
- The probe is per item: each per-question item first runs its first 50 questions (a corpus pass such as GLiNER or the G-L encoding, its first 1,000 units) and projects its full run from that pace.
  An item whose projection does not fit the pod's remaining cut (the cut minus the time x rate already spent) is skipped and recorded `not run (cut)`; later items in the order are still probed, so one slow item does not stop the others.
- Each pod terminates itself at its cut: a watchdog started at pod creation, its cut divided by the offered rate, 3.07 h for the 4090 (2.27 / 0.74) and 1.82 h for the A100 80 GB (2.89 / 1.59) at today's figures (derived), recomputed from the rate read before launch.
  It extends Phase 06's hard-cut watchdog in `scripts/pod_gr2.sh` (`HARD_CUT_USD`, `SPENT_S`, `POD_COST_PER_HR`; a 30 s sampler that writes `STOP_CUT` and kills the work once elapsed seconds x rate reach the cut), which stops the work but does not terminate the pod (measured, reading the script).
  At the cut it stops the work, flushes the outputs already written to the laptop-readable volume or download location, and then calls `runpodctl remove pod $RUNPOD_POD_ID` (or the equivalent RunPod API call); whether the pod's own credentials allow that call is not verified yet (interpretation), which the backup covers.
- Backup: a laptop timer per pod, the same length plus 5 min, calls `podTerminate` if the pod still exists when it expires.
  It covers the in-pod watchdog only partly: if the Windows laptop sleeps, hibernates or loses network, the timer does not fire on time, so the laptop is kept awake and on mains power while a pod lives (risk below).
- Files already written are kept as a record; a system cut short is `not run` under the partial-run rule above, never scored on a subset.
- No A100 80 GB available, or an offered rate above 1.59 USD/h: G-A1 is re-projected at the offered rate and runs only if it still passes the gate; otherwise it is recorded `not run (no GPU / cost gate)`.
  No swap to another GPU type unless it has at least 80 GB of GPU memory and passes the gate at its own rate.
- Out of GPU memory once: one rerun with identical settings and a memory-only fix (06.4), or one resize to the next tier if the re-projection fits the cut; otherwise `not run`.
- Spend recorded as time x rate (derived), balance delta (measured) and invoice when the author copies it.

## Scope
In, in order: train/dev counts and rules (laptop); contamination check; QASPER loader and units; test archive opened once; laptop rankings; 4090 pod; G-A1 pod if gated in; scoring and results by code.
Out: any new candidate or setting; G-A2; `p14` or `p10-b` in the verdict (context rows only, D1); a reader or answer metric; numeric table content; any rerun after scoring.

## Acceptance criteria
Frozen on approval.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | The paper figures above are checked against the paper text and corrected visibly before freezing (done 2026-10-06: Sections 3 and 4.1 quoted above, arXiv 2105.03011) | quotes with section in `plan.md` |
| C2 | Before the test archive is downloaded: mean units per paper, evidence match rate (within the question's own paper), annotator counts, in-scope counts and the single- versus multi-evidence breakdown (train 734 of 2,172, dev 210 of 901) measured on train and dev only with the tested loader, one single- and one multi-evidence example checked by hand, whether the terrain's token counts include the unit title recorded, and the cost projection re-fixed from them | `plan.md` record, its commit earlier than the test archive's download time |
| C3 | Before opening: licence and QASPER training-data status of every system's models recorded with source and date; each labelled ghost, own or upper reference | `plan.md` |
| C4 | The test archive is downloaded once; its sha256, the units, questions and gold files and their digests are recorded; no gold-derived figure exists before every ranking is digested | manifests and timestamps under `data/phase07/` |
| C5 | Every system in the approved line-up has a depth-100 ranking (G-A1: evidence lists) for every one of the 1,451 test questions, written without reading the gold file, pooled (with the D8 query if approved), and the within-paper control for its systems, written once, with manifests pinning code commit, models and revisions, hardware, `costPerHr` and seconds | manifests, digests in `plan.md` |
| C6 | `results.json` and `results.md`, written by code in one scoring call, give every metric of record with a label, offline and online cost per system, exact McNemar per comparison, the single- and multi-evidence breakdown, the in-scope filter applied only here, and the per-class verdict under the frozen rule | tests on hand-built examples for the verdict: a context row (`p14`, `p10-b` or `j-rrf3`) beating the entrant leaves the verdict unchanged; G-A1 not run gives class A `not run`; a bar tie resolved by the tie-break; an upper-reference label printed without changing the state; a system with fewer than 1,451 rankings is `not run` and never scored on its subset; an entrant `not run` and a needed ghost `not run` each give that class `not run`; the `j-rrf4` against `j-rrf3` state printed on the R and A verdict lines without changing them; the generated page |
| C7 | Money: spend at most the cap, no pod past its hard cut, `myself { pods }` empty at the close, three labelled spend figures | readings in `plan.md` |
| C8 | `npm run check` passes | exit 0 |
| C9 | Before any pod is launched: the in-pod self-termination (watchdog flushes outputs, then calls the remove command at its cut) and the laptop backup timer (calls `podTerminate` at its expiry) are both tested on a stub, with a short cut, a fake remove command and a fake API endpoint, no real pod and no spend | test output and commit in `plan.md`, earlier than the first pod's creation time |

## Assumptions
- QASPER paragraphs (70-73 words on average, measured on train and dev) are close to MultiHop-RAG's units in length, so the judges' measured paces hold (assumed; the stress case covers 2.7 times slower).
- G-A1 on an A100 80 GB with a G-L index of 24,424 units fits in memory, as the Phase 06 run did with 27,989 MultiHop-RAG units (interpretation).
- GLiNER's entity types, built for Wikipedia and news, find usable entities in NLP papers (interpretation; the hop may add little, which the context comparison shows).

## Risks
- A single run can fail for a technical reason; the rerun rule above allows one identical rerun before scoring, nothing after.
- The terrain wins on HotpotQA carry in-domain caveats (Phase 06); the exam is the first test without them.
- G-A1 may be slower than projected on QASPER; its cut and probe stop it, and the gate keeps it out when the first stage overspends.
- If the in-pod self-termination fails (for example the pod's credentials cannot remove it) while the Windows laptop sleeps or is offline, neither stop fires on time and the pod keeps billing past its cut; keeping the laptop awake and checking `myself { pods }` after each expected end reduces but does not remove this risk.

## Anti-goals
- No new candidate, constant or variant; no look at any test figure before scoring; no reader.

## Decisions (resolved by the author in chat, 2026-10-06)
1. **D1 Line-up**: option B - own entrants `j-rrf4` (R/A) and `rrf4` (L), control `j-rrf3`, ghosts G-L, G-R, G-R2 and G-A1; `p10-b` and `p14` are context rows only, never in the verdict; `p14` added at the author's request, its HotpotQA-fitted alpha keeping it out of the verdict.
2. **D2 G-A1 hardware**: A100 80 GB, not A40 - about 65 GB of GPU memory measured in Phase 06 (deviation 06.4); the A40 has 48 GB and is untested.
3. **D3 Gold rule**: any annotator's full evidence set, as recommended in the draft.
4. **D4 Within-paper control**: BM25, Dense and J-strong over all of the paper's units, as recommended in the draft (0.07 USD base after the train/dev count).
5. **D5 Claim rule**: "keeps its place" only on `win`, as recommended in the draft.
6. **D6 Money order**: 4090 pod first, G-A1 last and gated on the measured balance after the first stage minus the 1.0 USD buffer and the 4090 pod's 0.14 USD tail covering G-A1's hard cut (tail added by the author, 2026-10-06); otherwise `not run (cost gate)`; the 4.17 USD threshold is pending (below).
7. **D7 Downloads**: train and dev authorized now and downloaded from the official AllenAI source on 2026-10-06 (10.8 MB archive, above); the test split only after the spec is frozen; nothing is downloaded, opened or computed on test before then.

## Decisions pending the author's approval at freeze
- **D6 threshold**: G-A1's own 0.14 USD tail also subtracted in the gate, so a threshold of 4.17 USD (1.0 + 0.14 + 0.14 + 2.89) at today's figures (added after the adversarial review of f1c1895).
- **D8 Pooled query**: `<paper title>. <question>` in the pooled setting for every system, question alone in the within-paper control (proposed after the adversarial review of f1c1895; reason, hand check and rejected alternative under "Rules fixed before the exam opens").
