# Phase 04 - Judge over a pooled bag with the hop: specification

Status: approved (by delegation, 2026-10-04)
Approved: 2026-10-04, by the agent under the author's delegation (master plan, decision of 2026-10-02)
Master plan: `../0_plan_maestro.md`; candidate: charter section 4, candidate 1 (`docs/refs/successor_project_charter.md`); comparison frame: `../fase-03-fusion/` (`spec.md`, `results.md`, `03.1-balance-reading.md`); ghosts: `../fase-02-ghosts/`.

## Objective
Measure the second tournament candidate, a zero-shot cross-encoder judge over a pooled bag (charter candidate 1, rerank class R), on the three terrain corpora, with one short RTX 4090 pod inside the phase's 2.5 USD soft cap.
Said first and plainly: the class ghost G-R (J-strong over G-L's top 100) loses to the old `j-union` on all three sets (Phase 02, FS@2,048 5,652 / 425 / 661 against 5,922 / 846 / 809, measured), so the advance gate alone is weak.
The strongest cheap literature recipe for this task is hybrid retrieval fused by RRF, then a cross-encoder over its top 100; it is reproduced here as the control `j-rrf3` and is a bar for the exam entrant.
The candidate `j-rrf4` differs from that control by one thing only, the entity hop in the pool, so the phase answers a narrow, new question: does the hop add evidence once a judge orders the pool.
It also says how far a pool bounded at 100 units with nothing fitted stands from `j-union`, which judges 141 to 175 units per question built from fitted fusions.

## Question
On HotpotQA FullWiki dev (7,405), MuSiQue (2,417) and MultiHop-RAG (2,255), does J-strong over the top 100 of reciprocal rank fusion of Dense, BM25, G-L and the entity hop reach a higher Full Support @2,048 than G-R, paired on the same questions, and does the hop add anything over the same judge on the same pool without it?

## Inputs measured on disk (2026-10-04)
Read on the laptop while writing this spec; every figure is measured unless labelled otherwise.
- Phase 02 depth-100 rankings (`data/phase02/rankings/<set>/`): `p10-a` (Dense), `p10-c`, `g-l`, `g-r`, `j-p10b`, `j-union`, and `g-a1` on MuSiQue and MultiHop-RAG; the sha256 of `p10-a` and `g-l` are pinned in `config.P10A_SHA256` and `config.GL_SHA256`.
- Phase 03 rankings (`data/phase03/rankings/<set>/`): `bm25`, `rrf3`, `f3`, each pinned by its manifest.
- Ranking files hold unit ids only, no scores; G-R's J-strong scores were not stored.
- The old Phase 17 J-strong scores (`../concept-embeddings-RAG/data/phase17/scores-strong-<set>.jsonl.gz`, with `pairs-<set>.jsonl.gz`) hold one score per (question, unit) of the old union pool; their sha256 are pinned in `src/edge_rag/pod/rerank.py` (`OLD17`), and Phase 02 C3 reproduced 300 of them within 1e-3 with our code.
- Old pool size, mean units per question: 174.9 HotpotQA, 165.8 MuSiQue, 140.9 MultiHop-RAG.
- Units of `rrf3`'s top 100 without a stored J-strong score, mean per question: 17.2 HotpotQA, 16.6 MuSiQue, 15.4 MultiHop-RAG (laptop count, unit-id set difference, no gold read; the 83-85 % overlap shows the ids match).
- G-R measured on the pod (Phase 02 manifests): 740,500 pairs in 4,894.5 s on an A100 (151 pairs/s, HotpotQA), 241,700 in 1,005.3 s (240 pairs/s, MuSiQue) and 225,500 in 842.9 s (268 pairs/s, MultiHop-RAG) on an RTX 4090; model load 57-197 s per session; settings: `BAAI/bge-reranker-v2-m3` revision `953dc6f6…`, float32, matmul `highest`, batch 32, max length 8,192, raw logit.
- Neighbouring passages: MultiHop-RAG's 27,989 units under 609 titles fall in 27,915 runs of consecutive same-title units in corpus file order, so file order is not article order; MuSiQue units are pooled paragraphs with no document order; HotpotQA has one unit per title.
- The entity hop (`retrieval/hop.py`, `Hops.entity_hop`, P10-C's hop) starts from Dense's top unit, scores units by shared rare entities from the old GLiNER index and excludes Dense's top 10; it has no fitted parameter; Phase 01 reproduced P10-C exactly on all three sets with it.

## What stays fixed
- Corpora, units, questions, gold and token counts: the old project's files, digest-checked through `OldData` (Phase 01).
- Metrics, as in Phases 02 and 03: FS @1,024 / @2,048 (of record) / @4,096, FS@2/5/20 units, gold share at 5, Recall@2/5/10/20/100, nDCG@10; added here, the pool ceiling, FS@100 units (every gold unit among the 100 judged).
- Paired test: exact McNemar on per-question FS@2,048, two-sided, alpha 0.05 per comparison, no correction (stated as a limitation).
- The rankings of earlier phases are read as they are, against their digests; none is rebuilt or rewritten.
- The exam corpus is not opened in this phase.

## Frozen definition of the systems
- **RRF4** (the candidate's pool, light, context row): RRF with k = 60 over four lists, in this order: Dense (`p10-a`), BM25 (Phase 03 `bm25`), G-L (`g-l`) and the entity hop at depth 100 (`hop`, computed here); score = sum of 1 / (60 + rank) over the lists that contain the unit, order score descending then unit id ascending, top 100 (`retrieval/rrf.rrf`, unchanged).
  An empty or short hop list adds only what it holds.
- **j-rrf4** (the candidate): J-strong scores every unit of RRF4's top 100; order score descending, ties by RRF4 rank.
- **j-rrf3** (the literature control): the same judge over Phase 03 `rrf3`'s top 100, same order rule with ties by RRF3 rank.
- J-strong is `BAAI/bge-reranker-v2-m3` at the pinned revision and weights with the G-R settings above, pair text `(question, f"{title}. {' '.join(sentences)}")`, through `pod/rerank.py`'s `load_strong` and `strong_scores`.
- **Score cache**: a pair already scored in the old Phase 17 J-strong file takes that stored score; every other pair is scored on the pod.
  Both judged systems read one score table per set, so a pair shared by both has one score.
- Nothing is fitted: k = 60 is the literature constant, the judge is zero-shot, the hop has no weight, and no constant is varied.

## Selection rule
There is one candidate, `j-rrf4`; `j-rrf3` is its control and a bar, never an entrant from this phase.
The rule below is applied by code to terrain figures only.
- Per set and comparison, the state is `win` when wins > losses and p < 0.05, `loss` when losses > wins and p < 0.05, `tie` otherwise, and `not run` when the set was not run.
- A set not run is never a win and never a loss; with one set not run, advancing needs a win on both other sets.
- `j-rrf4` **advances** when it wins against G-R on at least two of the three sets and loses to G-R on none.
- `j-rrf4` is the **exam entrant** (Phase 06) when it advances, wins against `j-rrf3` on at least one set, loses to `j-rrf3` on none, and loses on no set to the best in-bound rerank-class system so far; otherwise there is no entrant from this phase.
- The best in-bound rerank-class system so far is, per set by FS@2,048 in Phase 02 results, the better of `g-r` and `j-p10b` (both judge 100 units): today `g-r` on HotpotQA (5,652), `j-p10b` on MultiHop-RAG (822) and MuSiQue (704) (measured).
- The verdict is written by code as `entrant`, `advances, no entrant` (naming each failed bar) or `does not advance`.
- Context, not bars: `j-rrf3`'s verdict against G-R under the same advance rule; both judged systems against `j-union`, against the best system so far of any class (today `j-union` on HotpotQA and MultiHop-RAG, `g-a1` on MuSiQue), and against Phase 03's candidate `f3`.

### Class check
- Code checks the rerank-class bounds per set, per component and per hardware, labelled derived: at most 100 judged units per question; GPU online seconds per question against 1 s (stated for one RTX 4090; the hardware used is named); laptop CPU online seconds per question against 2 s (the light-class laptop bound, which class R includes).
- `j-union` judges more than 100 units per question (141 to 175 on average), so it is outside the class R bound (derived); this is why it is context, not a bar.
- Known before running (derived from Phase 02): G-L's search takes 0.358 s per question on an A100 on HotpotQA and G-R's whole online path 1.019 s there, so a HotpotQA system containing G-L may sit near or above the bound; MuSiQue and MultiHop-RAG ran on an RTX 4090 at 0.441 and 0.386 s.
- GPU and laptop seconds are never summed into one figure checked against a bound; a cross-hardware total is context only; a set over a bound is reported "outside the rerank class on this set (derived)" beside the verdict, which does not change (Phase 03 DS10).

## Controls
- G-R, the rerank-class ghost (the advance gate; weaker than `j-union`, see Objective).
- `j-rrf3`, the literature recipe reproduced here on the same lists and judge (a bar for the entrant).
- The best in-bound rerank-class system so far (a bar), and the best system so far of any class (context; a loss there crosses classes and is reported as a cost-quality position).
- `j-union` and `f3` as context.
In-domain caveats on HotpotQA: BGE-small was fine-tuned on HotpotQA train, answerai-colbert-small-v1's and bge-reranker-v2-m3's training mixes include HotpotQA (Phase 02 spec); `p10-b` behind `j-p10b` has HotpotQA-fitted weights.

## Scope
In:
1. A `pool` command: the entity hop at depth 100 and RRF4 per set, with a manifest, laptop timing and an equality check against `p10-c`.
2. A `judge-pairs` command: per set, the uncached pairs of `j-rrf3` and `j-rrf4` and the timing sample, with texts, counts and the cost projection.
3. A pod mode of `pod/rerank.py` that scores a pairs file and the timing sample, checks fidelity, and writes a scores file with a manifest.
4. A `judge` command that assembles `j-rrf3` and `j-rrf4` from the cache and the pod scores.
5. The results table (JSON by code, `results.md` generated from it), reusing the Phase 03 results code rather than adding per-phase stage code.

Out: neighbouring passages; any other judge, pool depth, k, cap or diversity rule; fusing judge scores with first-stage scores; a G-L rebuild; any LLM; the exam corpus; a reader or answer metric.

## Acceptance criteria
Frozen on approval.
Changing them requires a deviation.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | `edge-rag pool` writes `hop` and `rrf4` depth-100 rankings (hop lists may be shorter) for every question of the three sets into `data/phase04/rankings/<set>/`, once, with a manifest (input sha256, code commit, laptop seconds offline and per question for the hop and for RRF4, peak RSS); the 0.5 / 0.3 / 0.2 fusion of the recomputed Dense and BM25 hits and the hop (`fuse_lists`, `config.WEIGHTS_TRIPLE`, top 100) equals `p10-c` for every question (7,405 + 2,417 + 2,255 of 12,077) | command output and manifest; the equality count written by the command |
| C2 | Tests pin: RRF4 order on a hand-computed four-list example with a short hop list; the cache split (a stored pair is never sent to the pod, an unstored one always is); the judge order with the tie by pool rank; assembly from mixed cached and new scores; one score per pair shared by both systems; the pairs-file text equal to the corpus `title. sentences` text | `uv run pytest -q` |
| C3 | `edge-rag judge-pairs` writes per set, once, the uncached pairs (question id, question, unit ids, texts) and the timing sample (the first 200 questions in question-file order, every unit of both pools, cached or not), with a manifest giving cached and uncached pair counts per system, every input sha256, and the projected pod cost; for every question with a stored score, the old question text equals the harness question text | manifest; the projection and the decision to open the pod are recorded in `plan.md` before the pod is created |
| C4 | On the pod, before any full set is scored: the fresh J-strong scores of every cached pair in the timing sample are within 1e-3 of the stored scores (pairs compared and maximum absolute difference written); otherwise the pod stops and the mismatch is a deviation | `data/phase04/scores/<set>.check.json`, downloaded |
| C5 | Per set, a scores file (question id, unit ids, scores) for all uncached pairs, written once with a manifest pinning model, revision, weights sha256, settings, code commit, GPU, `costPerHr`, seconds (load, read, score, timing sample per question) and cgroup memory; its sha256 is equal on the pod and on the laptop | manifests in `data/phase04/scores/`; both `sha256sum` outputs in `plan.md` |
| C6 | `edge-rag judge` writes `j-rrf3` and `j-rrf4` depth-100 rankings for every question run, once, with a manifest pinning the sha256 of the pools, the old score files and the pod score files; every judged list is a permutation of its pool's top 100 and every pair has exactly one score | manifest; counts written by the command |
| C7 | `data/phase04/results.json` and `results.md`, written by code, list per set `j-rrf4`, `j-rrf3`, RRF4, G-R, the best in-bound rerank-class system so far, `j-union`, the best system so far and `f3` with every metric of record and the pool ceiling, a label per value, the class check, and cost columns: offline and online, seconds and USD, per component and in total, with the hardware of each; inherited costs carry their source label; also the count of RRF4 top-100 units contributed only by the hop (measured) and the questions whose gold enters RRF4's top 100 only through the hop (exploratory) | files; `results.md` regenerated from `results.json` byte-equal |
| C8 | The same table gives exact McNemar on FS@2,048 for `j-rrf4` against G-R, `j-rrf3`, the best in-bound rerank-class system so far, `j-union`, the best system so far and `f3`, and for `j-rrf3` against G-R and the best system so far, per set, with wins, losses, ties, p and state; the verdict, `j-rrf3`'s context verdict and the entrant are written by code | `results.json` fields; a test of the state and verdict code pinning each bar and the `not run` case (HotpotQA not run, `j-rrf4` wins MuSiQue and ties MultiHop-RAG: `does not advance`) |
| C9 | Money: `clientBalance` is read before the pod is created and again at least 2 h after the last pod of the phase terminated, both readings recorded with UTC time; the phase spend is recorded as time x rate (derived), balance delta (measured) and invoice when the author copies it; C9 is met when time x rate is at most 2.5 USD, no pod is left (`myself { pods }` empty), and either the balance delta is at most 2.5 USD or the author's billing-explorer reading shows at most 2.5 USD of charges dated inside the phase | readings and figures in `plan.md`; money left inside the 25 USD authorization = 10.22 USD minus the spend (derived) |
| C10 | `npm run check` passes | exit 0 |

## Cost accounting and money
- Online per question, each its own row with its hardware: BGE-small question encoding, Dense, BM25 and RRF (laptop, Phase 03 manifests); the hop and RRF4 (laptop, measured here); G-L search (Phase 02 pod manifests); J-strong over 100 units (RTX 4090, measured here on the timing sample, every pair scored fresh, as G-R was timed).
- Offline per corpus: BM25 build (Phase 03); G-L encode and index (Phase 02); the GLiNER entity index behind the hop (old project, inherited labels: HotpotQA 6.18 h, 4.57 USD attributable; MuSiQue 10.3 min, 0.13 USD; MultiHop-RAG inside 0.0404 USD with BGE); Dense corpus embeddings as in Phase 03; the judge's model load per session as a setup row.
- The stored old scores are reused, so the pod cost of this phase is lower than the cost of running the system from scratch; the online row is what one question costs, and the results state both.
- Pod: one RTX 4090, Secure Cloud, about 0.74 USD/h (rate recorded at creation), sets in the order MuSiQue, MultiHop-RAG, HotpotQA; inputs uploaded are the pairs files with texts, so no corpus is uploaded.
- Forecast (projection): about 35 uncached pairs per question (17 measured for `rrf3`, about 18 more assumed for the hop's units) plus 120,000 timing-sample pairs, about 543,000 pairs; at 151 pairs/s on HotpotQA and 240-268 elsewhere (G-R's measured rates) about 0.8 h of scoring, 0.2 h of model loads and 0.5 h of setup and transfers, 1.5 h, **about 1.1 USD**.
- Gate before the pod: `judge-pairs` gives the exact count; the projection is that count at the per-set G-R rates plus 0.75 h at 0.74 USD/h; the pod opens if it is at most 2.0 USD; otherwise HotpotQA is run on the master plan's preregistered 1,000-question subsample (`random.Random(20261002).sample` over the sorted qids), recorded in `plan.md` before the pod opens, and every other set in full.
- Hard stop: the pod is terminated when its time x rate reaches 2.5 USD (3.4 h at 0.74 USD/h), or on a failed C4 check; finished sets are downloaded first and an unfinished set is `not run`.
- After the phase at most 2.5 USD is spent, leaving at least 7.72 USD of the 25 USD authorization, enough for Phase 05's 2.5 USD soft cap and the exam's 5.0 USD (derived).

## Assumptions
- A1. A stored old J-strong score equals what our code computes for the same pair within 1e-3; Phase 02 C3 showed it on 300 pairs, and C4 checks it again on every cached pair of the timing sample.
- A2. The pod scores at about G-R's measured rates; the gate uses the slowest measured rate per set.
- A3. The laptop can recompute Dense, BM25 and the hop on HotpotQA as Phase 01 did (1,851.6 s for the four old systems; Phase 03 components peaked at 13.5 GB RSS); if it cannot, HotpotQA is `not run` (a deviation), not moved to a pod.
- A4. Class R's "at most the top-100" means at most 100 judged units per question.

## Risks
- The candidate differs from its control only by the hop; old Phase 17 recorded the hop hurting under J-strong on news (MultiHop-RAG), so a tie or loss against `j-rrf3` is a plausible outcome (interpretation).
- The bounded pool may hold less gold than `j-union`'s larger pool; the pool ceiling row shows it.
- Mixing cached and fresh scores can reorder pairs whose scores differ by less than 1e-3; C4 bounds the difference and the cross-encoder showed 0 reorders under batching in the old project.
- HotpotQA is in-domain for BGE-small, G-L and J-strong; its figures favour every judged system alike.
- On HotpotQA the G-L component was measured on an A100 and the judge will be on an RTX 4090; the class check cannot give one per-hardware total there.
- Up to eight paired tests per set (24 in all), at alpha 0.05 without correction: a single significant comparison outside the rule is reported, not acted upon.
- Billing lags behind a terminated pod; C9 reads the closing balance at least 2 h after termination and keeps the billing-explorer branch (deviation 03.1).

## Anti-goals
- No other k, pool depth, judge or hop setting; no second variant after a result is seen; no fitted fusion of judge and first-stage scores.
- No pre-measurement of `j-rrf4` or `j-rrf3` before this spec; the counts above read no gold and no scores of the new systems.

## Decisions taken
By the agent under the author's delegation, 2026-10-04; none is open.
- DJ1. Pool sources Dense, BM25, G-L and the entity hop. Reason: they are the light-class lists with depth-100 rankings or code on all three sets at zero money; G-L is the master plan's addition and the old pool never had it; G-R and G-A1 are not first-stage lists.
- DJ2. No neighbouring passages. Reason: the inherited units carry no position on any set (measured above); rebuilding positions for MultiHop-RAG alone would add a source on one set and change the units' provenance.
- DJ3. The entity hop, not P14's relevance hop. Reason: the relevance hop mixes with alpha 0.75 fitted on HotpotQA; the entity hop has nothing fitted.
- DJ4. Pool = RRF (k = 60) top 100 of the four lists, deduplicated by unit id, no per-source cap and no diversity pass. Reason: 100 judged units is the class R bound; RRF is the literature's weight-free rule and the control's own pool, so the candidate differs from the control by the hop alone; Phase 03 measured the diversity pass losing to plain RRF on two sets.
- DJ5. J-strong (`bge-reranker-v2-m3`, 568 M parameters) with G-R's settings. Reason: already pinned, costed and checked here (Phase 02 C3); zero-shot, under 1B, float32 (no bfloat16 determinism risk); measured 0.374-0.661 s per 100 pairs per question; a second judge would be a second candidate.
- DJ6. Control `j-rrf3`, the hybrid-plus-rerank recipe, reproduced on the same lists. Reason: it is the strongest cheap literature recipe for the class and costs the same per question; G-R alone would be a weaker bar.
- DJ7. G-R is the advance gate; the entrant also needs a win and no loss against `j-rrf3` and no loss to the best in-bound rerank-class system. Reason: same frame as Phase 03 (DS12): a candidate enters the exam on its own claim, not on a weak gate.
- DJ8. `j-union` is context, not a bar. Reason: it judges more than 100 units per question (outside class R, derived) and its pool comes from fitted fusions, so it cannot be an exam system under the rules; it is still reported with paired tests.
- DJ9. Reuse the old Phase 17 J-strong scores as a cache, read in place against their digests. Reason: same model, revision, settings and pair text; it saves most of the pod time, and C4 checks it on every cached pair of the timing sample.
- DJ10. Time the judge on a fixed sample, every pair scored fresh. Reason: with the cache the pod's time per question is not the system's cost; G-R was timed over full 100-unit pools, so this keeps the rows comparable.
- DJ11. Terrain and metrics exactly as Phase 03, all questions of each set. Reason: one comparison frame across candidates; the terrain is open and the exam stays locked.
- DJ12. Pod gate at 2.0 USD projected, hard stop at 2.5 USD, HotpotQA subsample as the only fallback. Reason: fits the soft cap with margin, keeps the reserve for Phases 05-06, and the subsample is already preregistered.
- DJ13. Money criterion on a closing balance read at least 2 h after termination, with the billing-explorer branch. Reason: deviation 03.1, billing lag.
- DJ14. No new row in the master plan's decision table. Reason: these decisions apply the existing class bounds, budget lines and charter candidate; none changes the plan.

## Open decisions
None.
