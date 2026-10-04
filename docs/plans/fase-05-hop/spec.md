# Phase 05 - Multi-seed convergent hop with a refined query: specification

Status: approved (by delegation, 2026-10-04)
Approved: 2026-10-04, by the agent under the author's delegation (master plan, decision of 2026-10-02)
Master plan: `../0_plan_maestro.md`; candidate: charter section 4, candidate 2 (`docs/refs/successor_project_charter.md`); comparison frame: `../fase-03-fusion/` and `../fase-04-judge/` (`spec.md`, `results.md`, `04.1-results-code.md`); ghosts: `../fase-02-ghosts/`.

## Objective
Measure the third tournament candidate, a multi-seed convergent entity hop with a refined query (charter candidate 2, light class L), on the three terrain corpora, on the laptop CPU, at zero money.
Said first and plainly: the advance gate, the light-class ghost G-L, is weak (Phase 02 FS@2,048 5,000 / 533 / 187 on HotpotQA / MuSiQue / MultiHop-RAG, measured), weaker than the best light-class systems already on disk (RRF4 5,291 on HotpotQA, P14 761 on MuSiQue, P10-B 587 on MultiHop-RAG, measured), so the gate alone proves little.
The working control, `rrf-prf` (the candidate without its hop), is weaker than the strongest light-class literature recipe for multi-hop retrieval, MDR (question plus the first passage re-encoded for a second dense round), which is not reproduced: it is a single-task fit trained on HotpotQA (Phase 00 `survey.md`: "an in-domain upper reference, not a ghost") and would need a GPU encode of every corpus; its published FullWiki figures (Recall@2 65.9, Recall@20 80.2, another corpus version and metric) are context only.
The exam entrant therefore also has to beat `rrf-prf` and must not lose to the best light-class system so far, which today includes fitted fusions and G-L-based lists.
The phase answers whether hopping from several seeds of two first-stage lists, rewarding units that several seeds reach and adding a second dense round with a query moved towards the seeds, adds evidence over the same hybrid without the hop.
Beside it, the phase merges the Phase 03 and Phase 04 results code into one shared module and fixes the open Phase 04 review round 2 finding 4, both as the author decided after Phase 04 (master plan, 2026-10-04).

## Question
On HotpotQA FullWiki dev (7,405), MuSiQue (2,417) and MultiHop-RAG (2,255), does reciprocal rank fusion of Dense, BM25, Dense with a refined query and a multi-seed convergent entity hop reach a higher Full Support @2,048 than G-L, paired on the same questions, and does the convergent hop add anything over the same fusion without it?

## Inputs measured on disk (2026-10-04)
Read on the laptop while writing this spec; every figure is measured unless labelled otherwise; no gold was read and no ranking of a Phase 05 system exists.
- Dense depth-100 rankings `data/phase02/rankings/<set>/p10-a.jsonl.gz` (BGE-small; sha256 in `../fase-03-fusion/spec.md` and `config.P10A_SHA256`).
- BM25 depth-100 rankings `data/phase03/rankings/<set>/bm25.jsonl.gz`, sha256 hotpotqa-dev `0c3e1a3d51798b4b9db41391cd0acce65f11b8198e1e5450eb3e8c96f098f376`, musique `7931e75641b5d6cc7c5bea5fa6659134dfc592af4cb6dff4efa23e2f2f1fdd0a`, multihop-rag `8ca1e9cd5f5713f8890813b205770f362123d783baf435fe426c77ced18bc178` (Phase 04 `pool.manifest.json`).
- The single-seed entity hop at depth 100 `data/phase04/rankings/<set>/hop.jsonl.gz`, sha256 hotpotqa-dev `57f6bf67bc4a2f26b9c6b87415c85598220435f3f6597eb96ffe003ba1b7bcb5`, musique `5f75017fc2df2c2f67c9ec2837b2e751d704f1cf7d5566ff7ef4bcb3f95713e7`, multihop-rag `b072f1e6937ad7d76432f578567027f1e5768376d8e5105c461bc91fe9a0ab8f`.
- The old GLiNER entity index (`urchade/gliner_medium-v2.1`, configuration digest `2f7864661b8ce7ff`) loads on all three sets through `retrieval/hop.load_node_index`, digest-checked, its rows equal to the corpus units in order (Phase 04 `pool`); no new extraction is needed and no GPU work is needed in this phase.
- The old BGE-small unit vectors and cached question vectors load on all three sets through the Phase 01 loaders (`pool.load_inputs`).
- Laptop costs, Phase 04 `pool.manifest.json`: the single-seed hop takes 0.024093 / 0.000343 / 0.000171 s per question (HotpotQA / MuSiQue / MultiHop-RAG); entity index and weights load in 21.654 / 0.663 / 0.055 s; peak RSS 13,051.1 / 539.2 / 232.7 MB; the HotpotQA "hop and RRF4" stage, Dense and BM25 retrieval included, took 1,662.6 s.
- Laptop costs, Phase 03 manifests: BGE-small question encoding 0.023502 / 0.018511 / 0.040273 s per question on 200 questions; Dense 0.147475 / 0.004279 / 0.001207 s; BM25 0.04842 / 0.000915 / 0.00043 s (HotpotQA / MuSiQue / MultiHop-RAG).
- Distinct seeds per question under the rule below (top 5 of Dense united with top 5 of BM25), unit ids only: mean 8.59 / 8.79 / 8.97, minimum 5, maximum 10 (HotpotQA / MuSiQue / MultiHop-RAG).
- Results pages: `../fase-03-fusion/results.md` sha256 `e50c85a9ca1e201dd30cbc50bde426c5fddeab638e2c4f53e3c04d3227e0a9a9`, `../fase-04-judge/results.md` sha256 `6b4b6bd19cb937b962cc1245b10ed793eb5d649a0ce67cf7a273391db972cc0c`; `data/phase03/results.json` sha256 `ccf4dfc98ebc893d81220dbbc3db3c324a3c8be1a2a471afb5f906e4a4b31efd`, `data/phase04/results.json` sha256 `93fa91404e2ec9a180225ce6f22740adbee45b69ba7163c6e5ea93d260ae4dc9`; both JSON files carry `git_commit` and `git_src_changes`, the pages do not.
- Results code: `src/edge_rag/fusion_results.py` (Phase 03; states, advance rule, verdict, inherited cost labels, `write_pair`, `regenerate`) and `src/edge_rag/judge_results.py` (Phase 04; its own set table, cost rows, class check and page, deviation 04.1).
- Finding 4 of Phase 04 review round 2: `judge_results.system_cost` writes the component `J-strong over 100 units` with seconds per 100 pairs, while the judged units per question come from the judged lists (`judged_units`); the value is 100 on every Phase 04 set (measured, Phase 04 `plan.md`).

## What stays fixed
- Corpora, units, questions, gold and token counts: the old project's files, digest-checked through `OldData` (Phase 01).
- Metrics, as in Phases 02-04: FS @1,024 / @2,048 (of record) / @4,096, FS@2/5/20 units, gold share at 5, Recall@2/5/10/20/100, nDCG@10.
- Paired test: exact McNemar on per-question FS@2,048, two-sided, alpha 0.05 per comparison, no correction (stated as a limitation).
- The rankings and results of earlier phases are read as they are, against their digests; none is rebuilt or rewritten.
- The exam corpus is not opened in this phase.

## Frozen definition of the systems
Per question, with `q` the cached BGE-small question vector:
- **Seeds**: the top 5 units of Dense (`p10-a`), then the top 5 of BM25 (Phase 03 `bm25`) that are not already seeds, in that order (5 to 10 distinct units).
- **Per-seed hop**: for one seed `s`, the old entity hop's rarity score from `s` instead of Dense's first unit: every unit sharing at least one `entity` node with `s` scores the sum of `w = log(1 + N / (1 + df))` over the shared nodes (`retrieval/hop.node_weights`, unchanged); units with a positive score that are not seeds are ordered by score descending, then unit id ascending, top 100.
  A seed with no entity node gives an empty list.
- **Convergent hop `hop-ms`**: RRF with k = 60 over the per-seed lists (`retrieval/rrf.rrf`, unchanged), top 100; a unit reached from several seeds sums one 1 / (60 + rank) per seed, so a unit at rank 50 under two seeds (2 / 110 = 0.0182) is placed above a unit at rank 1 under one seed (1 / 61 = 0.0164).
- **Refined query**: `q' = (q + c) / ||q + c||`, with `c` the mean of the seeds' BGE-small unit vectors (Rocchio's form with both weights 1: the question and the centroid of its feedback count equally).
- **Refined dense list `dense-prf`**: Dense retrieval with `q'` over the same unit vectors, exact inner product, top 100, ties by unit id (`retrieval/dense_bm25.Dense`, unchanged).
- **`mch`** (the candidate): RRF with k = 60 over four lists, Dense (`p10-a`), BM25 (`bm25`), `dense-prf` and `hop-ms`, top 100, order score descending then unit id ascending.
- **`rrf-prf`** (the control, the candidate without its hop): RRF with k = 60 over Dense, BM25 and `dense-prf`, top 100.
- **`rrf-1s`** (context, the single-seed version): RRF with k = 60 over Dense, BM25, `dense-prf` and the Phase 04 single-seed `hop` list (Dense's first unit as the only seed, Dense's top 10 excluded), top 100.
- `hop-ms` and `dense-prf` are also scored as context rows.
- Nothing is fitted and nothing is tuned: k = 60 is the literature constant, the seed count 5 per list is the lower end of the charter's preregistered "top 5-10 of Dense and BM25", the Rocchio weights are equal by construction, the depths are the harness's 100, and no constant is varied on any split of any terrain set.

## Selection rule
There is one candidate, `mch`; `rrf-prf` is its control and a bar, never an entrant from this phase; `rrf-1s`, `hop-ms` and `dense-prf` are context.
The rule below is applied by code to terrain figures only.
- Per set and comparison, the state is `win` when wins > losses and p < 0.05, `loss` when losses > wins and p < 0.05, `tie` otherwise, and `not run` when the set was not run.
- A set not run is never a win and never a loss; with one set not run, advancing needs a win on both other sets.
- `mch` **advances** when it wins against G-L on at least two of the three sets and loses to G-L on none.
- `mch` is the **exam entrant** (Phase 06) when it advances, wins against `rrf-prf` on at least one set, loses to `rrf-prf` on none, and loses on no set to the best light-class system so far; otherwise there is no entrant from this phase.
- The best light-class system so far is, per set by FS@2,048 in Phase 02-04 results, the best of `p10-a`, `p10-b`, `p10-c`, `p14`, `g-l`, `rrf3`, `f3` and `rrf4`: today `rrf4` on HotpotQA (5,291), `p14` on MuSiQue (761), `p10-b` on MultiHop-RAG (587) (measured); the code reads it from the stored results and names it.
- The verdict is written by code as `entrant`, `advances, no entrant` (naming each failed bar) or `does not advance`.
- Context, not bars: `rrf-prf`'s verdict against G-L under the same advance rule; `mch` against `rrf-1s` (does a multi-seed convergent hop add over the single seed); `mch` and `rrf-prf` against the best system so far of any class, per set by FS@2,048 in Phase 02-04 results, today `j-rrf4` on HotpotQA (5,950), `g-a1` on MuSiQue (1,101), `j-rrf3` on MultiHop-RAG (873) (measured).

### Class check
- Code checks the light-class bounds per set, per component and per hardware, labelled derived: laptop CPU online seconds per question against 2 s; GPU online seconds against 0.1 s for any GPU component (none in `mch`, `rrf-prf` or `rrf-1s`).
- Offline: GLiNER extraction, inherited, against 2 GPU-h per million units: HotpotQA 6.18 h over 5,233,329 units (1.18 h per million), MuSiQue 10.3 min over 101,962 (1.68), MultiHop-RAG 154.77 s over 27,989 (1.54) (derived from the handover's cost table; hardware as recorded there).
- GPU and laptop seconds are never summed into one figure checked against a bound; a set over a bound is reported "outside the light class on this set (derived)" beside the verdict, which does not change (Phase 03 DS10).
- Projection (not measured): `mch` online on HotpotQA about 0.58 s per question (encoding 0.024, Dense 0.147, BM25 0.048, 8.6 seeds x 0.024 s of hop, a second Dense search 0.147, two RRF passes 0.003), about 0.03 s on MuSiQue and 0.05 s on MultiHop-RAG, all under 2 s.

## Controls
- G-L, the light-class ghost (the advance gate; weak, see Objective).
- `rrf-prf`, the same hybrid with the same refined query and no hop (a bar for the entrant), so the hop's contribution is read from `mch` against `rrf-prf`.
- The best light-class system so far (a bar): it holds fitted fusions (P14 in-sample on HotpotQA, P10-B) and G-L-based lists (`rrf4`), so it is stronger than any weight-free laptop-only recipe has been here.
- The best system so far of any class (context; a loss there crosses classes and is reported as a cost-quality position).
- MDR, the light-class literature recipe for a second round from the first results, is named and not reproduced (Objective); this spec cites no published weight-free multi-seed hop in class L, and none was searched for beyond the Phase 00 survey (a gap, stated).
In-domain caveat on HotpotQA: BGE-small, behind Dense, `dense-prf` and every seed from Dense, was fine-tuned on HotpotQA train; `p14` is in-sample there.

## Scope
In:
1. A `converge` command: seeds, per-seed hops, `hop-ms`, `dense-prf`, `mch`, `rrf-prf` and `rrf-1s` per set, with a manifest, laptop timing per component and the equality checks of C1.
2. One shared results module for Phases 03, 04 and 05, replacing `fusion_results.py` and `judge_results.py`; Phase 03 and Phase 04 pages regenerate byte-equal; the commands `fusion-results` and `judge-results` stay as entry points, and `hop-results` is added.
3. The finding 4 fix: the judge's cost row takes its unit count from the judged lists.
4. The Phase 05 results table (JSON by code, `results.md` generated from it).

Out: G-L, G-R or any judge in the candidate; a refined BM25 query (RM3); any other seed count, k, Rocchio weight, depth or exclusion rule; P14's relevance hop; neighbouring passages; any new extraction, encoder or GPU work; any LLM; the exam corpus; a reader or answer metric; a gold-informed ceiling (none is run in this phase).
If something would need a GPU or money, it is out of scope and recorded as a deviation.

## Acceptance criteria
Frozen on approval.
Changing them requires a deviation.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | `edge-rag converge` writes `hop-ms`, `dense-prf`, `mch`, `rrf-prf` and `rrf-1s` depth-100 rankings (`hop-ms` may be shorter) for every question of the three sets into `data/phase05/rankings/<set>/`, once, with a manifest (input sha256 including the entity index digest, constants 5 / 60 / 100, code commit and `git_src_changes`, laptop seconds offline and per question per component, peak RSS, seed-count mean and range, empty and short `hop-ms` lists); the recomputed Dense and BM25 lists equal `p10-a` and Phase 03 `bm25`, and the per-seed hop from Dense's first unit with Dense's top 10 excluded equals the Phase 04 `hop` list, for every question (each 7,405 + 2,417 + 2,255 of 12,077); one question per set is traced by hand (seeds, the cosine of `q'` with `q`, the top `hop-ms` unit and its seeds) | command output and manifest; equality counts written by the command; the hand trace in `plan.md` |
| C2 | Tests pin, on hand-computed examples: the seed order and deduplication; the per-seed hop's exclusion of seeds and its equality with `Hops.entity_hop` when the seed is Dense's first unit and Dense's top 10 is excluded; the convergence example of the definition (two seeds at rank 50 above one seed at rank 1); an empty per-seed list adding nothing; `q'` on a two-dimensional example; the four-list RRF with an empty `hop-ms` | `uv run pytest -q` |
| C3 | One results module serves Phases 03, 04 and 05; `fusion_results.py` and `judge_results.py` no longer exist; the Phase 03 and Phase 04 tests of states, bars and verdicts pass with their assertions unchanged (imports may move); `--from-json` regenerates both stored pages byte-equal (sha256 above); a full recompute of each phase from its run files, written to a scratch path and never over the stored files, gives a `results.json` equal to the stored one except `git_commit` and `git_src_changes`, and a page byte-equal to the stored page | command outputs with sha256 and the field comparison in `plan.md` |
| C4 | Finding 4 fixed: the judge's online cost row is named `J-strong over N units` with N from the judged lists (`judged_units`) and seconds = seconds per 100 pairs x N / 100, its label saying `x N`; a test with an 80-unit judged list gives `J-strong over 80 units` and 0.8 times the per-100 seconds; on Phase 04 N = 100 on every set, so C3's recompute gives the stored text and figures | `uv run pytest -q`; C3's recompute |
| C5 | `data/phase05/results.json` and `docs/plans/fase-05-hop/results.md`, written by the shared module, list per set `mch`, `rrf-prf`, `rrf-1s`, `hop-ms`, `dense-prf`, G-L, the best light-class system so far and the best system so far with every metric of record, a label per value, the class check, and cost columns: offline and online, seconds and USD, per component and in total, with the hardware of each; inherited costs carry their source label; also the count of `mch` top-100 units contributed only by `hop-ms` (measured) and the questions whose gold enters `mch`'s top 100 only through `hop-ms` (exploratory) | files; `results.md` regenerated from `results.json` byte-equal |
| C6 | The same table gives exact McNemar on FS@2,048 for `mch` against G-L, `rrf-prf`, `rrf-1s`, the best light-class system so far and the best system so far, and for `rrf-prf` against G-L and the best system so far, per set, with wins, losses, ties, p and state; the verdict, `rrf-prf`'s context verdict and the entrant are written by code | `results.json` fields; a test of the verdict pinning each bar and the `not run` case (HotpotQA not run, `mch` wins MuSiQue and ties MultiHop-RAG: `does not advance`) |
| C7 | Money: no pod is created and no paid API is called; spend 0 USD; `clientBalance` and `myself { pods }` are read at the phase start and at the close, with UTC time; met when no pod existed at either reading and the balance did not fall, or the billing explorer shows no charge dated inside the phase | readings in `plan.md`; money left inside the 25 USD authorization stays 9.89 USD (derived, provisional as in the master plan) |
| C8 | `npm run check` passes | exit 0 |

## Cost accounting and money
- Online per question, each its own row with its hardware (laptop CPU, ARM64, Windows): BGE-small question encoding (Phase 03 row, same model and laptop, source-labelled); Dense; BM25; per-seed hops summed over the seeds; refined query and `dense-prf` search; the two RRF passes (measured here over every question, except the encoding).
- Offline per corpus: GLiNER extraction (inherited labels: HotpotQA 6.18 h, 4.57 USD attributable; MuSiQue 10.3 min, 0.13 USD; MultiHop-RAG inside 0.0404 USD with BGE); Dense corpus embeddings (MultiHop-RAG 41.20 s inside the same 0.0404 USD, others not recorded); BM25 build (Phase 03); entity index load and weights (laptop, measured here).
- Laptop USD is 0 by assumption (owned laptop, energy not counted), labelled so.
- Money: 0 USD; the master plan's 2.5 USD soft cap for Phase 05 is not used, and the author has set the money reserve aside.
- Runtime projection (not measured): HotpotQA about 1.3 h of wall time (Phase 04's 1,662.6 s question stage plus about 0.36 s per question for the extra hops and the second Dense search, plus about 397 s of loading from Phase 04's stage times), memory near Phase 04's 13.1 GB peak; MuSiQue and MultiHop-RAG under a minute each.

## Assumptions
- A1. The laptop can hold HotpotQA's vectors, BM25 index and entity index together, as Phase 04 did (13.1 GB peak RSS); if it cannot, HotpotQA is `not run` (a deviation), not moved to a pod.
- A2. The full recompute of Phases 03 and 04 needs only their run files on disk (rankings, manifests, Phase 02 results), all present in the main checkout's `data/`.
- A3. A seed's entity nodes and rarity weights are those of the old index; seeds from BM25 are units of the same index (its rows are the corpus units).

## Risks
- MultiHop-RAG: the hop hurt there in old Phase 16 and under J-strong in old Phase 17, and Dense's first unit is gold for only 17 % of its questions (handover, measured there); BM25 seeds may help or the hop may add noise (interpretation).
- Query drift: a refined query moves towards non-gold seeds when the first results are wrong, a known failure of pseudo-relevance feedback (interpretation); `dense-prf` is scored alone so the effect is visible.
- A unit reached from many seeds through common entities: each seed contributes only its top 100 by rarity, which bounds it, but popular entities on HotpotQA can still dominate a seed's list (interpretation).
- Three of four lists in `mch` come from BGE-small or its neighbourhood; RRF may over-weight Dense's view (interpretation).
- G-L is a weak gate and the best light-class bar holds fitted and in-sample systems; a loss there is plausible and is reported as a cost-quality position.
- Twenty-one paired tests (seven per set) at alpha 0.05 without correction: a single significant comparison outside the rule is reported, not acted upon.
- The results merge touches code behind two closed phases; C3's recompute is the guard, and any difference is a defect fixed before Phase 05's results are written.

## Anti-goals
- No tuning of the seed count, k, Rocchio weights, depths or exclusions; no second variant after a result is seen; no fitted fusion weight.
- No pre-measurement of any Phase 05 system before this spec; the seed counts above read unit ids only, no gold; no oracle ceiling is run.
- No per-phase results module; no change to a closed phase's figures, rule or page.

## Decisions taken
By the agent under the author's delegation, 2026-10-04; none is open.
- DH1. Seeds from Dense and BM25, the top 5 of each, deduplicated (5 to 10 seeds). Reason: the charter names Dense and BM25 and a top 5-10; both lists exist on disk for every question; 5 per list is the low end, which limits drift from weak tail seeds, and BM25 seeds give news questions (where Dense's first unit is rarely gold) seeds from another view.
- DH2. No G-L in the seeds or the fusion. Reason: G-L needs a GPU per question and is outside the light class on HotpotQA (0.358 s on an A100), it is the gate the candidate is measured against, and it is the weakest list on MultiHop-RAG (187); without it the candidate runs entirely on the laptop and inside class L on every set.
- DH3. Convergence = RRF (k = 60) over the per-seed hop lists. Reason: summing reciprocal ranks rewards a unit reached from several seeds with no weight or threshold; a count-of-seeds rule would let common entities reach most units with a high count, and the old lesson is that hard filters on the seed lose (handover fact 7).
- DH4. Per-seed lists are the old entity hop's rarity order, generalized only in the seed and the exclusion set. Reason: the hop has no fitted parameter and Phase 01 and 04 reproduced it exactly; C1 checks the generalization against Phase 04's `hop`.
- DH5. Seeds are excluded from the hop lists, not Dense's top 10. Reason: seeds already reach the fusion through Dense and BM25; with several seeds the read set is the seed set.
- DH6. Refined query = Rocchio with equal weights on the question and the seeds' centroid, through the same Dense. Reason: it needs no generative LLM and no new encoder (class L), no weight to fit, and the same unit vectors; MDR's trained second-hop encoder is a single-task fit and needs a GPU encode.
- DH7. The question enters through `dense-prf`, Dense and BM25 in the fusion, not inside the hop's order. Reason: putting similarity inside the hop needs P14's fitted alpha or a lexicographic rule chosen without evidence; RRF already lifts hop units the question lists also hold.
- DH8. `mch` = RRF (k = 60) of four lists, top 100, no cap or diversity pass. Reason: the literature's weight-free rule and the frame of Phases 03 and 04; the 2,048-token budget is filled from this order by the harness; Phase 03 measured the diversity pass losing to plain RRF on two sets.
- DH9. Control `rrf-prf`, the candidate without the hop. Reason: it isolates the claim of the phase, as `j-rrf3` did for `j-rrf4`; the refined query is part of the charter candidate, so it stays in the control.
- DH10. `rrf-1s` is context, not a bar. Reason: it answers whether several seeds beat one at zero cost (Phase 04's `hop` file), but a third bar would test a sub-claim; it is reported with its paired test.
- DH11. The best light-class bar includes systems outside the bound on a set (`rrf4` on HotpotQA) and fitted fusions. Reason: the stricter bar, as in Phase 03; a reference's cost does not make the candidate better on quality, and the class check reports cost beside it.
- DH12. No tuning on any split. Reason: the protocol allows choosing on terrain, but every constant here has a source (charter range, literature constant, harness depth, equal Rocchio weights), so no dev split is carved.
- DH13. Merge the results code into one module, keep the command names `fusion-results` and `judge-results` and add `hop-results`. Reason: the author overturned deviation 04.1; the stored pages name their command in their header, so keeping the names keeps them true and byte-equal.
- DH14. Finding 4 is fixed by computing the row's unit count and seconds from the judged lists. Reason: the label then follows the data; on Phase 04 the count is 100 on every set, so the stored page is unchanged and C3's full recompute proves it, and no closed figure is edited.
- DH15. The full recompute writes to a scratch path and compares, with `git_commit` and `git_src_changes` excluded. Reason: stored artifacts are write-once and those two fields record the producing code, which changes by construction.
- DH16. Online costs measured over every question; the encoding row reused from Phase 03. Reason: same model and laptop; every other component is timed where it runs.
- DH17. Money 0 USD, no pod. Reason: every input exists on disk and the candidate is CPU-only; the author set the reserve aside.
- DH18. No new row in the master plan's decision table. Reason: these decisions apply the charter candidate, the class bounds and the author's decisions of 2026-10-04; none changes the plan.

## Open decisions
None.
