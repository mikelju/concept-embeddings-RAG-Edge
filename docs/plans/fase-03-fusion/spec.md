# Phase 03 - Weight-free fusion with source diversity: specification

Status: approved (by delegation, 2026-10-03)
Approved: 2026-10-03, by the agent under the author's delegation (master plan, decision of 2026-10-02)
Master plan: `../0_plan_maestro.md`; candidate: charter section 4, candidate 3 (`docs/refs/successor_project_charter.md`); ghosts and old references: `../fase-02-ghosts/` (`spec.md`, `plan.md`, `results.md`, `02.1-money.md`).

## Objective
Measure the first tournament candidate, weight-free rank fusion with source diversity (charter candidate 3, light class), on the three terrain corpora, on the laptop, at zero money.
It is the first candidate because it needs no GPU and no fitting: every input it fuses already exists on disk or can be computed on the laptop CPU.
Said first and plainly: the advance gate, the light-class ghost G-L, is weaker than the project's fitted Dense + BM25 fusions on all three sets (Phase 02 FS@2,048: 5,000 / 533 / 187 against 5,224 / 761 / 587, measured).
The strongest cheap literature recipe, a convex combination, is not reproduced here; it is represented only by those old fitted fusions (P10-B, P14), which are reported and bar the exam entry (selection rule) but are not the advance gate.
The phase says whether a rule with nothing fitted beats G-L, whether its diversity pass adds anything over plain reciprocal rank fusion, and how far it stands from the best light-class system and the project's best system so far, on the same questions, with its cost beside its score.

## Question
On HotpotQA FullWiki dev (7,405), MuSiQue (2,417) and MultiHop-RAG (2,255), does reciprocal rank fusion of Dense, BM25 and G-L, followed by a fixed source-diversity rule, reach a higher Full Support @2,048 tokens than G-L, paired on the same questions, and what does it add over plain reciprocal rank fusion of the same lists?

## Inputs measured on disk (2026-10-03)
Read on the laptop while writing this spec; every count below is measured (read from the files with the harness loader) unless labelled otherwise.
- Depth-100 rankings in `data/phase02/rankings/<set>/`, every question present, every list exactly 100 units long, on all three sets: `p10-a` (Dense, BGE-small), `p10-b`, `p10-c`, `p14` and `g-l` (answerai-colbert-small-v1, PLAID).
- `j-p10b` and `j-union` (J-strong over the old pools) are pools of 105 to 276 units per question, not depth-100 lists; `g-r` is G-L's top-100 reordered; `g-a1` (Search-R1 evidence) has 3 to 18 units per question and exists on MuSiQue and MultiHop-RAG only.
- No BM25-only ranking file exists.
  The Phase 01 laptop pipeline (`src/edge_rag/reproduce.py`) rebuilds the old BM25 index (digest-checked) and runs Dense and BM25 at depth 100 on all three sets; on the laptop it took 1,851.6 s for the retrieval of the 7,405 HotpotQA questions with all four old systems, 20.0 s on MuSiQue and 8.8 s on MultiHop-RAG (measured, `data/results/reproduce-<set>.json`).
- G-L reported rankings, sha256: hotpotqa-dev `b5b348ee15dba2017bfde6ae6c25e06047ab1f2b1ffd8feb038c16561795a68b` (A100, commit f3f499e), musique `8d7807345acf1eea111c943460c6a53c0e6dd167525c3cb53ebb3e8778f3b150`, multihop-rag `791ed3653e1f90aa49d240df7c51767f7e846dbf5996930dbe6e1f6a72201c52` (RTX 4090, commit cc47510); each equals the sha256 its manifest records.
- Dense (`p10-a`) sha256: hotpotqa-dev `0b498fd1ec26359a4bb1f18eede3e3ecad4babaa8ecfffafffaa830aa657a330`, musique `4810eb4e62e95233df523cf35f289e92df2f9276e01fc3e001ac2f2e29db7ad0`, multihop-rag `bffefb1cb8bea53ee44c3b0acb22ffdec0f123f2ecd7d0cbbc863f145c9a2b0b`.
- The source document of a unit is its `title` field in the old corpus file (read in place, digest-checked by `OldData`).
  HotpotQA: 5,233,329 units and 5,233,329 distinct titles, so the per-document cap below never acts there.
  MuSiQue: 101,962 units, 84,459 titles, 4,854 titles with more than one unit, at most 122 units under one title.
  MultiHop-RAG: 27,989 units, 609 titles (articles), at most 456 units under one title.
- Boilerplate in MultiHop-RAG, checked by hand on samples: bodies such as "advertisement" (under 32 titles), "click here to get the fox news app" (28) and share-button rows; 434 distinct bodies appear under two or more titles (measured).
  MuSiQue has 3 such bodies.
- The harness scores any depth-100 ranking file with every metric of record (`edge-rag score`, Phase 02 C1), so fused rankings need no new scoring code; the paired test is `metrics.paired` (exact McNemar).
- Gold-informed, exploratory, computed while writing this spec, before the constants were frozen (terrain, where looking is allowed): questions with three or more gold units under one title are 8 of 2,255 on MultiHop-RAG, 1 of 2,417 on MuSiQue and 0 on HotpotQA; one MultiHop-RAG gold body appears under two titles; no MultiHop-RAG gold unit has five words or fewer.
  The first count is a lower bound on the cap's exposure, not a measure of its harm: the cap also demotes a single gold unit when two non-gold units with its title are kept above it; C6 carries the harm figure.

## What stays fixed
- Corpora, units, questions, gold and token counts: the old project's files, digest-checked through `OldData` (Phase 01).
- Metrics, as in Phase 02: FS @1,024 / @2,048 (of record) / @4,096, FS@2/5/20 units, gold share at 5, Recall@2/5/10/20/100, nDCG@10.
- Paired test: exact McNemar on per-question FS@2,048, two-sided, alpha 0.05 per comparison, no correction (stated as a limitation, not hidden).
- G-L's reported rankings of Phase 02 are used as they are; G-L is not rebuilt (its build-to-build spread is recorded in Phase 02 results).

## Frozen definition of the candidate
Two systems are built from the same three input lists per question: Dense (the Phase 02 `p10-a` file, sha256 above), BM25 (the old index, depth 100, computed here) and G-L (reported rankings, sha256 above).
The Phase 03 `dense` lists are used only for timing and the C1 equality check; the fused Dense input is `p10-a`.
- **RRF3** (control, the literature recipe): reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).
  For each unit in the union of the three lists, score = sum over the lists that contain it of 1 / (60 + rank), rank starting at 1; a list that does not contain the unit adds nothing.
  Order: score descending, then unit id ascending; the output is the top 100.
  The constant 60 is the paper's fixed value and the default in the fusion literature; it is not tuned.
  Context, not a reason to change it: Bruch et al. 2023 report on BEIR-HotpotQA, nDCG@1000, RRF with k = 60 at 73.7 over BM25 + SPLADE, but at 67.5 over BM25 + all-MiniLM-L6-v2, below BM25 alone (68.2) and below k = 5 (69.3) (`../fase-00-state-of-the-art/survey.md`); F3 fuses that kind of pair plus G-L.
- **F3** (the candidate): RRF3's order over the whole union (up to 300 units), then one greedy pass that splits it into kept and demoted units, then kept units in order followed by demoted units in order, truncated to 100.
  For each unit in RRF3 order, the first rule that applies demotes it:
  1. Boilerplate: its normalized body appears under two or more distinct titles in the corpus.
  2. Near-duplicate: the Jaccard similarity of its body's word 5-gram set with the set of any unit already kept is at least 0.8.
  3. Per-document cap: two units with its title are already kept.
  Otherwise it is kept.
- Definitions: the body is the unit's sentences joined by spaces, without the title (in code, the corpus text after its `title + ". "` prefix, equal to the sentences joined by spaces by construction); normalization is lowercase and the regular-expression word tokens `\w+` joined by single spaces; a body with fewer than five tokens has one shingle, its whole token sequence; an empty body is boilerplate.
- Sources of the constants: the cap of two per source follows web search's host crowding (at most two results per site), the charter's "at most a few passages per source document"; the near-duplicate setting (word 5-grams, Jaccard at least 0.8) follows Lee et al. 2022 ("Deduplicating Training Data Makes Language Models Better"), computed exactly instead of with MinHash; "two or more titles" is the smallest repetition across sources.
- Nothing is fitted: the constants come from the cited sources and none is varied.
  The gold-informed checks listed under Inputs were seen before freezing them (terrain, allowed); no figure of F3 or RRF3 existed then.

## Selection rule
There is one candidate, F3.
RRF3 is its control, not a second candidate, and is never the exam entrant from this phase (research protocol: a candidate that fails a gate is not replaced inside the phase).
The frozen rule below is applied by code to terrain figures only.
- Per set and comparison, the state is `win` when wins > losses and p < 0.05, `loss` when losses > wins and p < 0.05, `tie` otherwise, and `not run` when the set was not run.
- A set that was not run is never a win and never a loss; with HotpotQA not run, advancing needs a win on both other sets.
- F3 **advances** when it wins against G-L on at least two of the three sets and loses to G-L on none.
- F3 is the **exam entrant** (Phase 06) when it advances, wins against RRF3 on at least one set, loses to RRF3 on none, and loses to the best light-class system so far on no set; otherwise there is no entrant from this phase.
  The second bar tests the candidate's own claim (the diversity pass adds something); the third keeps out of the exam a system that an already measured light-class system beats.
- The verdict is written by code as `entrant`, `advances, no entrant` (naming each bar that failed) or `does not advance`.
- RRF3's verdict against G-L under the same advance rule, and the comparisons with the best system so far, are reported as context; they do not change the entrant.

### Class check
- Code checks the light-class bounds per set, per component and per hardware, labelled derived: GPU online seconds per question against 0.1 s (the bound is stated for one RTX 4090; the hardware actually used is named), laptop CPU online seconds per question against 2 s.
- A set where a per-hardware sum exceeds its bound is reported as "outside the light class on this set (derived)" beside the verdict; the verdict is not changed (DS10).
- Known before running (derived from Phase 02 measurements): G-L's search alone takes 0.358 s per question on an A100 on HotpotQA (`g-l.manifest.json`), above the 0.1 s bound, so G-L, and therefore F3 and RRF3, are outside the light class on HotpotQA.
  On MuSiQue (0.025 s) and MultiHop-RAG (0.012 s), both on an RTX 4090, G-L is inside.
- GPU and laptop seconds are never summed into one figure checked against a bound; a cross-hardware total is shown only as context, labelled derived.

## Controls
The working controls, from Phase 02 results (`data/phase02/results.json`), on the same questions:
- The light-class ghost G-L (primary, the advance gate; weaker than the old fitted fusions, see Objective).
- The literature's weight-free fusion, RRF3, reproduced here on the same lists (a bar for the entrant).
- The best light-class system so far, by FS@2,048 per set among `p10-a`, `p10-b`, `p10-c`, `p14` and `g-l`: today P14 on HotpotQA (5,224) and MuSiQue (761), P10-B on MultiHop-RAG (587) (measured, Phase 02).
  These are the old fitted fusions: P10-C and P14 are in-sample on HotpotQA, and their weights are not weight-free.
- The best system so far, by FS@2,048 per set among every system in Phase 02 results: today `j-union` (rerank class) on HotpotQA (5,922) and MultiHop-RAG (846), `g-a1` (LLM class) on MuSiQue (1,101) (measured, Phase 02).
Said plainly: the strongest cheap literature recipe for this task is a convex combination of a lexical and a learned retriever (Bruch et al. 2023: 75.1 against 73.7 for RRF, nDCG@1000 on BEIR-HotpotQA, context only, another metric and depth), but it fits a weight; the weight-free class has RRF as its best published recipe, so RRF3 is the literature control and the fitted P10-B / P14 stand in for the convex combination.
The comparison against the best system so far crosses cost classes: a loss there is expected and is reported as a cost-quality position, not a failure.
In-domain caveat on HotpotQA: BGE-small was fine-tuned on HotpotQA train and answerai-colbert-small-v1's training mix includes HotpotQA.

## Scope
In:
1. A `components` command: Dense and BM25 depth-100 rankings per set, from the Phase 01 pipeline, written once with a manifest and per-retriever timing.
2. A fusion module (RRF and the diversity pass) with tests, and a `fuse` command that writes RRF3 and F3 rankings per set with a manifest.
3. Boilerplate flags per corpus (offline, per unit) and gold-informed demotion counts per rule (exploratory diagnostics).
4. A results table for the phase (JSON written by code and `results.md` generated from it) with every metric, labels, cost columns, paired tests and the outcome states.

Out: any GPU or pod work, any money; a G-L rebuild; any other fusion constant, cap or threshold; adding G-R, J-strong, hops or G-A1 lists to the fusion; the exam corpus; a reader or answer metric.
If something would need a GPU, it is out of scope and recorded as such.

## Acceptance criteria
Frozen on approval.
Changing them requires a deviation.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | `edge-rag components` writes `dense` and `bm25` depth-100 rankings for every question of the three sets into `data/phase03/rankings/<set>/`, once, with a manifest (input digests, code commit, laptop seconds per retriever, peak RSS); every `dense` list equals the `p10-a` list of the same question, and the 0.5 / 0.5 fusion of the same Dense and BM25 hits (`fuse_lists`, `config.WEIGHTS_P10B`, top 100) equals the `p10-b` list of the same question (each 7,405 + 2,417 + 2,255 of 12,077) | command output and manifest; both equality counts written by the command |
| C2 | The fusion code passes tests that pin: RRF scores and order on a hand-computed example, the unit-id tie-break, absent units adding nothing, the demotion order of the three rules, a kept-then-demoted output, truncation at 100, the 5-gram and short-body shingles and the boilerplate flag | `uv run pytest -q` |
| C3 | `edge-rag fuse` writes `rrf3` and `f3` depth-100 rankings for every question of the three sets, once, with a manifest pinning the sha256 of its three inputs, the constants (60, 2, 5, 0.8), the code commit, the peak RSS, the count of boilerplate units per corpus and the laptop seconds (offline flags, online per question) | manifest in `data/phase03/rankings/<set>/`, digests listed in the plan |
| C4 | `data/phase03/results.json` and `results.md`, written by code, list per set RRF3, F3, G-L, the best light-class system so far and the best system so far with every metric of record, a label per value, and cost columns: offline and online, seconds and USD, per component and in total, with the hardware each was measured on; a row whose cost was not measured (old fitted fusions and old references) carries the inherited "not measured" label, plus any figure the handover records, labelled with its source; the class check of the selection rule | files; `results.md` regenerated from `results.json` byte-equal |
| C5 | The same table gives the exact McNemar on FS@2,048 of F3 and RRF3 against G-L, of F3 against RRF3, and of both against the best light-class system and the best system so far, per set, with wins, losses, ties, p and the state; F3's verdict, RRF3's context verdict and the exam entrant are written by code under the selection rule | `results.json` fields; a test of the state and verdict code that pins each bar and the `not run` case (HotpotQA not run, F3 wins MuSiQue and ties MultiHop-RAG: `does not advance`) |
| C6 | Gold-informed diagnostics, labelled exploratory: per set and per rule, the questions where a gold unit demoted by that rule was inside RRF3's FS@2,048 context (`metrics.fill_context`) but not inside F3's; the same count for gold units demoted out of the top 100, as a second field; the count of gold units flagged boilerplate | `results.json` fields |
| C7 | Money: no pod is created and no paid API is called in this phase; spend 0 USD; money left inside the 25 USD authorization stays 10.22 USD (derived, deviation 02.1) | RunPod balance read at the close equals 14.49 USD (measured in deviation 02.1), or the billing explorer shows no charge dated inside the phase; the reading is recorded in the plan |
| C8 | `npm run check` passes | exit 0 |

## Cost accounting
- Money: 0 USD in this phase (laptop only); G-L's pod cost is Phase 02's and is shown as a component, not spent again.
- Offline per corpus: BM25 build (laptop seconds, measured here); boilerplate flags (laptop seconds, measured here); G-L encode and index (pod seconds and time x rate USD from the Phase 02 manifests, measured and derived).
  Dense corpus embeddings come from the old project: MultiHop-RAG 41.20 s of encoding on the old pod, inside 0.0404 USD attributable to GLiNER + BGE (measured, old project; handover cost table, 16.results.md section 3); HotpotQA and MuSiQue "not recorded".
- Online per question, each its own row with its hardware: BGE-small question encoding (laptop CPU seconds, measured here on a fixed sample, the first 200 questions of each set in question-file order, batch size 1); Dense and BM25 retrieval (laptop CPU seconds, measured here, retrieval only, cached question vectors); G-L search (pod GPU seconds from Phase 02, measured there, query encoding included); RRF and diversity (laptop seconds, measured here).
  If BGE-small cannot run on the laptop, the encoding row reads "not measured, cached vectors" and every online total and class check is labelled a lower bound.
- The class check is per hardware (selection rule); a cross-hardware total is context only, labelled derived.

## Risks
- HotpotQA has one unit per title: F3 can differ from RRF3 there only through boilerplate and near-duplicates (measured corpus property; expected small, interpretation).
- G-L is weaker than the old fitted fusions on all three sets (Phase 02: 5,000 / 533 / 187 against 5,224 / 761 / 587); RRF may dilute strong lists with weak ones, so RRF3 may lose to the best light-class system (interpretation, not a measurement).
- The boilerplate rule demotes a gold unit whose body is repeated under another title (one such body on MultiHop-RAG, exploratory); C6 reports the harm.
- The cap demotes a gold unit whenever two units with its title are already kept above it, gold or not; MultiHop-RAG holds up to 456 units per article, so this can happen on many questions there and is the cap's main harm path (interpretation); C6 reports it at the reading budget.
- RRF with k = 60 over BM25 and a small dense model scored below BM25 alone in Bruch et al. 2023 (BEIR-HotpotQA, nDCG@1000); RRF3 and F3 may inherit that weakness (interpretation).
- G-L, and so F3, is outside the light-class online bound on HotpotQA (derived, class check); the HotpotQA result is a cost-quality position on that hardware, not a light-class result.
- G-L is not bit-stable across builds (Phase 02, F5); the result holds for the reported build.
- Twenty-one paired tests (seven per set) at alpha 0.05 without correction: a single significant comparison outside the advance rule is reported, not acted upon.
- HotpotQA laptop runs take about half an hour for Dense and BM25 (Phase 01 measured 1,888.4 s of retrieval for the four old systems).
  Memory: Phase 01 recorded a working set of about 5.4 GB during the BM25 build, peak not recorded, and the passage vectors take about 8 GB; titles for 5.23 M units and the 8-byte body hashes add about 0.5-1 GB (projection); the HotpotQA runs record their peak RSS.
  If the laptop cannot finish, HotpotQA is reported as not run (a deviation), not moved to a pod, and the selection rule's `not run` case applies.

## Anti-goals
- No tuning of k, cap, shingle size or thresholds; no second variant after a result is seen.
- No pre-measurement of F3 or RRF3 before this spec; no per-phase stage code beyond the two commands and the results table.

## Decisions taken
By the agent under the author's delegation, 2026-10-03; none is open.
- DS1. Fuse Dense, BM25 and G-L, and no other list. Reason: they are the light-class retrievers with depth-100 lists on all three sets at zero money; hops need GLiNER outputs and fitted P14 settings, G-R and J-strong are rerank class, G-A1 is LLM class and missing on HotpotQA.
- DS2. RRF with k = 60. Reason: the literature's fixed constant, weight-free; ranks are all the input files hold, so score-based fusion would need recomputing scores and normalizing them, and convex combination needs a fitted weight.
- DS3. Demote instead of remove. Reason: a removed unit cannot return; demoting keeps the list at depth 100 and leaves the reading budget to decide.
- DS4. Source document = corpus title. Reason: it is the only source field the units carry; on HotpotQA it is the article, on MultiHop-RAG the news article, on MuSiQue the Wikipedia page.
- DS5. Boilerplate = body repeated under at least two titles, not a length rule. Reason: repetition across sources is label-free and corpus-agnostic; a word-count rule would need a threshold, and short units such as headings are often unique, not boilerplate.
- DS6. One preregistered candidate, with RRF3 as its control and a bar for the entrant, never an entrant itself. Reason: no variant is chosen by a figure, a failed candidate is not replaced inside the phase, and the diversity pass's contribution is read from F3 against RRF3.
- DS7. BM25-only lists are computed in a new `components` command that reuses the Phase 01 loading code and writes to `data/phase03/`. Reason: Phase 02's ranking directory and manifest, and Phase 01's `data/results/reproduce-<set>.json`, belong to closed phases and are not written; the Phase 01 gate rerun writes to `data/phase03/gate/`.
- DS8. G-L's online cost enters F3's cost as measured on the pod in Phase 02. Reason: no GPU in this phase; the PLAID search is not reproducible on the laptop CPU within the phase.
- DS9. Fuse `p10-a`, not the new `dense` file. Reason: one digest chain from the frozen sha256; `dense` exists only for timing and the equality check.
- DS10. The class check is per hardware and does not change the verdict. Reason: G-L, the class ghost, is itself outside the bound on HotpotQA; changing the verdict on it would penalize the candidate for its control's cost.
- DS11. BGE-small question encoding is timed on the laptop on a fixed sample of 200 questions per set. Reason: the old runs read cached question vectors, so Dense's online cost would otherwise leave out its most expensive step; a fixed sample keeps the cost of measuring small.
- DS12. The exam entrant needs, besides advancing against G-L, a win over RRF3 and no loss to RRF3 or to the best light-class system so far. Reason: G-L is a weak gate, and a candidate should enter the exam on its own claim, not on a cheaper system's strength.

## Open decisions
None.
