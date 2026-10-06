---
name: research-protocol
description: Experimental protocol of this project - terrain/exam separation, preregistered selection, the baseline rule, cost recording, artifact integrity, value labels and known traps. Load before specifying, planning, implementing or reviewing anything that produces, selects on or reports a measured retrieval result. Not needed for read-only questions.
---

# Research protocol

Carried from `concept-embeddings-RAG` (handover section 7), adapted to the tournament.
Where a rule names a past phase, the record is in that repository's `docs/plans/phase_N/`.

## Terrain and exam

- **Terrain**: corpora open for choosing (HotpotQA FullWiki, MuSiQue, MultiHop-RAG unless the master plan says otherwise).
  Any figure on them may be looked at while choosing.
- **Exam**: one fresh corpus, locked, run once, with the chosen candidates and the ghosts, under a rule frozen before it is opened.
  Nothing is chosen, fitted or rerun after it is opened.
- A new terrain corpus records, before anything runs on it, whether a dev subset is carved or the whole set is open.
- **No labels at use time**: a strategy may be chosen on open corpora, but it runs on a new corpus with nothing fitted to it.
  Fitted weights do not travel between kinds of text (old Phase 16); prefer rules with nothing fitted, or report a fitted figure as an upper bound.
- A candidate that fails a pre-declared gate is not replaced by another one inside the same phase; a new candidate is a new written experiment (old Phase 8, and the spaCy case of old Phase 7).
- Never pre-measure a candidate before its spec; only oracle ceilings, labelled exploratory, may run.

## The baseline rule

- Every spec names the strongest cheap literature recipe for its task, with its published figure, and makes it the control or says in one sentence why not.
- If the working control is weaker, tell the author plainly and first: the old line compared 16 phases against a weak control.
- **Ghosts** are literature reference systems reproduced on this harness, one per cost class; a paper's figure under a different metric or unit is context, never a comparison.
- **Two comparisons**: every new system states its gain against the ghost of its class and against the project's best system so far.
- Before recommending a stronger or larger model, check this project's own measurements (old Phase 8: a larger Dense was weaker).

## Cost is half the result

- Every result carries its cost beside its score: offline (indexing, extraction) and online (per question), time and money, separately.
- Each candidate competes in its cost class; a strategy is interesting if it is on the quality-cost frontier.
- Money is recorded as three distinct labelled figures when they exist: time x rate (derived), balance delta and invoice (measured).

## Specifying a phase

- One main question per phase, roughly 5-8 implementation steps; reuse code, do not generalize pre-emptively.
- No branches or guards for failures that have not occurred; test code that can change a result, not every invariant.
- No elaborate statistical framework when a direct paired comparison answers the question (exact McNemar on the same questions).
- A spec fixes the question, what stays fixed, what is measured, how the decision is made (bars, ranking, stop states) and the anti-goals.
  Once approved it is frozen; a real problem outside it is a deviation for the author.
- When verification finds the code right and the spec stale, correct the spec visibly, in its own commit, with no code change.
- Check one example by hand before a count enters a spec (old deviation 16.1: "29 facts cross a paragraph break" were sentences printed twice).
- Implementation effort weighs little when choosing between designs: agents make code cheap, a weak experimental design is expensive.

## Labels

Every reported value carries one: **measured** (read from an artifact or a command), **derived** (arithmetic on measured values),
**exploratory** (gold-informed or fitted on the figures it is measured on; decides nothing), **interpretation** (a reading, not tested),
**projection** (neither measured nor derived). Terminal states and comparison labels are written by the outcome code from the run files, never by hand.

## Artifact integrity

- Expensive deterministic outputs (embeddings, indexes, extractions, judge scores) are cached on disk and reused.
  A different model or configuration writes a distinct cache; it never overwrites the artifact behind a result.
- A cached result is reused only when every input of its key matches, the commit and the dirty state of `src/` included; any mismatch recomputes.
- A digest chain from extraction to final run, so a figure cannot be detached from the reading and selection that authorized it.
- Non-executing formats only (JSON, JSONL, NPZ); no pickle. Pin model identity and revision for every download.
- A provenance problem in a closed phase gets a prospective fix in the producer, never an edit of the frozen artifact.
- The old project's `data/` is read in place, read-only, each file checked against its recorded digest.

## Known traps

- Every corpus gets its own caches: the old `question_cache_key` does not include the corpus.
- Tokenize large corpora in batches: the old `TokenCounter.count_units` tokenizes its whole input in one call.
- The old exact McNemar overflowed past 1,024 discordant questions; take the fixed version (commit `9a9ad3f`).
- bfloat16 scores can reorder under batching (old J-decision); a judge or model in bfloat16 needs a determinism check as a gate.
- HotpotQA dev is in-sample for the old P10-C and P14 weights; BGE-small was fine-tuned on HotpotQA train.
- MuSiQue mined its distractors with BM25, which works against lexical retrieval.
- For mostly single-evidence corpora, Full Support degenerates to Recall at the budget; state the metric before opening.

## Anti-patterns

- **Assuming a richer or larger system is better.** Graphs, LLM loops and bigger models are candidates, measured at their cost, not upgrades.
- **Reporting a projection, an in-sample refit or an exploratory ceiling as a result.**
- **Overengineering.** A small research phase stays small; one generic tournament runner, not per-phase stage code.
