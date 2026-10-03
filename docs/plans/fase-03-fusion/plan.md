# Phase 03 - Weight-free fusion with source diversity: plan

Status: in progress
Spec: `spec.md` (approved by delegation, 2026-10-03)
Base: branch `fase-03-fusion` from `main` at 010cb33

## Decisions
Taken by the agent under the author's delegation; the design decisions are DS1-DS8 in the spec.
- D1. `components` reuses `reproduce.py`'s loading and digest checks through a shared helper extracted from `reproduce.run`, not a copy; `reproduce` gains an `--out` directory (default unchanged) so the gate rerun writes to `data/phase03/gate/`, never to `data/phase02/rankings/` or `data/results/`. Reason: one loading path, already gated in Phase 01, and closed phases' files are not rewritten (DS7).
- D2. The fusion code goes in `src/edge_rag/retrieval/rrf.py` (pure functions over rank lists, bodies and titles); the existing min-max `fusion.py` is not touched. Reason: the old fusion is frozen behind the Phase 01 gate.
- D3. `Corpus` gains a `titles` field; the body is `text[len(title) + 2:]`, which `load_corpus` builds as `f"{title}. {' '.join(sentences)}"`, so it equals the sentences joined by spaces (spec definition); a test pins it with a title containing ". " and with an empty sentence list. Reason: the cap needs the title and the diversity rules need the body without it, and slicing avoids a second copy of 5.23 M bodies; the change is checked by the Phase 01 gate rerun at the end of increment 1.
- D4. Boilerplate flags are computed by hashing each normalized body (sha1, 8 bytes) to the set of title hashes; HotpotQA's 5.23 M bodies fit in memory this way. Reason: a dict of full bodies would double the corpus's memory.
- D5. Phase 03 outputs live in `data/phase03/` (rankings, manifests, `results.json`); `results.md` is generated into this folder. The Phase 03 table reads Phase 02's `results.json` for the old references and G-L, read-only.
- D6. Order of runs: MultiHop-RAG, then MuSiQue, then HotpotQA (the long one, run in the background with timed logs).
- D7. The HotpotQA gate is not rerun (about an hour of laptop time); the D1 and D3 changes are checked there by C1's two equality counts (`dense` against `p10-a`, Dense + BM25 fusion against `p10-b`) over all 7,405 questions. Reason: those counts cover the same code path and the BM25 lists end to end.
- D8. BGE-small question encoding is timed by `components` on the first 200 questions of the set, batch size 1, laptop CPU (spec DS11), with the model at `config.EMBEDDING_REVISION`. Reason: frozen sample, no new dependency (torch and transformers are already in the project).

- D9. Manifests name the commit that holds the code that wrote them: the first MultiHop-RAG and MuSiQue `components` runs were made on an uncommitted tree and recorded fd638e5, so they were rerun after commit d194813 (identical ranking files, write-once check passed), and the HotpotQA run was restarted for the same reason. Reason: digest chain (research protocol).

## Increments
| # | Increment | Criteria | Where | Status | Evidence |
|---|---|---|---|---|---|
| 1 | `edge-rag components --set <set>`: Dense and BM25 depth-100 lists, manifest (input digests, commit, seconds per retriever and per question, question-encoding seconds on the D8 sample, peak RSS), counts of `dense` lists equal to `p10-a` and of Dense + BM25 fusions equal to `p10-b`; `Corpus.titles` (D3); `reproduce --out` (D1); tests of the manifest, both equality counts and the body slice on a toy set; then the Phase 01 gate rerun on MultiHop-RAG and MuSiQue into `data/phase03/gate/`, before any other run | C1, C8 | laptop | done | Code d194813. `components` at d194813: MultiHop-RAG `dense_equals_p10-a` 2,255 of 2,255 and `fusion_equals_p10-b` 2,255 of 2,255; MuSiQue 2,417 of 2,417 both (measured, `components.manifest.json`). BGE-small question encoding, min cosine to the cached vectors 1.0 on both sets (measured), so the timed encoder is the one behind the lists. `uv run pytest -q`: all pass (3 new tests in `tests/test_components.py`). Gate rerun into `data/phase03/gate/`: GATE PASS on MultiHop-RAG and MuSiQue, every stored list equal (logs `data/phase03/logs/gate-<set>.log`); `sha256sum -c` of the 32 Phase 01-02 files (`data/results/reproduce-*.json`, `data/phase02/rankings/*/*.gz` and manifests) taken before the rerun: all OK after it. `dense` sha256 equals `p10-a` on both sets; `bm25` sha256: multihop-rag 8ca1e9cd5f5713f8890813b205770f362123d783baf435fe426c77ced18bc178, musique 7931e75641b5d6cc7c5bea5fa6659134dfc592af4cb6dff4efa23e2f2f1fdd0a. HotpotQA components is part of increment 5. |
| 2 | `retrieval/rrf.py`: `rrf(lists, k=60, depth)`, `shingles(body)`, `jaccard`, `boilerplate_flags(bodies, titles)`, `diversify(order, titles, bodies, flags, cap=2, threshold=0.8, depth)`; tests listed in C2 | C2, C8 | laptop | done | `uv run pytest -q tests/test_rrf.py`: 5 passed (hand-computed RRF scores and order, unit-id tie-break, absent units add nothing, truncation at 100 of a 300-unit union, 5-gram and short-body shingles, boilerplate flag including an empty body and a body repeated under one title only, rule order boilerplate > near-duplicate > cap, kept-then-demoted output and truncation). The near-duplicate scan skips pairs whose set sizes alone rule out Jaccard >= 0.8 (exact bound, with a 1e-9 margin), a speed-up that cannot change an output. |
| 3 | `edge-rag fuse --set <set>`: reads `p10-a` and `g-l` (phase 02, against the spec's sha256) and `bm25` (phase 03, against its manifest's sha256), writes `rrf3` and `f3` with a manifest (input sha256, constants, commit, boilerplate count, seconds offline and per question), and the exploratory per-rule gold demotion counts at the FS@2,048 budget and at depth 100 (spec C6) | C3, C6 | laptop | pending | |
| 4 | Results: `edge-rag fusion-results` writes `data/phase03/results.json` and `results.md`: metrics, labels, cost columns per component and total, the seven paired tests per set, states, advance verdicts and exam entrant by the spec's rule; class check per component and hardware; tests of the state, verdict and `not run` code | C4, C5, C6, C8 | laptop | pending | |
| 5 | Runs: components and fuse on MultiHop-RAG, MuSiQue, HotpotQA, peak RSS recorded; results table; one fused list checked by hand per set (RRF score of the top unit recomputed from the three inputs; one demotion explained) | C1, C3-C6 | laptop | pending | |
| 6 | Validation: RunPod balance reading (C7), `npm run check`, adversarial review, results by criterion, master plan status, delivery on the branch with a PR | C7, C8 | laptop | pending | |

How each increment is checked:
- 1: `uv run edge-rag components --set multihop-rag` prints both equality counts 2,255 of 2,255; `uv run pytest -q`; `uv run edge-rag reproduce --set multihop-rag --out data/phase03/gate` and `--set musique` print GATE PASS, and the sha256 of `data/results/reproduce-<set>.json` and of the Phase 02 ranking files are the same before and after.
- 2: `uv run pytest -q tests/test_rrf.py`.
- 3: `uv run edge-rag fuse --set multihop-rag`; `uv run edge-rag score --set multihop-rag --rankings data/phase03/rankings/multihop-rag/f3.jsonl.gz --against data/phase02/rankings/multihop-rag/g-l.jsonl.gz` agrees with the table.
- 4: `uv run pytest -q`; `results.md` regenerated byte-equal.
- 5: manifests and sha256 listed in this plan; the hand checks written here.
- 6: RunPod balance reading recorded under Money; `npm run check` exit 0.

## Money
Phase 03 spends 0 USD: no pod, no paid API (C7).
Money left inside the project's 25 USD authorization at the start of the phase: 10.22 USD (derived, deviation 02.1); it is unchanged at the end unless a later entry here says otherwise.
Check at the close (C7): the RunPod balance equals 14.49 USD (measured in deviation 02.1), or the billing explorer shows no charge dated inside the phase; the reading goes here.

## Deviations
| ID | Summary | Affected criteria | Status |
|---|---|---|---|

## Results by criterion
Filled at the close from the run files.

## Findings
None yet.

## Adversarial review
| Round | Backend | Range | Lenses | Findings | Status |
|---|---|---|---|---|---|

## Candidate learnings
Only reusable lessons with a quote from the session; consolidated at the close.
