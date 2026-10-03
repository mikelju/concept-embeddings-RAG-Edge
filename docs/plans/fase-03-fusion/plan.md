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
| 3 | `edge-rag fuse --set <set>`: reads `p10-a` and `g-l` (phase 02, against the spec's sha256) and `bm25` (phase 03, against its manifest's sha256), writes `rrf3` and `f3` with a manifest (input sha256, constants, commit, boilerplate count, seconds offline and per question), and the exploratory per-rule gold demotion counts at the FS@2,048 budget and at depth 100 (spec C6) | C3, C6 | laptop | done | Code 081f974 (one extra test pins the C6 counting on a toy question). `uv run edge-rag fuse --set multihop-rag`: rrf3 and f3 written, manifest with the three input sha256, constants (60, 2, 5, 0.8), commit, peak RSS, boilerplate count and seconds. `uv run edge-rag score --set multihop-rag --rankings .../f3.jsonl.gz --against .../g-l.jsonl.gz`: FS@2,048 490 against 187, wins 355, losses 52, ties 1,848, exact p 1.458e-56, the same figures as the table (measured). |
| 4 | Results: `edge-rag fusion-results` writes `data/phase03/results.json` and `results.md`: metrics, labels, cost columns per component and total, the seven paired tests per set, states, advance verdicts and exam entrant by the spec's rule; class check per component and hardware; tests of the state, verdict and `not run` code | C4, C5, C6, C8 | laptop | done | Code 29fbed0; `tests/test_fusion_results.py` pins the state code, every bar of the verdict, the spec's `not run` case (`does not advance`), the per-hardware class check and the outcome with a set not run; `uv run pytest -q` all pass. `uv run edge-rag fusion-results` run twice: `results.md` sha256 4c464057cce5ed2e97c7c4b7262b95f7889dca24fa245dc905beea467235acb4 and `results.json` cc39edcfc24ae2a326b69fa3e251be4c4f5b9791c66efe57f22d4a3fc419aa5a both times (byte-equal). The table rescoring of every Phase 02 row is checked equal to Phase 02's metrics, and each read ranking file against its recorded sha256. |
| 5 | Runs: components and fuse on MultiHop-RAG, MuSiQue, HotpotQA, peak RSS recorded; results table; one fused list checked by hand per set (RRF score of the top unit recomputed from the three inputs; one demotion explained) | C1, C3-C6 | laptop | done | See "Runs (increment 5)" below. |
| 6 | Validation: RunPod balance reading (C7), `npm run check`, adversarial review, results by criterion, master plan status, delivery on the branch with a PR | C7, C8 | laptop | pending | |

How each increment is checked:
- 1: `uv run edge-rag components --set multihop-rag` prints both equality counts 2,255 of 2,255; `uv run pytest -q`; `uv run edge-rag reproduce --set multihop-rag --out data/phase03/gate` and `--set musique` print GATE PASS, and the sha256 of `data/results/reproduce-<set>.json` and of the Phase 02 ranking files are the same before and after.
- 2: `uv run pytest -q tests/test_rrf.py`.
- 3: `uv run edge-rag fuse --set multihop-rag`; `uv run edge-rag score --set multihop-rag --rankings data/phase03/rankings/multihop-rag/f3.jsonl.gz --against data/phase02/rankings/multihop-rag/g-l.jsonl.gz` agrees with the table.
- 4: `uv run pytest -q`; `results.md` regenerated byte-equal.
- 5: manifests and sha256 listed in this plan; the hand checks written here.
- 6: RunPod balance reading recorded under Money; `npm run check` exit 0.

## Runs (increment 5)
All on the laptop CPU (ARM64, Windows 11), one process at a time, logs in `data/phase03/logs/`.
Manifests record the HEAD at write time; the producer code (`components`, `reproduce`, `corpus`, `fuse`, `rrf`, `config`) has no diff between d194813, 081f974 and 29fbed0 (`git diff --stat`, empty), so each recorded commit holds the code that wrote the file.

| Set | `components` commit | Equality dense / fusion | BM25 build s | Dense / BM25 retrieval s per q | BGE encoding s per q | Peak RSS MB (components / fuse) | `fuse` commit | Flags s | RRF / diversity s per q | Boilerplate units |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| multihop-rag | d194813 | 2,255 / 2,255 of 2,255 | 1.148 | 0.00125 / 0.00047 | 0.0408 | 547.5 / 170.2 | 081f974 | 0.211 | 0.000099 / 0.0113 | 1,241 |
| musique | d194813 | 2,417 / 2,417 of 2,417 | 5.900 | 0.00431 / 0.00092 | 0.0183 | 719.0 / 238.1 | 29fbed0 | 1.247 | 0.000107 / 0.0468 | 6 |
| hotpotqa-dev | 29fbed0 | 7,405 / 7,405 of 7,405 | 212.721 | 0.1511 / 0.0504 | 0.0240 | 13,478.7 / 4,347.3 | 29fbed0 | 45.575 | 0.000108 / 0.0388 | 15,925 |

All measured (manifests); BGE encoding min cosine to the cached question vectors is 1.0 on every set.
HotpotQA ran Dense and BM25 retrieval in 1,495.3 s and fitted in memory (peak 13.5 GB of 32 GB), so the spec's `not run` fallback was not needed.

sha256 (`data/phase03/rankings/<set>/`):
- hotpotqa-dev: bm25 0c3e1a3d51798b4b9db41391cd0acce65f11b8198e1e5450eb3e8c96f098f376, dense 0b498fd1... (= p10-a), rrf3 83960852e24694c46e07ed897e9716826b965252244887c1846d07ae1a672026, f3 9ac0e74c1c7346a68e88b825813d0b6618306e8d6d5c846a259333ffc3704769, components.manifest.json 919008adeee84e2c2de973a41a5eb83127e6a928af2a0b7d2bf1397ccfb75b31, fuse.manifest.json 4f95819d1fa98ec3a04593a54008b7b90b60f9e4cb1c5596eb261ba66a9673b8.
- musique: bm25 7931e75641b5d6cc7c5bea5fa6659134dfc592af4cb6dff4efa23e2f2f1fdd0a, dense 4810eb4e... (= p10-a), rrf3 f3934e7e40c0d25081676fc27af07002e901a1ab6050a30c4ceb5f866bffcc1f, f3 2793dc7bd50e8c9855ebc71cde7658428230c6efd9e5262aaad34b23e7ffff22, components.manifest.json f1c178caad81d262be7448a3e8b59a2718e16636a803881a8c508106ed6b1d70, fuse.manifest.json a0887ee408a1694656d1d0c1f63c3518387e5505eca1f7f4c9fa9aba6bc40eb6.
- multihop-rag: bm25 8ca1e9cd5f5713f8890813b205770f362123d783baf435fe426c77ced18bc178, dense bffefb1c... (= p10-a), rrf3 402106e371b4c8933bc1d3cea663b8ac678070efecee5865c1f1c4d756256de5, f3 a682a597eee82305646ab6b798c11febfb8d38c560fdf39f40eb9930aaffe482, components.manifest.json 6510dde302a4055f6689137b5c230d3d97e4f2dfdbfd5b46f30c0ce58c423160, fuse.manifest.json e02b44eafbe6102d78692bb89281d872ac6a2e4b5bf2e108f0bd3422b3ed8a2c.

Hand checks (scripts in the session scratchpad, reading the files above with the harness loaders):
- multihop-rag, qid mhr-0000: RRF3's top unit 745f28b71648586d sits at ranks 7 / 2 / 3 in Dense / BM25 / G-L; 1/67 + 1/62 + 1/63 = 0.046927, the code's score. F3 moves it to position 82: its normalized body (a recurring introduction on Sam Bankman-Fried's trial) appears under 4 article titles, so rule 1 (boilerplate) demotes it. Unit 26585272f0b17822 (RRF3 position 9) goes to 86 by rule 3: two units with its title ("The FTX trial is bigger than Sam Bankman-Fried") were already kept, at F3 positions 1 and 4.
- musique, qid 2hop__460946_294723: top unit 4fb6ff21c7274493 at ranks 6 / 1 / 2, 0.047674 by hand and by the code. In qid 2hop__3131_3300, unit 7203c67deeb60f98 ("American Idol (season 3)", RRF3 position 3) goes to 71 by rule 2: Jaccard 0.810 of its 5-gram set with the kept unit 16e4323e14c87eef under the same title.
- hotpotqa-dev, qid 5a8b57f25542995d1e6f1371: top unit 45558f75b4cfebcb at rank 1 in all three lists, 3/61 = 0.049180 by hand and by the code. The first demotion inside RRF3's top 100 found (qid 5adbf0a255429947ff17385a, unit fccaea325becd212, position 96, dropped out of the top 100) is rule 1: the same body is the text of two titles, "Sokollu Mehmed Pasha Mosque (Azapkapi)" and "(Buyukcekmece)". Rerunning `diversify` from the inputs reproduced the stored F3 list for every question up to that one.

Outcome written by code (`results.md`): F3 **advances, no entrant** (wins against G-L on HotpotQA and MultiHop-RAG, ties on MuSiQue; no win against RRF3, a loss to RRF3 on MuSiQue and MultiHop-RAG, a loss to the best light-class system so far on all three sets); exam entrant: none from this phase; RRF3's context verdict against G-L: advances.

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
