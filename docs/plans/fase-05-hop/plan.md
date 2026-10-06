# Phase 05 - Multi-seed convergent hop with a refined query: plan

Status: closed, delivered with a PR (2026-10-05)
Spec: `spec.md` (approved by delegation, 2026-10-04; frozen)
Base: branch `worktree-fase-05-hop` from `main`, spec commit bc3fb9e

## Decisions
Taken by the agent under the author's delegation when writing this plan; the design decisions are DH1-DH18 in the spec.
- D1. Phase 05 outputs live in `data/phase05/` (`rankings/<set>/`, `logs/`, `results.json`); `results.md` is generated into this folder, as in Phases 03 and 04.
  `config` gains `PHASE05_DIR`, `PHASE05_RANKINGS_DIR` and `SEEDS_PER_LIST = 5`; `RRF_K` (60), `DEPTH` (100) and `READ_DEPTH` (10) are reused.
  Reason: same layout as Phases 03 and 04; closed phases' folders are read only.
- D2. The worktree reaches the main checkout's `data/` through a directory junction, and the old project's data through `OLD_DATA_ROOT`.
  Evidence: `config.REPO_ROOT = Path(__file__).resolve().parents[2]` and `DATA_DIR = REPO_ROOT / "data"` (`src/edge_rag/config.py` lines 12-14), with no environment override, so from the worktree `DATA_DIR` is `<worktree>/data`, which does not exist on a fresh worktree.
  `OLD_DATA_ROOT_DEFAULT = REPO_ROOT.parent / "concept-embeddings-RAG" / "data"` (line 18) resolves to `.claude/worktrees/concept-embeddings-RAG/data` from the worktree, so `OLD_DATA_ROOT` must be set (as Phase 01 D10 and its HotpotQA rerun recorded).
  A junction named `data` in the worktree was tested while writing this plan (to a throwaway target, then removed): `git check-ignore -v data` prints `.gitignore:12:/data/	data` and `git status --porcelain --ignored` prints `!! data/`, so the junction is ignored and never committed.
  Create it once, before increment 1, from the worktree root: `python -c "import _winapi; _winapi.CreateJunction(r'C:\Python Projects\concept-embeddings-RAG-Edge\data', r'C:\Python Projects\concept-embeddings-RAG-Edge\.claude\worktrees\fase-05-hop\data')"` (or `cmd /c mklink /J data "C:\Python Projects\concept-embeddings-RAG-Edge\data"`).
  Remove it only with `python -c "import os; os.rmdir('data')"` or `cmd /c rmdir data`, never `rm -r`, and remove it before the worktree is removed, so no recursive delete can follow it into the main checkout's run data.
  Every command in this plan runs from the worktree root with `OLD_DATA_ROOT="C:/Python Projects/concept-embeddings-RAG/data"` exported.
- D3. The shared results module is new `src/edge_rag/phase_results.py`; it takes the common code of `fusion_results.py` (states, `advances`, `verdict`, `handover_cost`, cost-row helpers, `num`, `write_pair`, `regenerate`, constants) and holds one small phase description per phase (output paths, rows, comparisons, bars, set table, page), with the Phase 03 and Phase 04 set-table and page code moved in unchanged in output.
  `fusion_results.py` and `judge_results.py` are deleted; `edge-rag fusion-results`, `judge-results` and new `hop-results` call it; Phase 02's `results.py` is not touched.
  Reason: spec C3 and DH13; one module, three descriptions.
- D4. The three results commands gain `--out DIR`, which writes `results.json` and `results.md` into `DIR` instead of the stored paths; the C3 recompute writes to `.runtime/c3/phase03/` and `.runtime/c3/phase04/` (gitignored, inside the worktree).
  In this phase `fusion-results` and `judge-results` run only with `--from-json` or `--out`, never bare, because a bare run would overwrite the stored write-once files (DH15).
  Reason: the spec's scratch path needs a target; the stored pages' headers are static text, so a page written elsewhere is still byte-comparable.
- D5. `retrieval/hop.Hops` gains `seed_hop(seed: str, excluded: Collection[str], depth: int) -> list[Hit]` beside `entity_hop`, which stays untouched, so C1's equality against the Phase 04 `hop` file is a cross-check of two code paths, not a tautology.
  `seed_hop` uses the same `node_weights`, `arm_columns` mask (entity nodes) and tie order (score descending, unit id ascending) and returns an empty list for a seed with no entity node.
- D6. New `src/edge_rag/converge.py` holds the pure functions (`seeds`, `refined_query`, `hop_ms`, `fusions`) and `run(set_name)`; `edge-rag converge --set <set>` calls it.
  `run` follows `pool.run`: `keep_manifest`, `fuse.read_inputs` for the pinned `p10-a` and `bm25` files, the Phase 04 `hop` file pinned by its manifest sha256, `reproduce.load_inputs`, `load_node_index` with the recorded digest and the row-order check, `Dense`, `scoring.write_rankings`, `write_manifest`.
- D7. Seeds, `mch`, `rrf-prf` and `rrf-1s` read the stored `p10-a` and `bm25` lists (the files the spec names); Dense and BM25 are also recomputed per question, only for the online timing rows and the C1 equality counts.
  Reason: as Phase 04 D2; C1 shows both views agree on every question.
- D8. `q' = (q + c) / ||q + c||` is computed in float64 from the cached float32 question vector and the float32 unit vectors, then cast to the unit vectors' dtype before `Dense.retrieve`; the manifest records the dtype.
  Reason: the spec fixes the formula, not the precision; one fixed choice, stated before any run.
- D9. Online timing per question (manifest `online_seconds_per_question`): `dense`, `bm25` (recomputed), `seed_hops` (sum over the seeds), `refine_and_dense_prf`, `rrf_hop_ms`, `rrf_mch`, `rrf_rrf_prf`, `rrf_rrf_1s`; the equality-check hop is not timed. The encoding row is Phase 03's (DH16).
- D10. Order of laptop runs: MultiHop-RAG, MuSiQue (inside group B, short), then HotpotQA alone in the background (increment 5); nothing else runs while a timed command runs (Phase 03 D12).
- D11. The count of `mch` top-100 units contributed only by `hop-ms` is written by `converge` (measured, as Phase 04's `rrf4_units_only_in_hop`); the questions whose gold enters `mch`'s top 100 only through `hop-ms` are counted by `hop-results`, which reads gold (exploratory).
- D12. Money readings use `.agents/skills/remote-gpu/SKILL.md` (RunPod GraphQL `clientBalance`, `myself { pods }`), taken by the main session, never by an implementation subagent.
- D13. At the close, results by criterion and candidate learnings go in this plan, not in `results.md` (Phase 03 D15).

## Groups
Each group is one implementation subagent with about 120k tokens, briefed from this table; the main session integrates, checks and commits.
- Main session: increment 0, increment 5 (the HotpotQA run), increment 8, increment 9.
- Group A (results refactor): increments 1 and 2.
- Group B (retrieval and `converge`): increments 3 and 4.
- Group C (Phase 05 results): increments 6 and 7.

## Increments
Each increment leaves the product working and covers named criteria.
This file is the durable state: a new session resumes from here and from Git.

| # | Increment | Criteria | Files | Group | Status | Evidence |
|---|---|---|---|---|---|---|
| 0 | Phase start: create the `data` junction and check it (D2); read `clientBalance` and `myself { pods }` with UTC time and record them under Money | C7 | `plan.md` | main | done | junction created with `_winapi.CreateJunction`; the ignored-files status prints `!! data/`; `ls data` lists phase02-04 and results; 2026-10-04T14:09:46Z `clientBalance` 14.1147504042 USD, `pods` [] (measured) |
| 1 | Shared results module (D3, D4): `phase_results.py` with the Phase 03 and Phase 04 descriptions; delete `fusion_results.py` and `judge_results.py`; CLI `fusion-results` and `judge-results` on the new module with `--from-json` and `--out`; tests move imports only. `--from-json` regenerates both stored pages byte-equal; full recompute of each phase into `.runtime/c3/`, JSON equal to the stored one except `git_commit` and `git_src_changes`, page byte-equal to the stored page | C3 | `src/edge_rag/phase_results.py`, `src/edge_rag/cli.py`, `tests/test_fusion_results.py`, `tests/test_judge_results.py` | A | done | commit bb645a8; fusion-results and judge-results `--from-json` print sha256 `e50c85a9...` and `6b4b6bd1...` (byte-equal); full recompute into `.runtime/c3/phase0{3,4}` gives the same two page hashes and JSON `equal` (git fields popped); stored `results.json` still `ccf4dfc9...` and `93fa9140...`; test diff import lines only; checked again by an independent verifier |
| 2 | Finding 4 fix: the judge's online row is `J-strong over N units`, N = `judged_units` of the judged lists, seconds = per-100 seconds x N / 100, label saying `x N`; test with an 80-unit judged list (name and 0.8 x seconds); increment 1's recompute rerun after the fix gives the stored Phase 04 page and figures | C4, C3 | `src/edge_rag/phase_results.py`, `tests/test_judge_results.py` | A | done | commit bb645a8; row `J-strong over {units} units`, seconds per 100 x units / 100 (factor exactly 1.0 at 100); new test with an 80-unit list; the increment 1 recompute rerun after the fix gives the stored Phase 04 page and JSON |
| 3 | Retrieval code and tests: `config` constants (D1); `Hops.seed_hop` (D5); `converge.py` pure functions; `tests/test_converge.py` pinning seed order and deduplication, seed exclusion, `seed_hop` equal to `entity_hop` with Dense's first unit and top 10 excluded, the convergence example (two seeds at rank 50 above one seed at rank 1), an empty per-seed list adding nothing, `q'` on a two-dimensional example, the four-list RRF with an empty `hop-ms` | C2 | `src/edge_rag/config.py`, `src/edge_rag/retrieval/hop.py`, `src/edge_rag/converge.py`, `tests/test_converge.py` | B | done | commit 5f5dcb3 (docstring 5f313b6); `uv run pytest -q tests/test_converge.py tests/test_data_and_hops.py` 18 passed, 1 skipped (symlink privileges); three-lens review found no behaviour defect against the spec |
| 4 | `converge.run` and `edge-rag converge --set <set>` (D6-D9, D11): `hop-ms`, `dense-prf`, `mch`, `rrf-prf`, `rrf-1s` depth-100 rankings into `data/phase05/rankings/<set>/`, write-once, `converge.manifest.json` with every C1 field and the three equality counts; run on MultiHop-RAG then MuSiQue with timed logs; one question per set traced by hand (seeds, cosine of `q'` with `q`, top `hop-ms` unit and its seeds with ranks and RRF sum), computed by a separate snippet from the stored files, not by `converge.py` | C1 | `src/edge_rag/converge.py`, `src/edge_rag/cli.py`, `plan.md` (trace) | B | done | `converge` at 5f313b6, logs `data/phase05/logs/converge-<set>.log` end `exit 0` (29.5 s, 61.6 s); equality 2,255 of 2,255 and 2,417 of 2,417 on all three checks; seed mean 8.97 and 8.79, range 5-10; `hop-ms` empty 0 and 0, short 34 and 12 (short counts every list under 100, empty ones included); peak RSS of the stored reruns 220.9 and 517.7 MB (first run 219.1 and 516.8); outputs byte-identical to the first run at 5f5dcb3; an independent script (no `edge_rag` import) reproduces `mch`, `rrf-prf`, `rrf-1s` from the stored inputs with 0 mismatches; traces below |
| | **Session boundary A**: code for all three sets is committed and the two small sets are run; the HotpotQA run starts from a clean tree. | | | | | |
| 5 | HotpotQA run on the laptop, launched by the main session in the background (see Laptop run); after it, check the manifest and record the HotpotQA hand trace | C1 | `plan.md` | main (trace: C) | done | log `data/phase05/logs/converge-hotpotqa-dev.log` ends `exit 0`, real 75m27.7s; commit 260c0f7 with `git_src_changes` empty, GLiNER digest `2f7864661b8ce7ff`, constants 5/60/100; equality 7,405 of 7,405 on all three checks; seed mean 8.59, range 5-10; `hop-ms` empty 2, short 9 (the 2 empty included); peak RSS 13,555.0 MB; the five files match the manifest sha256; trace below |
| | **Session boundary B**: every Phase 05 ranking file and manifest is on disk; the results session starts from this table. | | | | | |
| 6 | Phase 05 description in `phase_results.py` and `edge-rag hop-results [--from-json] [--out DIR]`: per set `mch`, `rrf-prf`, `rrf-1s`, `hop-ms`, `dense-prf`, G-L, best light-class so far (read and named from Phase 02-04 results among `p10-a`, `p10-b`, `p10-c`, `p14`, `g-l`, `rrf3`, `f3`, `rrf4`), best so far of any class; every metric of record with labels; class check (laptop 2 s, GPU 0.1 s, GLiNER offline 2 GPU-h per million units, derived); cost rows offline and online, seconds and USD, per component and hardware, inherited ones source-labelled; hop-only unit count and gold-through-hop-only questions (exploratory); `data/phase05/results.json` and `docs/plans/fase-05-hop/results.md` | C5 | `src/edge_rag/phase_results.py`, `src/edge_rag/cli.py` | C | done | commit bb645a8; `uv run edge-rag hop-results` at bb645a8 with `git_src_changes` empty, then `--from-json` prints byte-equal, sha256 `3d851d0e...`; an independent verifier (no `phase_results` import) recomputed every FS@2,048 count and the exploratory counts (gold through `hop-ms` only: 146 / 8 / 136 questions on HotpotQA / MultiHop-RAG / MuSiQue) |
| 7 | McNemar on FS@2,048 for the seven comparisons per set (`mch` vs G-L, `rrf-prf`, `rrf-1s`, best light, best any; `rrf-prf` vs G-L, best any) with wins, losses, ties, p, state; verdict, `rrf-prf` context verdict and entrant by code; `tests/test_hop_results.py` pinning each bar and the `not run` case (HotpotQA not run, `mch` wins MuSiQue, ties MultiHop-RAG: `does not advance`) | C6 | `src/edge_rag/phase_results.py`, `tests/test_hop_results.py` | C | done | `tests/test_hop_results.py` pins each bar and the plan's `not run` case; the verifier's own exact McNemar matches every wins / losses / ties and p; verdict `does not advance`, `rrf-prf` context verdict `does not advance`, entrant `none from this phase` |
| 8 | Close checks: closing `clientBalance` and `myself { pods }` with UTC time; `npm run check` | C7, C8 | `plan.md` | main | done | 2026-10-05T06:57:23Z `clientBalance` 14.1147504042 USD, `pods` [] (measured); spend 0 USD (measured, delta of two real readings); `npm run check` exit 0 at bb645a8 |
| 9 | Review and close: adversarial review (`.agents/skills/sdd-review/SKILL.md`), fixes, results by criterion in this plan, master plan status and decision row if the verdict needs one, delivery on the branch with a PR (`.agents/skills/sdd-delivery/SKILL.md`) | all | `plan.md`, `docs/plans/0_plan_maestro.md` | main | done | close review round 1 (four lenses) and round 2 (fix diff) below; fixes in 0c827e6; results by criterion below; master plan row and decision row; branch pushed with a PR to `main` |

How each increment is checked (from the worktree root, `OLD_DATA_ROOT` exported, D2):
- 0: `git check-ignore -v data` prints `.gitignore:12:/data/`; `ls data/phase04/rankings/hotpotqa-dev/pool.manifest.json` succeeds; readings written under Money.
- 1: `uv run pytest -q tests/test_fusion_results.py tests/test_judge_results.py` with `git diff main -- tests/` showing import lines only; `uv run edge-rag fusion-results --from-json` prints sha256 `e50c85a9ca1e201dd30cbc50bde426c5fddeab638e2c4f53e3c04d3227e0a9a9`; `uv run edge-rag judge-results --from-json` prints `6b4b6bd19cb937b962cc1245b10ed793eb5d649a0ce67cf7a273391db972cc0c`; `uv run edge-rag fusion-results --out .runtime/c3/phase03` and `uv run edge-rag judge-results --out .runtime/c3/phase04`, then `sha256sum .runtime/c3/phase0*/results.md` equal to the two hashes above, and per phase `uv run python -c "import json,sys; a,b=(json.load(open(p,encoding='utf-8')) for p in sys.argv[1:]); [d.pop(k) for d in (a,b) for k in ('git_commit','git_src_changes')]; print('equal' if a==b else 'DIFFER')" data/phase03/results.json .runtime/c3/phase03/results.json` (and `phase04`) prints `equal`; `sha256sum data/phase03/results.json data/phase04/results.json` still `ccf4dfc9...` and `93fa9140...` (stored files untouched).
- 2: `uv run pytest -q tests/test_judge_results.py`; increment 1's Phase 04 recompute and comparison rerun.
- 3: `uv run pytest -q tests/test_converge.py tests/test_data_and_hops.py`.
- 4: `uv run edge-rag converge --set multihop-rag` prints the three equality counts 2,255 of 2,255 (MuSiQue 2,417 of 2,417), seed-count mean and range (spec, unit ids only: MultiHop-RAG 8.97, MuSiQue 8.79, range 5-10), empty and short `hop-ms` counts, peak RSS; the trace written under Hand traces.
- 5: as 4 on `hotpotqa-dev` (7,405 of 7,405), log ends with `exit 0`.
- 6: `uv run edge-rag hop-results` then `uv run edge-rag hop-results --from-json` prints that `results.md` regenerates byte-equal.
- 7: `uv run pytest -q tests/test_hop_results.py`; the fields in `data/phase05/results.json`.
- 8: readings under Money; `npm run check` exit 0.

## Laptop run (increment 5)
Run by the main session with the Bash tool's `run_in_background`, never inside a subagent (subagent calls stop near 1,000 s), with nothing else running.
MultiHop-RAG and MuSiQue run the same command inside group B, in the foreground, with logs `data/phase05/logs/converge-<set>.log`.

```
cd "C:/Python Projects/concept-embeddings-RAG-Edge/.claude/worktrees/fase-05-hop"
git status --porcelain -- src   # must print nothing: the manifest records git_src_changes
mkdir -p data/phase05/logs
export OLD_DATA_ROOT="C:/Python Projects/concept-embeddings-RAG/data" PYTHONUNBUFFERED=1
{ time uv run edge-rag converge --set hotpotqa-dev; echo "exit $?"; } > data/phase05/logs/converge-hotpotqa-dev.log 2>&1
```

Watch it with `tail -3 data/phase05/logs/converge-hotpotqa-dev.log` (a line every 500 questions) or a Monitor `until grep -q '^exit ' ...` loop.
Expected (projection, not measured): loading about 397 s (Phase 04 `pool` stages on HotpotQA, measured there: sha256 37.1, corpus 42.4, BM25 rebuild 216.6, vectors 74.2, entity index 21.7, files 4.8), the per-question stage about 1.2 h (spec), so about 1.3 h in all; peak RSS near Phase 04's 13,051.1 MB.
After it, check: the log ends with `exit 0`; `data/phase05/rankings/hotpotqa-dev/converge.manifest.json` has the three equality counts 7,405 of 7,405, `git_src_changes` empty, the commit of session boundary A, the GLiNER configuration digest `2f7864661b8ce7ff` and the input sha256 of the spec, constants 5 / 60 / 100, seed-count mean 8.59 with range 5-10, empty and short `hop-ms` counts, online seconds per question per component and peak RSS; the five ranking files exist with the sha256 the manifest records (`sha256sum`).
If the laptop cannot hold it (memory error or a stall), HotpotQA is `not run` and a deviation is opened (spec A1), never moved to a pod.

## Hand traces (C1)
One question per set, computed by a separate snippet from the stored files, not by `converge.py`.

### MultiHop-RAG
- multihop-rag, question `mhr-0000` (the set's first question), measured with a separate snippet that does not import `edge_rag.converge` or `Hops.seed_hop`: it rebuilds each seed's hop from the incidence matrix and `node_weights`, and computes RRF with exact fractions.
- Seeds, 9 in order: Dense (`p10-a`) top 5 `26585272f0b17822`, `49e91e49ea3c94ea`, `acc64fb778603c17`, `401d139722cc3dc9`, `ac3588c17b9c6beb`, then BM25's `1370c5372d049f31`, `745f28b71648586d`, `6800b4e1c14dc607`, `4bb5b2c6778915e0` (BM25's rank-4 `ac3588c17b9c6beb` was already a seed, so it was dropped as a duplicate).
- Every seed has 2 to 5 entity nodes, and every per-seed list is full at 100 units.
- The cosine of `q'` with `q` is 0.9327 (`q` has unit norm).
- The top `hop-ms` unit is `00be98c484904588`, which all 9 seeds reach, at per-seed ranks 19, 1, 52, 21, 3, 28, 3, 3, 3 in seed order; its RRF sum is 60199547/480897648 = 0.12518, ahead of `020cdee06f6142fb` at 0.12303.
- The recomputed `hop-ms` list (100 units) is identical, in order, to the stored `data/phase05/rankings/multihop-rag/hop-ms.jsonl.gz` entry for `mhr-0000`, and the recomputed `dense-prf` list is identical to the stored one (its top unit is the seed `acc64fb778603c17`).

### MuSiQue
- MuSiQue, question `2hop__460946_294723` ("Who is the spouse of the Green performer?"), the first question of the set and the first line of the stored `hop-ms` file (measured).
- Seeds in order are Dense (`p10-a`) top 5 `adc6fd8efa68d39f`, `d6f7054b295788c6`, `4566c5b8f905ddc1`, `711e267af7b6d7ea`, `1f57f15c66100eee`, then BM25 top 5 `4fb6ff21c7274493`, `690c1792a72ff2fb`, `6dc309a79b6d0f80`, `deb2e19528b1e5e2`, `ab80d69e2d9bb5e0`, with no overlap, so 10 seeds (measured).
- Per-seed hop lists have lengths 100, 100, 100, 100, 100, 4, 100, 61, 100 and 5, and none is empty (measured).
- The cosine of q' = (q + c) / ||q + c|| with q is 0.95635 (measured).
- The top `hop-ms` unit is `0dc20226fbcb414a`, reached from seeds `adc6fd8efa68d39f` at rank 25 and `d6f7054b295788c6` at rank 17, with an RRF sum of 1/85 + 1/77 = 162/6545 = 0.024752 (measured).
- The runner-up `05f668d6a88a91de` scores 0.022285, so the top unit comes from two Dense seeds converging, not from one high rank (measured).
- An independent recomputation that does not import `edge_rag.converge` or call `seed_hop` reproduces the stored `hop-ms` list for this question exactly, all 100 units in order, and also the stored `dense-prf` list (measured).

### HotpotQA
- HotpotQA dev, question `5a8b57f25542995d1e6f1371` ("Were Scott Derrickson and Ed Wood of the same nationality?"), is the first question of the set and the first line of the stored `hop-ms` file (measured).
- Seeds in order are Dense (`p10-a`) top 5 `45558f75b4cfebcb`, `0683169f72079434`, `336febac950c15ce`, `087b940832bef4d1`, `4a0614849815b50d`, then BM25's `eec1b82cf56cda2c`, `59efff430e232c3e`, `0262ee7b66bb3aa6`, `f205c20414040c65`; BM25's rank-1 `45558f75b4cfebcb` was already a seed, so there are 9 seeds (measured).
- Per-seed hop lists have lengths 100, 100, 100, 100, 100, 100, 48, 48 and 53, and none is empty (measured).
- The cosine of q' = (q + c) / ||q + c|| with q is 0.96430, where q has unit norm (measured).
- The top `hop-ms` unit is `142f0a1d6fef82d1`, reached from 5 of the 9 seeds: `0683169f72079434` at rank 1, `087b940832bef4d1` at rank 69, `59efff430e232c3e` at rank 3, `0262ee7b66bb3aa6` at rank 3 and `f205c20414040c65` at rank 8, with an RRF sum of 793297/11236932 = 0.070597 (measured).
- The runner-up `0512b21eca235ac1` is reached from the same 5 seeds and scores 0.070305, so the top unit wins by a small margin of agreement across seeds rather than by a single high rank (measured).
- An independent recomputation that does not import `edge_rag.converge` or call `seed_hop` reproduces the stored `hop-ms` list for this question exactly, all 100 units in order, and also the stored `dense-prf` list, whose top unit is the seed `0683169f72079434` (measured).

## Money
Phase start reading (increment 0), closing reading (increment 8), both with UTC time; spend 0 USD; money left inside the 25 USD authorization stays 9.89 USD (derived, provisional as in the master plan).
- Start: 2026-10-04T14:09:46Z, 14.1147504042 USD, no pods (measured).
- Close: 2026-10-05T06:57:23Z, 14.1147504042 USD, no pods (measured).
- Phase spend 0 USD (measured); money left inside the authorization 9.89 USD (derived, provisional).

## Session boundary
Phase closed (2026-10-05): increments 0-9 are done and the branch is delivered with a PR; nothing is left to resume in this phase.

## Decisions taken during execution
- E1. Work moved from the worktree to the main checkout on branch `fase-05-hop` (2026-10-05), so `data/` is the real folder and the D2 junction is not used from here on; `OLD_DATA_ROOT` is still exported.
  Reason: the group B subagent was stopped and its commit pushed to `fase-05-hop`; the worktree branch holds nothing more.
- E2. The first MultiHop-RAG and MuSiQue runs (commit 5f5dcb3, no logs) were moved out of `data/` and rerun with timed logs at 5f313b6; the five output sha256 per set are identical, so the stored outputs are the reruns.
- E3. Registered, not fixed: `Dense.retrieve` (`retrieval/dense_bm25.py`, before this phase) does not break an exact score tie straddling rank 100 by unit id, because `np.argpartition` picks an arbitrary member; it changes `dense-prf` on 1 MultiHop-RAG and 3 MuSiQue questions.
  The spec fixes `dense-prf` as the unchanged `Dense.retrieve`, so the code stays; the effect is recorded here.
- E4. The results page is `docs/plans/fase-05-hop/results.md`, as the spec correction 24ef7e8 fixes; D1's sentence placing `results.md` in `data/phase05/` is superseded on that point.
- E5. Two cost rows corrected before the stored write (minor findings of the group C verifier): the MultiHop-RAG `Dense corpus embeddings` row carries its inherited 41.20 s (spec cost accounting), and the `rrf-1s` single-seed hop label names the owned laptop like every other laptop row.
  The first `results.json` (written with uncommitted code) was moved out of `data/` and rewritten at bb645a8 with a clean `src`.
- E6. Close review round 1 fixes (0c827e6): `hop-results` reads the Phase 02-04 `results.json` only when its sha256 is the pinned one (Phase 03 and 04 from the spec; Phase 02 `287718c3...`, measured at this close, as the spec pins none); a results command run without `--out` refuses to overwrite a stored `results.json`; `hop-results` refuses a set whose converge equality counts are not complete.
  After the fixes, `hop-results --out .runtime/r1` gives a `results.json` equal to the stored one except the git fields and the page byte-equal, sha256 `3d851d0e...` (measured).
- E7. The D2 junction `data` inside `.claude/worktrees/fase-05-hop` was removed with `os.rmdir` (junction only; `data/` in the main checkout intact), so a later removal of that worktree cannot follow it.

## Deviations
| ID | Summary | Affects criteria | Status |
|---|---|---|---|
| 05.a | Spec correction 24ef7e8 after approval: C5's page path and the loading time in the runtime projection; no bar, metric or set changed | C5 (path only) | recorded at the close for the author; no deviation file was opened at the time |
| 05.b | `dense-prf` ties at rank 100 are not broken by unit id on 1 MultiHop-RAG and 3 MuSiQue questions (E3), against the frozen 'ties by unit id' | C1 (definition of `dense-prf`) | recorded, not fixed: the spec also fixes `dense-prf` as the unchanged `Dense.retrieve`; for the author |

## Adversarial review
| Round | Backend | Range | Lenses | Findings | Status |
|---|---|---|---|---|---|
| 0 | subagents | 5f5dcb3 (increments 3-4) | spec fidelity, correctness, tests and stored outputs | no behaviour defect; 1 major (hand traces missing) fixed in 260c0f7; minors: `hop.py` docstring fixed in 5f313b6, `Dense.retrieve` tie recorded as E3 / 05.b, the rest noted | closed |
| 1 | subagents | main...19395b6 | correctness, security and data, evidence and tests, scope and maintenance | 2 important confirmed (earlier results read without their pinned sha256; live `data` junction in the old worktree), 1 plausible (a bare results run overwrites stored files, pre-existing on `main`), 11 minor; fixed in 0c827e6 and E7; doc minors fixed in this plan; the remaining minors are listed under Results by criterion | closed |
| 2 | subagent | 19395b6..0c827e6 (fix diff) | correctness, evidence and tests | 2 important (`--out` onto the stored folder still overwrote; the write-once check ignored the page), 3 minor (equality check accepted a missing or short check; no test of it; no test of the `--out` and page-only cases); all fixed in ff223d0 with tests; the three pages regenerate byte-equal and the three recomputes give `results.json` equal except the git fields | closed; fix diff checked by a fresh reviewer (row 3) |
| 3 | subagent | 0c827e6..ff223d0 (fix diff) | correctness, evidence and tests | no blocker, no important; bare runs and `--out` onto a folder holding either stored file are refused (also with other letter case); the new tests fail for the right reason; stored sha256 unchanged; pages byte-equal; 3 minor, not fixed (last round allowed): the write-once check is not atomic against two parallel runs; Phase 02 `edge-rag results` still writes `data/phase02/results.json` without the guard (outside this diff); `converge_complete` raises `KeyError` instead of `ArtifactError` on a manifest missing keys | closed; minors passed to the author |

## Results by criterion
- C1 met: `converge` wrote the five depth-100 rankings with manifests on the three sets, once; equality 7,405 + 2,417 + 2,255 = 12,077 of 12,077 on each of the three checks (per-set counts measured, total derived); seed mean 8.59 / 8.97 / 8.79, range 5-10; hand traces for one question per set from separate snippets (increments 4 and 5).
  Recorded against the frozen text: deviation 05.b (`dense-prf` ties at rank 100 on 4 questions).
- C2 met: `tests/test_converge.py` pins every listed example; full `uv run pytest -q` passes at ff223d0 (increment 3, rerun at the close).
- C3 met: one module `phase_results.py`; both old modules deleted; Phase 03 and 04 tests pass with import-only changes; `--from-json` byte-equal `e50c85a9...` and `6b4b6bd1...`; full recompute into `.runtime/c3/` equal except the git fields, pages byte-equal; stored JSON sha256 unchanged (increments 1 and 2, independent verifier).
- C4 met: row `J-strong over {units} units`, seconds per 100 x N / 100, label `x N`; the 80-unit test passes; the Phase 04 recompute gives the stored page and figures (increment 2).
- C5 met: `data/phase05/results.json` and `results.md` written at bb645a8 with a clean `src` by the shared module, every listed system, metric, label, class check and cost column; exploratory counts (gold through `hop-ms` only 146 / 8 / 136); `--from-json` byte-equal `3d851d0e...`; recompute after the review fixes equal (increments 6, E5, E6).
  Open minor: `hop-ms` and `dense-prf` appear with metrics but without cost rows (they are components inside the fused systems' cost rows).
- C6 met: the seven exact McNemar tests per set with wins, losses, ties, p and state, recomputed by an independent verifier; verdict, context verdict and entrant written by code; `tests/test_hop_results.py` pins each bar and the `not run` case (increment 7).
- C7 met: no pod, no paid API; `clientBalance` 14.1147504042 USD at 2026-10-04T14:09:46Z and at 2026-10-05T06:57:23Z, `pods` [] both times; spend 0 USD (measured); money left inside the authorization 9.89 USD (derived, provisional).
- C8 met: `npm run check` exit 0 at ff223d0.

Outcome written by code (`results.md`): `mch` **does not advance**; exam entrant: none from this phase; `rrf-prf` (context): does not advance.
Failed bars (measured, FS@2,048 paired tests): against G-L, HotpotQA loss 366 / 838 (p 4.7e-43), MuSiQue tie 135 / 140; against `rrf-prf`, MultiHop-RAG loss 27 / 121; against the best light-class system so far, a loss on every set (`rrf4` HotpotQA 90 / 853, `p10-b` MultiHop-RAG 31 / 252, `p14` MuSiQue 80 / 313).
`mch` also loses to `rrf-1s` on every set, so the multi-seed hop with a refined query is behind the single-seed hop it was meant to improve (measured).
Reading (interpretation, not tested): `hop-ms` alone finds almost no full gold set (FS@2,048 10 of 7,405 on HotpotQA), because the seeds, often gold themselves, are excluded from it; its 96k HotpotQA units that reach `mch`'s top 100 only through it displace more useful units than they add.

Open for the author (minor, not fixed): duplicated per-hardware totals code across the three phase descriptions; GLiNER and Dense inherited figures held twice (label text and numeric table); `converge.read_single_hop` duplicates `judge.pinned_rankings`; a rerun of `converge` replaces ranking files while keeping the old manifest; no unit test of `Phase05.system_cost`; the three round 3 minors (write-once not atomic against parallel runs, Phase 02 `results` without the write-once guard, `KeyError` instead of `ArtifactError` in `converge_complete`).

## Candidate learnings
Only reusable lessons with a textual quote from the session; consolidated at the close.
None proposed: no lesson of this phase has evidence from two sessions.
