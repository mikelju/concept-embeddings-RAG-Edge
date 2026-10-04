# Phase 05 - Multi-seed convergent hop with a refined query: plan

Status: in progress
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
| 0 | Phase start: create the `data` junction and check it (D2); read `clientBalance` and `myself { pods }` with UTC time and record them under Money | C7 | `plan.md` | main | pending | |
| 1 | Shared results module (D3, D4): `phase_results.py` with the Phase 03 and Phase 04 descriptions; delete `fusion_results.py` and `judge_results.py`; CLI `fusion-results` and `judge-results` on the new module with `--from-json` and `--out`; tests move imports only. `--from-json` regenerates both stored pages byte-equal; full recompute of each phase into `.runtime/c3/`, JSON equal to the stored one except `git_commit` and `git_src_changes`, page byte-equal to the stored page | C3 | `src/edge_rag/phase_results.py`, `src/edge_rag/cli.py`, `tests/test_fusion_results.py`, `tests/test_judge_results.py` | A | pending | |
| 2 | Finding 4 fix: the judge's online row is `J-strong over N units`, N = `judged_units` of the judged lists, seconds = per-100 seconds x N / 100, label saying `x N`; test with an 80-unit judged list (name and 0.8 x seconds); increment 1's recompute rerun after the fix gives the stored Phase 04 page and figures | C4, C3 | `src/edge_rag/phase_results.py`, `tests/test_judge_results.py` | A | pending | |
| 3 | Retrieval code and tests: `config` constants (D1); `Hops.seed_hop` (D5); `converge.py` pure functions; `tests/test_converge.py` pinning seed order and deduplication, seed exclusion, `seed_hop` equal to `entity_hop` with Dense's first unit and top 10 excluded, the convergence example (two seeds at rank 50 above one seed at rank 1), an empty per-seed list adding nothing, `q'` on a two-dimensional example, the four-list RRF with an empty `hop-ms` | C2 | `src/edge_rag/config.py`, `src/edge_rag/retrieval/hop.py`, `src/edge_rag/converge.py`, `tests/test_converge.py` | B | pending | |
| 4 | `converge.run` and `edge-rag converge --set <set>` (D6-D9, D11): `hop-ms`, `dense-prf`, `mch`, `rrf-prf`, `rrf-1s` depth-100 rankings into `data/phase05/rankings/<set>/`, write-once, `converge.manifest.json` with every C1 field and the three equality counts; run on MultiHop-RAG then MuSiQue with timed logs; one question per set traced by hand (seeds, cosine of `q'` with `q`, top `hop-ms` unit and its seeds with ranks and RRF sum), computed by a separate snippet from the stored files, not by `converge.py` | C1 | `src/edge_rag/converge.py`, `src/edge_rag/cli.py`, `plan.md` (trace) | B | pending | |
| | **Session boundary A**: code for all three sets is committed and the two small sets are run; the HotpotQA run starts from a clean tree. | | | | | |
| 5 | HotpotQA run on the laptop, launched by the main session in the background (see Laptop run); after it, check the manifest and record the HotpotQA hand trace | C1 | `plan.md` | main (trace: C) | pending | |
| | **Session boundary B**: every Phase 05 ranking file and manifest is on disk; the results session starts from this table. | | | | | |
| 6 | Phase 05 description in `phase_results.py` and `edge-rag hop-results [--from-json] [--out DIR]`: per set `mch`, `rrf-prf`, `rrf-1s`, `hop-ms`, `dense-prf`, G-L, best light-class so far (read and named from Phase 02-04 results among `p10-a`, `p10-b`, `p10-c`, `p14`, `g-l`, `rrf3`, `f3`, `rrf4`), best so far of any class; every metric of record with labels; class check (laptop 2 s, GPU 0.1 s, GLiNER offline 2 GPU-h per million units, derived); cost rows offline and online, seconds and USD, per component and hardware, inherited ones source-labelled; hop-only unit count and gold-through-hop-only questions (exploratory); `data/phase05/results.json` and `docs/plans/fase-05-hop/results.md` | C5 | `src/edge_rag/phase_results.py`, `src/edge_rag/cli.py` | C | pending | |
| 7 | McNemar on FS@2,048 for the seven comparisons per set (`mch` vs G-L, `rrf-prf`, `rrf-1s`, best light, best any; `rrf-prf` vs G-L, best any) with wins, losses, ties, p, state; verdict, `rrf-prf` context verdict and entrant by code; `tests/test_hop_results.py` pinning each bar and the `not run` case (HotpotQA not run, `mch` wins MuSiQue, ties MultiHop-RAG: `does not advance`) | C6 | `src/edge_rag/phase_results.py`, `tests/test_hop_results.py` | C | pending | |
| 8 | Close checks: closing `clientBalance` and `myself { pods }` with UTC time; `npm run check` | C7, C8 | `plan.md` | main | pending | |
| 9 | Review and close: adversarial review (`.agents/skills/sdd-review/SKILL.md`), fixes, results by criterion in this plan, master plan status and decision row if the verdict needs one, delivery on the branch with a PR (`.agents/skills/sdd-delivery/SKILL.md`) | all | `plan.md`, `docs/plans/0_plan_maestro.md` | main | pending | |

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
One question per set, written by the agent that ran the set.

## Money
Phase start reading (increment 0), closing reading (increment 8), both with UTC time; spend 0 USD; money left inside the 25 USD authorization stays 9.89 USD (derived, provisional as in the master plan).

## Session boundary
Empty state: nothing done yet.
The spec is approved and committed (bc3fb9e); no code, run or reading of this phase exists; the next session starts at increment 0.

## Decisions taken during execution
None yet.

## Deviations
| ID | Summary | Affects criteria | Status |
|---|---|---|---|

## Adversarial review
| Round | Backend | Range | Lenses | Findings | Status |
|---|---|---|---|---|---|

## Results by criterion
Written at the close.

## Candidate learnings
Only reusable lessons with a textual quote from the session; consolidated at the close.
