# G-A1 pod recipe (Phase 06, plan increment 5)

Search-R1 on the HotpotQA dev 1,000: what the pod runs, on which GPU, at what projected cost, and how the result comes back.
Prepared on the laptop on 2026-10-06 (branch `ga1-prep`); no pod was rented and no money was spent preparing it.

## Pieces
- `src/edge_rag/ga1_hotpot.py` (laptop): `export` writes the upload bundle; `rank` checks the pod output and writes `data/phase06/rankings/hotpotqa-dev/g-a1.jsonl.gz` with `g-a1.manifest.json`, in the G-R2 format.
- `src/edge_rag/pod/ga1_subset.py` (pod, `pod` group): runs the Phase 02 driver `edge_rag.pod.searchr1` on the 1,000 qids (below).
- `src/edge_rag/pod/ga1_pace.py` (pod, standard library, run as a script): the pace gate.
- `scripts/pod_ga1.sh` (pod): upload checks, C3 diff, environments, CUDA check, G-L rebuild, retrieval server and driver, `sha256sums.txt`.
- `tests/test_ga1_hotpot.py`: qid list refusals, the restriction, the wrapper's call of the driver, the pace gate, evidence checks, the conversion and its refusals, and the export's refusal of qids outside the question file (10 pass).

## What runs (spec C3, plan D3)
- Driver: `src/edge_rag/pod/searchr1.py`, byte-equal to Phase 02 commit `33c0668` (session 4, where G-A1 ran on MuSiQue and MultiHop-RAG); checkpoint, prompt, greedy decoding, top-3, at most 8 searches, 1,024 new tokens per turn, all as there.
- Evidence lists: written by the driver itself (`dedup_in_order`: retrieved units in retrieval order, later duplicates dropped, at most 24), as in Phase 02; `rank` only checks them and copies them into the Phase 06 format.
- Index: `src/edge_rag/pod/colbert.py` and `colbert_env` byte-equal to commit `f3f499e` (session 6, D17), run with `pod_run.sh`'s HotpotQA arguments: one `create` call (`--chunk-units 6000000`), float16 embeddings encoded 200,000 units at a time.
- The script writes `git diff 33c0668 -- searchr1.py` plus `git diff f3f499e -- colbert.py colbert_env` to `c3-g-a1.diff` and stops if it is not empty; `rank` refuses a non-empty diff.
  Checked on the laptop at 1943b2a: both diffs are empty; `common.py` gained only `git_provenance` since `f3f499e` (no line the driver calls changed).
- Restriction to the 1,000 qids: the driver cannot take a qid list, and a smaller question file cannot pass its loader (`common.set_questions` checks the file's digests and its count of 7,405).
  So `ga1_subset` replaces that one function for the run with one that calls the original (every check still runs) and keeps the listed qids in question-file order, then calls the driver's own `main` with `--set hotpotqa-dev`.
  It records the qid list's sha256, the driver files' sha256 and the commit in `subset.json`.
- The build's own search is cut to the first 100 questions (`--limit-questions 100`, an existing option; `GL_SEARCH_QUESTIONS=` for all 7,405, about 0.74 h more at F9's 0.358 s/question): the spec's cost row counts encode and index only, and G-A1 searches through the server.

## The GPU memory split (open: author decision)
- Read from source (PyLate 1.6.0 and fast-plaid 1.4.6.2110 wheels, 2026-10-06): PyLate's `PLAID` passes `low_memory=False` by default, so the retrieval server loads the whole index onto the GPU when it starts.
- The HotpotQA index is 35 GB on disk (F9); the MuSiQue and MultiHop-RAG indexes beside which the driver's `GPU_MEMORY_UTILIZATION` 0.80 was set (D15) were small.
  On an 80 GB card, 0.80 asks vLLM for 64 GB while the server holds about 35-40 GB (projection), so vLLM would refuse to start.
- The script therefore measures the server's GPU memory once it answers, and gives vLLM `min(0.80, (total - used - 6 GiB) / total)`, refusing below 0.30; with a 37 GB server that is about 0.46 (projection): 15 GB of bf16 weights and about 20 GB of KV cache.
  `ga1_subset` sets it through the driver's module constant, so the file is unchanged, and records both values; the conversion copies it into the manifest settings.
- This is a memory setting, not a ghost setting (D15 and F6 of Phase 02 treat it as a run setting), but it changes vLLM's batch composition, which may move greedy outputs slightly; it is the author's call before the pod starts.
  Alternatives: a 2x A100 80 GB pod with the server on the second card (no setting changes; about 3.18 USD/h, projected 3.3 h is about 10.3 USD, past the 6.9 USD cut); or the server on the CPU (Phase 02 F6: too slow on MultiHop-RAG; not tried on a 5.2 M-unit index).

## Pod
- 1x NVIDIA A100-SXM4-80GB, Secure Cloud, on-demand, `minMemoryInGb` 250 (largest single-GPU host offer in Phase 02 D17; session 6 peak cgroup memory 150.5 GB), CUDA 13 in `allowedCudaVersions` (vLLM's CUDA 13 build, Phase 02 F4).
- Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, ports `22/tcp,8888/http`, SSH key `~/.ssh/runpod_ed25519`, as in Phase 02.
- Volume 150 GB at `/workspace`, `HF_HOME=/workspace/hf`: index 35 GB, two environments about 23 GB, Search-R1 checkpoint about 30 GB, old data 0.7 GB (Phase 02 F6 and F9).
- Rate: 1.59 USD/h (Phase 02 D15 and D17, measured then); read `securePrice` again and record the `costPerHr` returned at creation.

## Projected time and cost (projection, not measurement)
| Step | Hours | Basis |
|---|---:|---|
| Setup and uploads | 0.4 | spec |
| G-L rebuild | 1.3 | F9 measured: encode 3,877 s, index 685 s |
| Index digests, 100-question search | 0.05 | assumed, 35 GB read once |
| Server start | 0.25 | assumed: 5.2 M units read, 35 GB index to the GPU |
| G-A1, 1,000 questions | 0.5 | spec: about 3 searches each at 0.358 s plus generation |
| Collect and download | 0.1 | spec; outputs under 20 MB |
| Total + 25 % | 3.3 | spec row 2.9 h without the two added steps |

- At 1.59 USD/h: about 5.2 USD (spec row: 4.6 USD); hard cut 6.9 USD, about 4.3 h of pod time including the minutes before the script starts (`SPENT_S`).
- Gate (`ga1_pace.py`, every 60 s): stop when the spend reaches the cut; during the encode, once 10 minutes of pace are sampled, stop when the measured pace plus the rest of the table passes the cut; during G-A1, stop when the mean turn time times the turns left (at most 9) passes it.
  A stop writes `STOP_CUT` in the log, ends the stage's processes and the script; the outputs so far are summed and downloaded, and G-A1 is recorded as stopped.
- Phase cap: G-A1 starts only if the phase spend so far plus this projection at the offered rate stays within the cap (spec money rule).

## Uploads
- The repository at the frozen commit (cloned on the pod; the commit pushed first).
- `data/phase06/ga1/` from the main checkout, to `data/phase06/ga1/` in the clone (export of 2026-10-06, `uv run python -m edge_rag.ga1_hotpot export`):
  - `hotpotqa-dev-1000.txt` sha256 6cebd41ff9e5e9f02e27ebb019620de0e37b77794fc6a783a2ff0d80fb3a62bd (25,000 bytes, 1,000 qids, the C2 file);
  - `upload.sha256`, `old.sha256` (checked by the script with `sha256sum -c`), `upload.manifest.json` (sizes, digests, commit; its sha256 goes to `plan.md` with the run).
- Old data, read in place from the old repository and uploaded unchanged to `/workspace/old-data/phase9/` (`OLD_DATA_ROOT=/workspace/old-data`), as in Phase 02 D16:
  - `corpus.json` d6987a19baccae0a11d18dd0f2481a5d34cd3fde3465e7c66be54b90c3b85553 (1,808 bytes);
  - `corpus.jsonl.gz` d982847985bcc5d9b26a5e2ce379e2e04d59fb9ead6b81e21ab0f1c168e93b0b (675,508,508 bytes);
  - `questions.json` f3f8b4f0df32be5578588aeb079656c1a1fbe943ef6709f5d3420ae62b09ef0f (5,268,564 bytes).

## Commands
1. Create the pod; record `costPerHr` and the creation time.
2. On the pod: clone, `git checkout <frozen commit>`, install `uv`; `scp` the uploads above.
3. `POD_COST_PER_HR=<costPerHr> SPENT_S=<seconds since creation> OLD_DATA_ROOT=/workspace/old-data nohup bash scripts/pod_ga1.sh > /workspace/pod_ga1.out 2>&1 &`
4. Watch `/workspace/pod_ga1.log` for `STAGE_DONE`, `STAGE_FAIL`, `PACE` and `STOP_CUT` lines, `/workspace/pod_ga1.samples` for GPU, host memory and disk.
5. Download `data/phase06/ga1/pod/` (flat; `sha256sums.txt` lists every file) to the main checkout's `data/phase06/ga1/pod/`, `sha256sum -c sha256sums.txt` there, then `podTerminate` and check `myself { pods }` is empty.
6. Laptop: `uv run python -m edge_rag.ga1_hotpot rank` (checks every file against `sha256sums.txt`, the empty C3 diff, the qid list, the D17 settings and the corpus units) writes the ranking and manifest.

## Downloads (`data/phase06/ga1/pod/`)
- `g-a1.jsonl.gz` and `g-a1.manifest.json` (driver), `trace.jsonl.gz` (each question's searches and answer), `subset.json`, `gpu_split.txt`.
- `g-l.manifest.json` and `g-l.jsonl.gz` (the rebuild, its 100-question search), `index.sha256` and `index.du` (the rebuilt index's file digests and size: the index itself is not downloaded).
- `c3-g-a1.diff`, `c3-g-a1.status`, `commit.txt`, logs (`pod_ga1.log`, `.progress`, `.samples`, `serve.out`), `sha256sums.txt`.

## Failure modes
- Host RAM: the D17 `create` peaked at 150.5 GB of cgroup memory (F9) on a 250 GB host; a host with less memory fails the build (Phase 02 F8 history); the sampler logs `memory.current` and `memory.peak`.
- GPU memory: the split above; the script refuses to start vLLM below 0.30, and a vLLM out-of-memory error ends the stage (spec: one resize to the next tier only if the re-projection fits the cut).
- Disk: about 90 GB of 150 GB used; the sampler logs `/workspace` use (Phase 02 F6: a full 100 GB volume killed a stage).
- Driver CUDA: vLLM 0.30's CUDA 13 build needs a CUDA 13 driver (F4); FlashInfer's sampler build is avoided as in D13.
- Server start: reading 5.2 M units and loading 35 GB to the GPU is unmeasured; the script waits for `/` to answer and fails if the server process dies.
- A stage that fails twice on the same problem is recorded failed with its log (Phase 02 rule); G-A1 is aborted without retry under other settings if C3 fails.

## For the integrator (not done here: `cli.py`, `plan.md` and `phase_results.py` are edited on `fase-06-rivals`)
- `cli.py`: two commands mapping to `ga1_hotpot.run_export(out)` and `ga1_hotpot.run_rank(bundle, pod, rankings)`, defaults `data/phase06/ga1`, `data/phase06/ga1/pod`, `config.PHASE06_RANKINGS_DIR` (or keep `python -m edge_rag.ga1_hotpot`).
- Increment 6 results: read `rankings/hotpotqa-dev/g-a1.jsonl.gz` (1,000 rows, evidence lists of at most 24 units, as Phase 02 G-A1); FS@2,048 and every McNemar test of G-A1 on HotpotQA on those 1,000 qids only (spec C6), pairing the other systems' rows restricted to the same qids; offline cost from `offline_seconds` (the rebuild's encode and index) and online from `online_seconds`, at `pod.cost_per_hr_usd`.
- `plan.md` row 5: the frozen commit, the export manifest's sha256, the author's choice on the GPU memory split, and the pod record.
