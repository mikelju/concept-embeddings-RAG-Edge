# G-A2 pod recipe (Phase 06, plan increment 4)

HippoRAG 2 on MultiHop-RAG: what the pod runs, on which GPU, at what projected cost, and how the result comes back.
Prepared on the laptop on 2026-10-06 (branch `ga2-prep`); no pod was rented and no money was spent preparing it.

## Pieces
- `src/edge_rag/ga2.py` (laptop): `export` writes the upload bundle; `rank` turns the pod output into the depth-100 ranking `data/phase06/rankings/multihop-rag/g-a2.jsonl.gz` with `g-a2.manifest.json`, in the G-R2 format.
- `src/edge_rag/pod/hipporag_run.py` (pod, HippoRAG's own venv, stdlib plus `hipporag` only, run as a script): the adapter.
- `scripts/pod_ga2.sh` (pod): clone at the pinned commit, venv, C3 diff, models, vLLM, probe, probe gate, full run, `sha256sums.txt`.
- `tests/test_ga2.py`: export manifest and sums, pod bundle check, passage groups, token split, ranking conversion and its refusals.

## What HippoRAG runs (spec C3)
- Code: `github.com/OSU-NLP-Group/HippoRAG` at `2bfd831417202b49cda9da7973e141a456e16872` (setup.py 2.0.0a5), installed unchanged with its `vllm` and `gritlm` extras (torch 2.5.1, transformers 4.45.2, vllm 0.6.6.post1, gritlm 1.0.2).
- Flow: the README's, as in its `examples/demo_local.py` (which already uses Llama-3.1-8B-Instruct and GritLM-7B): `HippoRAG(global_config=BaseConfig(save_dir, llm_name, llm_base_url, embedding_model_name))`, `index(docs)`, `retrieve(queries, num_to_retrieve=100)`.
- Every other `BaseConfig` field stays at its default (online OpenIE, `openie_max_workers` 8, `linking_top_k` 5, `damping` 0.5, temperature 0, embedding batch 16); the full config is written to the pod manifest.
- Depth 100 is the `retrieve` argument, not a config change (`retrieval_top_k` stays 200 and is not used by `retrieve` when the argument is given).
- Models: `meta-llama/Llama-3.1-8B-Instruct` at `0e9e39f249a16976918f6564b8830bc894c89659`, served by `vllm serve --revision` on port 6578; `GritLM/GritLM-7B` at `cb7f7ffb99c0c24ca2c325d798d7ef5c455d5339`, loaded by name by HippoRAG, so the script downloads that revision, points the hub cache `refs/main` at it and runs the adapter with `HF_HUB_OFFLINE=1`.
- vLLM server flags beyond the model: `--gpu-memory-utilization 0.5` and `--max-model-len 8192` (memory only; OpenIE asks for at most 4,096 output tokens).
- Adapter, the only code of ours on the path: unit texts (harness text `title. body`, corpus order) go in as plain-string docs; questions go in as queries in batches of 100; each retrieved passage text maps back to its unit ids.
  The export found no two units with the same text, so every passage is one unit; the group mapping stays for safety.
- C3 record: `git diff <pinned commit>` of the clone goes to `data/phase06/ga2/pod/c3-g-a2.diff` and must be empty (the session stops otherwise); `git status` goes beside it (the editable install adds an untracked `*.egg-info`).

## GPU choice
- HippoRAG's normal flow needs both models resident at once: `HippoRAG(...)` loads GritLM-7B in-process at construction, `index()` encodes passages before calling the LLM for OpenIE, and every `retrieve()` query calls the LLM for the recognition-memory fact filter after GritLM has embedded it.
- Weights alone are about 16 GB (Llama-3.1-8B bf16) plus about 14.5 GB (GritLM-7B bf16), over a 24 GB RTX 4090 before any KV cache or activations.
- The spec's "staged on one GPU (extraction, then encoding)" works for indexing only through HippoRAG's offline OpenIE mode (a non-default setting) and cannot stage retrieval without changing HippoRAG's code, which C3 forbids.
- Choice: one 48 GB GPU, RTX A6000 first (cheapest 48 GB tier, assumed), A40 or L40S as fallbacks; A100 80 GB only if no 48 GB card is offered.
  This changes the GPU basis of the spec's cost row (4090), not a criterion; the author or the orchestrator confirms before the pod starts.

## Projected time and cost (projection, not measurement)
Basis: the export (measured): 27,989 units of 51 words on average (median 43), 2,255 questions.
Phase 00's per-passage token counts (790 in, 260 out) came from longer passages; for these short units the assumed figures are about 1,300 input tokens (mostly the two few-shot prompts) and 130 output tokens per unit, about 36M input and 3.6M output tokens.
The binding constraint is HippoRAG's default `openie_max_workers` 8: decoding runs at about 8 concurrent streams.

| GPU (rate) | Setup + models | OpenIE (prefill + decode at 8 streams) | Encoding | 2,255 queries | Total + 25 % | USD |
|---|---:|---:|---:|---:|---:|---:|
| RTX A6000 48 GB (0.49 USD/h, assumed) | 0.4 h | 1.2 + 3.6 h | 0.3 h | 0.8 h | 7.9 h | 3.9 |
| L40S 48 GB (0.86 USD/h, assumed) | 0.4 h | 1.0 + 3.1 h | 0.25 h | 0.8 h | 6.9 h | 6.0 |
| A100 80 GB (1.59 USD/h, measured Phase 02) | 0.4 h | 0.5 + 1.6 h | 0.15 h | 0.5 h | 3.9 h | 6.3 |

- Only the A6000 row fits the spec's hard cut for G-A2 (4.5 USD); the L40S and A100 rows pass it at 8 OpenIE streams.
- Assumed throughputs: about 35 output tokens per second per stream on an A6000, 40 on an L40S, 80 on an A100; prefill about 8,000 to 20,000 tokens per second; GritLM about 300 short texts per second for about 280,000 passages, entities and facts; about 1.3 s per query (one filter call plus PPR).
- The A6000 and L40S rates are not in any earlier phase record: read `costPerHr` from the RunPod API before the pod and re-project.
- Raising `openie_max_workers` (for example to 64) would cut decoding by roughly 5 times and make every row fit, but it is a non-default HippoRAG setting; it is the author's call, recorded as a deviation if taken (`hipporag_run.py` does not expose it today).
- The probe (500 units, 50 questions, in its own `save_dir`) measures the pace at the start; the probe gate stops the session when its linear projection plus elapsed time passes `CUT_USD` (default 4.5) at `POD_COST_PER_HR`.
  The projection is linear, so it under-counts the graph and synonymy steps, which grow faster than linearly with entities.

## Uploads
- The repository at the frozen commit (as for G-R2).
- `data/phase06/ga2/` from the main checkout: `units.jsonl.gz`, `questions.jsonl.gz`, `bundle.manifest.json`, `bundle.sha256` (the script checks it with `sha256sum -c`).
- Recorded digests (export of 2026-10-06, `uv run python -m edge_rag.ga2 export --out <main checkout>/data/phase06/ga2`):
  - `units.jsonl.gz` sha256 bd47f86c862b1c7e980ecd625870bbc0e1becf3d55f426d77aaf2a9989ac3f34 (3,981,449 bytes, 27,989 rows);
  - `questions.jsonl.gz` sha256 96a79a341d91a02d3dd7afd722a4a88bc09b80de65caa5dafdd2ed9c561e017f (183,060 bytes, 2,255 rows);
  - `bundle.manifest.json` is rewritten by the export at the commit that holds `ga2.py`; its sha256 goes to `plan.md` with the run.

## HF_TOKEN
- Llama-3.1 is gated; the author's read token with accepted access is needed (decision 5).
- Check presence only, never print it: `env | grep -o "^HF_TOKEN="`.
  On 2026-10-06 it was absent from the agent shell and from the user-level Windows environment.
- Pass it to the pod as an environment variable at creation; the script refuses to start without it.

## Commands
On the pod, from the repository root:

```bash
HF_TOKEN=... POD_COST_PER_HR=<costPerHr> CUT_USD=4.5 \
  nohup bash scripts/pod_ga2.sh > /workspace/pod_ga2.out 2>&1 &
tail -f /workspace/pod_ga2.log    # STAGE_START / STAGE_DONE / STAGE_SKIP / STAGE_FAIL
tail -n 3 /workspace/pod_ga2.samples
grep PROBE_PACE /workspace/pod_ga2.progress
```

A rerun skips stages whose marker exists (venv, probe, full) and resumes inside the full run (see failure modes).
On the laptop, after download:

```bash
uv run python -m edge_rag.ga2 rank --bundle data/phase06/ga2 --pod data/phase06/ga2/pod
```

## Downloads
- `data/phase06/ga2/pod/`: `hipporag.jsonl.gz`, `hipporag.manifest.json` (models, revisions, HippoRAG head and status, full config, index and retrieve seconds, LLM calls and tokens offline and online, filter errors, GPU, `costPerHr`, input and output digests), `hipporag.timing.json`, `c3-g-a2.diff`, `c3-g-a2.status`, `pip-freeze.txt`, `probe/`, `sha256sums.txt`.
- `/workspace/pod_ga2.log`, `.samples`, `.progress`, `/workspace/vllm.out`.
- Optional, not needed for the ranking: `/workspace/hr_save/llm_cache/*.sqlite` (every OpenIE and filter response, the token source).

## Failure modes
- `HF_TOKEN` absent or access refused (vLLM fails to download): G-A2 `not run`, reason gated access, no other LLM (decision 5).
- Non-empty C3 diff: stop; nothing of ours may patch HippoRAG.
- Out of memory with both models live: one resize to the next tier (A100 80 GB) only if its re-projection fits the cut; `VLLM_UTIL` may be lowered first (it changes memory, not results).
- Fact-filter error during retrieval (for example the vLLM server died): HippoRAG would silently fall back to dense retrieval; the adapter counts these log records and stops at the first one, so no fallback reaches the ranking. Dense fallbacks for "no facts after reranking" are HippoRAG's normal behaviour and are counted in the timing file.
- Crash during OpenIE: rerun; HippoRAG's LLM cache replays every finished call. Crash after passage encoding but before the graph is saved: HippoRAG refuses to resume; rerun with `FULL_ARGS=--force-index-from-scratch` (re-encodes, the cache still replays OpenIE); the flag is recorded in the timing file.
- Crash during retrieval: rerun; finished question batches are kept in `hipporag.partial.jsonl`.
- Offline seconds count completed index attempts only; the wall time of a failed attempt is read from the stage log.
- Token counts come from the cache rows: a cache hit is not counted again, and rows written before the end of indexing are offline, later rows online.
- vLLM batching can change greedy outputs slightly between runs (bf16 kernels); this rival has no determinism gate in the spec beyond the C3 diff, so the run is single and its outputs are kept.
- Memorisation risk (spec): MultiHop-RAG news may be in Llama-3.1's pretraining; labelled, not testable here.

## CLI wiring for the integrator
`cli.py` is not touched here; add two subcommands that call `ga2.run_export(out)` and `ga2.run_rank(bundle, pod, rankings_dir)` (for example `edge-rag ga2-export` and `edge-rag ga2-rank`, defaults `data/phase06/ga2` and `data/phase06/ga2/pod`), or keep `python -m edge_rag.ga2`.
