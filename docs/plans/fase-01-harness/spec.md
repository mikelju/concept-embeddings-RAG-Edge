# Phase 01 - Harness and reproduction gate: specification

Status: approved (frozen)
Approved: 2026-10-02, by the agent under the author's delegation (master plan approval delegated every spec approval)
Master plan: `../0_plan_maestro.md`

## Objective
A Python harness, built from the old project's code, that runs the four inherited systems over the cached artifacts of the old project and reproduces every recorded gate figure exactly.
Every later phase measures candidates on this harness, so a harness that cannot reproduce the recorded figures would make every later comparison unreadable.

## Scope
In:
- A uv project (Python >=3.12,<3.13), package `edge_rag` under `src/`, with the old project's torch source routing (pytorch-cpu on this Windows ARM64 laptop, cu126 on linux x86_64).
- Copied and adapted from the old project (handover 5.1): Dense, BM25, min-max weighted fusion, the columnwise Entity Hop, the relevance-ordered hop (P14), the budget fill and its metrics, the exact McNemar test, corpus and question loaders, the embedding cache reader, the node index reader and the pins.
- One generic module layout, not per-phase stage code.
- A CLI `uv run edge-rag reproduce --set {hotpotqa-dev,musique,multihop-rag}`.
- Recall@k (k in 2, 5, 10, 20, 100) and nDCG@10 over gold units, computed from the same rankings.
Out:
- Judges, pod code, the paper generator, GLiNER extraction runs (the indexes are cached), new corpora, new candidates.
- Re-embedding passages or questions (the cached vectors are the inputs of record).

## Acceptance criteria
Frozen on approval. Changing them requires a deviation.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | The reproduction gate of handover 5.4 is reproduced exactly by running the copied retrieval code over the cached vectors and indexes (not by reading stored rankings; stored rankings are a cross-check only). Full Support @2,048 tokens: HotpotQA dev P10-A 4,125 / P10-B 4,536 / P10-C 4,801 / P14 5,224 of 7,405; MuSiQue Dense 436 / Dense+BM25 524 / P10-C 669 / P14 761 of 2,417; MultiHop-RAG 338 / 587 / 522 / 513 of 2,255; GLiNER configuration digest `2f7864661b8ce7ff` recomputed from the recorded extraction manifests | `uv run edge-rag reproduce --set <set>` for the three sets; the result JSON's counts equal the table; the CLI prints `GATE PASS` per set |
| C2 | Every old artifact read is checked against its recorded digest and opened read-only from one data-root setting (env var `OLD_DATA_ROOT`, default the old repository's `data/`); the code refuses to write under that root | result JSON lists each digest checked; a test shows a write under the root raises; a test shows a tampered artifact is refused |
| C3 | Caches are per corpus: any cache the harness writes lives in a per-set directory, and the question-vector key includes the corpus | test on the key and directory functions |
| C4 | The paired test is the exact McNemar of old commit `9a9ad3f` (integer true division, no overflow past 1,024 discordant questions) | tests: known small values and n > 1,024 |
| C5 | `npm run check` also runs `uv run pytest -q`, `uv run ruff check .` and `uv run mypy src`, and passes | `npm run check` exit 0 |
| C6 | The harness computes, from the same rankings, Recall@k at k in {2, 5, 10, 20, 100} and nDCG@10 over gold units, per system, in the result JSON | tests on the metric functions; fields present in each result JSON |

## Constraints and risks
- The old repository is read-only; its `data/` (about 19 GB) is never moved or rewritten.
- FullWiki needs about 8 GB of passage vectors plus a BM25 rebuild (11.8 GB peak on the pod) on a 32 GB laptop; inputs are loaded in an order that frees the corpus texts before the vectors are read.
- Float summation order decides ties: the copied code keeps the old order of operations (per-query matrix-vector product, fusion summed in component order).
- A miss is a finding about the copy: the increment stops and the miss is recorded; code is never edited just to force a number.

## Open decisions
None (resolved by the agent under delegation; see plan.md "Decisions").
