# Phase 02 - Ghosts on the terrain: specification

Status: approved (frozen)
Approved: 2026-10-02, by the agent under the author's delegation (master plan, decision of 2026-10-02)
Master plan: `../0_plan_maestro.md`; ghosts and forecast: `../fase-00-state-of-the-art/survey.md` sections 5 and 7.

## Objective
Reproduce the Phase 00 ghosts on the three terrain corpora with this harness, and report each one's Full Support and cost beside the four old systems and the old Phase 17 union pool under J-strong, on the same questions.
Every later candidate is compared with the ghost of its class, so the ghosts must exist, measured here, before any candidate runs.

## Question
On HotpotQA FullWiki dev (7,405), MuSiQue (2,417) and MultiHop-RAG (2,255), what Full Support @2,048 tokens and what offline and online cost does each ghost reach, and how does it compare, paired, with the best old system on the same questions?

## What stays fixed
- Corpora, units, questions, gold and token counts: the old project's files, digest-checked through `OldData` (Phase 01).
- Metrics from depth-100 rankings, as in the master plan: FS @1,024 / @2,048 (of record) / @4,096, FS@2/5/20 units, gold share at 5, Recall@k, nDCG@10; Recall@100 and nDCG@10 are the HotpotQA dev figures that meet BEIR-HotpotQA.
- Paired test: the exact McNemar of Phase 01.
- No ghost is tuned on the terrain: published settings only; where a setting must be chosen, it is fixed here before anything runs.

## Preconditions read (survey section 5, "Preconditions for Phase 02")
Read 2026-10-02 from the sources named; "inferred" marks a conclusion not stated by the source.
- answerai-colbert-small-v1: Apache-2.0, 33M parameters, 96 dimensions, documents up to 512 tokens and queries 32 (model card); one averaged checkpoint trained on triplets including HotpotQA (Answer.AI blog): in-domain on HotpotQA.
- bge-reranker-v2-m3: Apache-2.0; card lists bge-m3-data, whose English part includes HotpotQA (BGE-M3 paper): in-domain on HotpotQA (inferred); MuSiQue not listed.
- Qwen3-Reranker-0.6B: Apache-2.0; reranker data not named, embedding data includes HotpotQA: HotpotQA labelled in-domain (inferred).
- GritLM-7B: Apache-2.0; E5S data includes the E5 public mix with HotpotQA (inferred).
- Search-R1: code Apache-2.0; checkpoints `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo` (base, used by `infer.py`) state no licence; base model Qwen2.5-7B is Apache-2.0.
  Decision: used as the ghost, labelled "checkpoint licence unstated"; results are reported, weights are not redistributed.
- HippoRAG code: MIT. Llama-3.1-8B: Llama 3.1 Community License, no non-commercial restriction (attribution and a 700M-user clause): eligible under the licence rule.
- The fusion J-strong reordered in old Phase 17 is P10-B, min-max Dense + BM25 with weights 0.5 / 0.5 from the old Phase 10 fit; it is not weight-free, so J(P10-B) is an old-project reference, not a ghost.

## Scope
In (core, in priority order):
1. A `score` command: scores any depth-100 ranking file (gz JSONL `{"qid", "ranked": [unit ids]}`) on a set, with every metric above, and a paired McNemar against another ranking file.
2. Old references, 0 USD: the four old systems (Phase 01), J(P10-B) and the union pool under J-strong, read from the old Phase 17 files.
3. G-L: answerai-colbert-small-v1 with a PLAID index (PyLate, its default PLAID settings), depth-100 rankings on the three corpora; documents truncated at 512 tokens and queries at 32, as the card states.
4. G-R: bge-reranker-v2-m3 with the old J-strong scoring code and settings over G-L's top-100 on the three corpora.
5. G-A1: Search-R1 base 7B PPO checkpoint, run by a batched vLLM driver with `infer.py`'s prompt, `</search>` stop strings and top-3 retrieval from G-L's index, greedy decoding, at most 8 search calls and 1,024 new tokens per turn; on MuSiQue and MultiHop-RAG in full.
   Its evidence list is the passages it retrieved, in retrieval order, duplicates dropped, then filled to the budget; answer EM is recorded as context only.
6. A results table (JSON, written by code) with every system's metrics, labels and cost columns, and the paired comparisons.

Optional, in this order, only under the money rule below: G-R2 (Qwen3-Reranker-0.6B over G-L's top-100, MuSiQue and MultiHop-RAG); G-A1 on 1,000 HotpotQA dev questions drawn with `random.Random(20261002).sample` over the sorted qids; G-A2 (HippoRAG 2, Llama-3.1-8B-Instruct and GritLM-7B, MultiHop-RAG only).

Out: new candidates, any fitting on the terrain, NV-Embed-v2, HippoRAG 2 on MuSiQue, Search-R1's own corpus and retriever, the exam corpus.

## Acceptance criteria
Frozen on approval. Changing them requires a deviation.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | `edge-rag score` reproduces, from the Phase 01 rankings, the gate counts of Phase 01 on the three sets, and the old Phase 17 FS@2,048 counts of J(P10-B) (5,198 / 704 / 822) and of the union pool under J-strong (5,922 / 809 / 846) from the digest-checked old files | command output equals the counts; tests on the scorer |
| C2 | G-L fidelity: nDCG@10 on HotpotQA dev within 76.1 plus or minus 1.5 points (published, same corpus and questions); a miss is recorded as a finding, not tuned | `score` output for G-L on hotpotqa-dev |
| C3 | G-R fidelity: our reranker code rescoring 300 old Phase 17 pairs (100 per set, first 100 in file order) reproduces the stored J-strong scores within 1e-3 absolute | check script output on the pod, saved JSON |
| C4 | G-A1 fidelity: on the first 50 MuSiQue questions in file order, with the same retriever, the vLLM driver and `infer.py` (patched only to greedy decoding and the same retriever endpoint) give the same first search query on at least 45 and the same normalized answer on at least 40 | saved comparison JSON |
| C5 | Each core ghost has depth-100 (G-L, G-R) or evidence (G-A1) rankings on its scope, written once with a sha256 manifest pinning model id and revision, code commit and pod `costPerHr` | manifests in `data/phase02/`, digests listed in the plan |
| C6 | The results table lists, per set, every ghost and old reference with all metrics, a label per value, offline and online time and USD (time x rate, derived), and exact McNemar on FS@2,048 of each ghost against the best old system on that set (by FS@2,048) and against the other ghosts | `data/phase02/results.json` and a summary in `results.md` generated from it |
| C7 | Money: core spend at most 7.9 USD and the phase at most 11.5 USD (time x rate); balance delta recorded; no pod left running | balance before and after, `podTerminate` confirmed, figures in the plan |
| C8 | `npm run check` passes | exit 0 |

## Money rule and stop states
- Authorized by the author: 25 USD in total for the project, 15 USD per pod session; this phase's soft caps are 7.9 USD core and 3.6 USD optional.
- A pod session stops at 8 USD; a stage whose measured rate projects past the core cap stops and is recorded.
- An optional item runs only if the phase's spend so far plus its projection stays within 11.5 USD and the money left inside 25 USD still covers the reserve of 9.9 USD (Phases 03-05 and the exam core).
- A fidelity miss (C2, C3, C4) does not block the other ghosts; the ghost is reported with the miss stated, and no setting is changed to close it.

## Anti-goals
- No tuning on the terrain, no second try of a ghost with other settings to close a gap.
- No reader, no LLM judge, no per-phase stage code beyond the scorer, the pod scripts and the results table.

## Open decisions
None.
