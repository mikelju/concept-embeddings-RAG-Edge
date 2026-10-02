# Phase 00 - State of the art: survey

Date: 2026-10-02.
Spec: `spec.md` (criteria C2 to C8); records: `ledger.json` (284 records, built and checked by `work/check_ledger.py`).
Sources behind this file: `work/notes-cheap.md`, `work/notes-expensive.md`, `work/benchmarks.md` and the old project's handover (`docs/refs/successor_project_handover.md`, sections 2.3, 5.4 and 8).
Nothing here is a measurement of this project.
Every figure is a published figure (label: reported), a figure measured by the old project (label: measured, old), arithmetic on those (derived), or a projection.
Ledger ids are cited in backticks only in the quality-cost maps and the ghost list, so the script's verified share counts exactly the figures C3 asks about.

## 1. Family coverage (C2)

Counts are records in `ledger.json` (derived from the script's family counts).

| Scope family | Ledger families counted | Records | Strongest figure found (setting) |
|---|---|---:|---|
| Dense bi-encoders, several sizes | dense bi-encoder; dense bi-encoder (closed API); dense bi-encoder (multi-hop, trained) | 68 | NV-Embed-v2 7B, nDCG@10 85.48 on BEIR-HotpotQA; small: bge-small-en-v1.5 69.9 (self-reported) |
| Hybrid lexical + dense | hybrid lexical+dense; hybrid dense+sparse+late interaction; hybrid lexical+learned sparse | 4 | BM25 + SPLADE convex combination, nDCG@1000 75.1 on BEIR-HotpotQA |
| Learned sparse (SPLADE family) | learned sparse; learned sparse (document expansion) | 9 | SPLADE++SD 69.3 nDCG@10 on BEIR-HotpotQA |
| Late interaction (ColBERT family, BGE-M3) | late interaction | 7 | Jina-ColBERT-v2 76.6 and answerai-colbert-small-v1 76.1 nDCG@10 on BEIR-HotpotQA |
| Zero-shot cross-encoder rerankers | zero-shot cross-encoder reranker; dense bi-encoder + zero-shot cross-encoder reranker; dense + zero-shot cross-encoder reranker (closed API) | 16 | Contriever + MiniLM-L6 71.5 nDCG@10 on BEIR-HotpotQA; bge-reranker-v2-m3 53.65 BEIR-17 average |
| Weight-free fusion | weight-free fusion | 4 | RRF of BM25 + SPLADE, nDCG@1000 73.7 on BEIR-HotpotQA |
| Graph methods without an LLM | graph without LLM | 3 | LinearRAG, Contain-Acc 70.2 on 2Wiki (setting unknown, end-to-end metric) |
| Iterative retrieval with an LLM (IRCoT and successors) | iterative-retrieval; query-decomposition | 63 | IRCoT + HippoRAG, passage recall@5 93.9 on 2Wiki pooled |
| Agentic RAG and search agents | agentic-rl | 24 | ReSearch-32B, EM 46.73 on HotpotQA full Wikipedia |
| LLM rerankers | llm-reranker | 9 | RankLLaMA-13B 76.4 nDCG@10 on BEIR-HotpotQA |
| LLM-built graphs (GraphRAG, HippoRAG and successors) | llm-built-graph | 60 | HippoRAG 2, passage recall@5 74.7 MuSiQue, 90.4 2Wiki, 96.3 HotpotQA (pooled) |

Every Scope family has at least one record.
Outside the Scope families the ledger also holds lexical baselines (11), trained multi-hop retrievers without an LLM (MDR, Beam Retrieval: 5) and one in-domain fine-tuned cross-encoder (QASPER), kept as context.
Thin families, said plainly: graph without an LLM has only LinearRAG, whose figures are answer accuracy in an unknown setting; hybrid lexical + dense has no figure on MuSiQue, 2Wiki or MultiHop-RAG.

## 2. Metric definitions (C4)

Full Support @B tokens (FS@B) is the metric of record: the share of questions whose every gold unit fits in a B-token context built from the ranking, over whole paragraphs.
No published system reports FS at any budget (`work/benchmarks.md` section 1; old handover fact 16).
"Computable" means computable from our depth-100 ranked lists of whole units.
The table covers every metric string in `ledger.json` (the script prints them; synonyms are grouped in the first column).

| Ledger metric strings | Definition | Unit and setting | Computable | Relation to FS |
|---|---|---|---|---|
| passage recall@2, passage recall@5 | share of gold passages in the top k, averaged over questions (HippoRAG convention, assumed, not verified) | passage; pooled 1,000-question corpora of 6,119 to 11,656 passages | yes | soft version of FS@k units; an upper bound on it |
| Recall@2 (both gold in top-2), Recall@20 (both gold in top-20), Retrieval EM (both gold in top-2), Retrieval EM (both gold paragraphs selected), Retrieval EM, Support-passage exact match | 1 if all gold paragraphs are in the top k or the selected set, averaged (MDR, Beam Retrieval) | paragraph; HotpotQA FullWiki 5.23M, or the 10- to 20-paragraph distractor sets | yes on FullWiki; distractor sets are a different setting | equals FS@k units: all-gold with a rank cut instead of a token cut |
| nDCG@10, mean nDCG@10, mean NDCG@10, mean nDCG@10 (MTEB-R), nDCG@10 (average over 8 BEIR sets), NDCG@1000 | normalised discounted cumulative gain with binary relevance, cut at 10 or 1000; "mean" rows average over several BEIR or MTEB sets | abstract; BEIR-HotpotQA is the 5,233,329-paragraph corpus with the 7,405 dev questions | yes for one set; averages over other sets are context only | rank quality, not co-presence: can be high while a gold paragraph is missing |
| Recall@100, Recall@1000 | share of relevant documents in the top k, averaged (BEIR) | abstract; BEIR-HotpotQA | yes | soft FS@k units; upper bound on FS@k units |
| Hits@10 | MultiHop-RAG original: fraction of the evidence found in the top 10, averaged | 256-token chunk; 609 articles | only after re-chunking (our unit is a paragraph) | per-question gold share; soft FS |
| MRR@10 | mean reciprocal rank of the first relevant chunk in the top 10 | 256-token chunk; MultiHop-RAG | same caveat | measures the first gold only; unrelated to FS |
| Recall@64 (character-span recall of gold snippets) | share of gold characters covered by the top 64 chunks | character span; LegalBench-RAG-mini sub-corpora | only after mapping spans to paragraphs | near FS for single-evidence queries; units differ |
| recall of answer-bearing passages at top-K% of a paper's passages | share of answer-bearing passages ranked in the top 10 or 20 percent of one paper's passages | paragraph; QASPER within one paper | yes in a within-paper setting | soft FS with a relative cut |
| Evidence-F1 (selected paragraphs vs gold evidence) | F1 of the predicted evidence set against the reference set, max over references | paragraph and float; QASPER within one paper | needs a set rule (threshold or top k) | set overlap; FS asks for all gold in a budget |
| retrieval recall (HotpotQA script, supporting facts vs retrieved chunks), retrieval F1 | recall and F1 of supporting facts covered by the retrieved chunks (KG2RAG) | chunk; HotpotQA distractor or a 66,581-document pool | no (chunk and sentence units) | soft co-presence over sentences |
| mean retrieval score | MMTEB aggregate over many retrieval tasks | mixed | no | context only |
| QA EM, QA Exact Match, EM, QA F1, F1, response F1, Answer F1 (F-1 Match) | token-level exact match or F1 of a generated answer against the gold answer | answer string; any setting | no (needs a reader) | reader quality; FS is a necessary retrieval condition only loosely |
| Contain-Acc (gold answer string contained in the generated answer) | 1 if the gold answer string appears in the generated answer | answer string | no | as above |
| LLM-as-Judge score, LLM-as-a-Judge (gpt-4o-mini), MBE (LLM-as-judge, gpt-4o-mini) | accuracy judged by an LLM against the gold answer | answer | no | as above, plus judge variance |

Comparison rule: a published figure is set beside ours only when unit, metric definition, corpus and questions all match.
The one exact match available is BEIR-HotpotQA: same 5,233,329 paragraphs and same 7,405 dev questions as our HotpotQA dev, so nDCG@10 and Recall@100 computed on our rankings are directly comparable with the BEIR rows (the paragraph count matches; byte identity of the two corpora is not verified).
Pooled-corpus figures (6,119 to 11,656 passages) are not comparable with our MuSiQue corpus of 101,962 units, and MultiHop-RAG chunk figures are not comparable with our 27,989 paragraph units.

## 3. Verified share (C3)

The script `work/check_ledger.py` counts the ids cited in this file and their `verified` flag.
Printed on 2026-10-02 (measured): 81 ids cited, 81 in the ledger, 81 verified, 100.0 percent, against the 80 percent C3 requires.
Four of the 81 are self-reported model-card figures (marked (S) below or named so in the ghost list).
Ledger-wide, 282 of 284 records are `verified: true` (measured by the script); the two unverified ones are R1-Searcher judge scores whose table header was not machine-readable, and neither is used below.
Fifteen records are self-reported model-card figures; they are marked (S) in the maps and count as verified only in the sense that the figure was read in the card.

## 4. Quality-cost maps per terrain benchmark (C5)

Classes are the ones decided in section 6 (L light, R rerank, A LLM); a system's class is assigned by its method and the class bounds, and "over L bound" marks a method of the light kind whose offline cost exceeds the light bound.
Costs are reported where the ledger has them, otherwise labelled projection.
Within one table every row shares metric, unit, corpus setting and questions; tables are never read against each other.
Values in fractions in the source are shown in percent (derived, times 100).

### 4.1 HotpotQA

**H1. BEIR-HotpotQA, full corpus 5,233,329 paragraphs, 7,405 dev questions, nDCG@10, zero-shot** (our exact setting).

| System | Class | nDCG@10 | Cost | Id |
|---|---|---:|---|---|
| NV-Embed-v2 (7B dense) | over L bound | 85.48 | encoding 5.23M paragraphs with a 7B model: about 20 to 30 GPU-h on one RTX 4090 (projection) | `c-nv-embed-v2-hotpotqa-paper` |
| NV-Embed-v1 (7B dense) | over L bound | 79.92 | as above (projection) | `c-nv-embed-v1-hotpotqa-paper` |
| Jina-ColBERT-v2 (late interaction, 560M) | L | 76.6 | not reported | `c-jina-colbert-v2-hotpotqa` |
| RankLLaMA-13B over RepLLaMA | A | 76.4 | 13B pointwise pass per candidate; not reported | `e-rankllama-13b-hotpotqa-ndcg10` |
| answerai-colbert-small-v1 (late interaction, small) | L | 76.1 | not reported; about 2 GPU-h to index (projection, section 5) | `c-answerai-colbert-small-v1-hotpotqa` |
| RankLLaMA-7B over RepLLaMA | A | 75.3 | not reported | `e-rankllama-7b-hotpotqa-ndcg10` |
| bge-large-en-v1.5 (335M dense) (S) | L | 74.1 | not reported | `c-bge-large-en-v1-5-hotpotqa-mteb` |
| e5-large-v2 (S) | L | 73.1 | not reported | `c-e5-large-v2-hotpotqa-mteb` |
| Contriever + MiniLM-L6 cross-encoder | R | 71.5 | BM25 + MiniLM: 450 ms per query GPU on DBPedia 1M (reported, BEIR Table 3) | `c-contriever-msmarco-ce-hotpotqa-ndcg` |
| BM25 + MiniLM-L6 cross-encoder | R | 70.7 | 450 ms per query GPU, 6,100 ms CPU (reported, DBPedia 1M) | `c-beir-bm25-minilmce-hotpotqa` |
| bge-small-en-v1.5 (S) | L | 69.9 | old project's Dense; measured cost in the handover | `c-bge-small-en-v1-5-hotpotqa-mteb` |
| SPLADE++ SelfDistil | L | 69.3 | not reported (FLOPS regulariser only) | `c-splade-pp-sd-hotpotqa` |
| SPLADE-v3 | L | 69.2 | not reported | `c-splade-v3-hotpotqa` |
| ColBERTv2 | L | 66.7 | 50 to 250 ms per query, index 16 to 25 GiB on MS MARCO (reported) | `c-colbertv2-hotpotqa` |
| BM25 | L | 60.3 | 20 ms per query CPU, 0.4 GB index (reported, DBPedia 1M) | `c-beir-bm25-hotpotqa` |

**H2. Same corpus and questions, Recall@100.**

| System | Class | Recall@100 | Id |
|---|---|---:|---|
| SPLADEv2 | L | 82.0 | `c-splade-v2-hotpotqa-recall100` |
| Contriever (MS MARCO) | L | 77.7 | `c-contriever-msmarco-hotpotqa-recall100` |
| BM25 | L | 74.0 | `c-bm25-hotpotqa-recall100` |

**H3. Same corpus and questions, nDCG@1000 (Bruch et al.; fusion at depth 1000).**

| System | Class | nDCG@1000 | Id |
|---|---|---:|---|
| Convex combination BM25 + SPLADE | L | 75.1 | `c-splade-bm25-tm2c2-hotpotqa-ndcg1000-bruch` |
| RRF (k=60) BM25 + SPLADE | L | 73.7 | `c-splade-bm25-rrf-hotpotqa-ndcg1000-bruch` |
| Convex combination BM25 + all-MiniLM-L6-v2 | L | 69.9 | `c-tm2c2-hotpotqa-ndcg1000-bruch` |
| RRF (k=5) BM25 + all-MiniLM-L6-v2 | L | 69.3 | `c-rrf5-hotpotqa-ndcg1000-bruch` |
| BM25 | L | 68.2 | `c-bm25-hotpotqa-ndcg1000-bruch` |
| RRF (k=60) BM25 + all-MiniLM-L6-v2 | L | 67.5 | `c-rrf60-hotpotqa-ndcg1000-bruch` |

**H4. FullWiki 5.2M, trained on HotpotQA train (in-domain, not zero-shot), all gold in the top k.**

| System | Class | Figure | Cost | Id |
|---|---|---:|---|---|
| MDR + Beam Retrieval reranker, retrieval EM top-2 | R | 82.2 | Beam: 124.64 ms per question at beam 1 (reported) | `c-beam-mdr-rerank-fullwiki-em` |
| MDR + ELECTRA reranker, support-passage EM | R | 81.2 | not reported | `c-mdr-rerank-hotpotqa-fullwiki-spem` |
| MDR, Recall@20 (both gold) | L | 80.2 | trained on 8 V100; HNSW on 16 CPU cores (reported) | `c-mdr-hotpotqa-fullwiki-r20` |
| MDR, Recall@2 (both gold) | L | 65.9 | as above | `c-mdr-hotpotqa-fullwiki-r2` |

These are the literature's closest figures to FS on HotpotQA (all-gold at a rank cut), but they are fitted on HotpotQA, so they are an in-domain upper reference, not a ghost (research protocol: nothing fitted to the corpus of use).

**H5. Pooled 1,000 questions, 9,811 passages, passage recall@5 (HippoRAG 2 paper, one table).**

| System | Class | Recall@5 | Cost | Id |
|---|---|---:|---|---|
| HippoRAG 2 (Llama-3.3-70B) | A | 96.3 | indexing scales to about 790 input and 260 output LLM tokens per passage (derived from the MuSiQue row, Table 12) | `e-hipporag2-hotpotqa-recall5` |
| NV-Embed-v2 | over L bound | 94.5 | 0.3 s per query (reported, MuSiQue) | `c-nv-embed-v2-7b-hotpotqa-r5-hipporag2` |
| GritLM-7B | over L bound | 92.4 | not reported | `c-gritlm-7b-hotpotqa-r5-hipporag2` |
| GTE-Qwen2-7B-Instruct | over L bound | 89.1 | not reported | `c-gte-qwen2-7b-instruct-hotpotqa-r5-hipporag2` |
| RAPTOR | A | 86.9 | 1.7M + 0.2M tokens, 100.5 min on MuSiQue (reported) | `e-raptor-hr2-hotpotqa-recall5` |
| Contriever | L | 75.3 | not reported | `c-contriever-hotpotqa-r5-hipporag2` |
| BM25 | L | 74.8 | not reported | `c-bm25-hotpotqa-r5-hipporag2` |

**H6. Pooled 9,221 passages, passage recall@5 (HippoRAG 1 paper, GPT-3.5 in the loop).**

| System | Class | Recall@5 | Cost | Id |
|---|---|---:|---|---|
| IRCoT + HippoRAG | A | 83.0 | IRCoT: 1 to 3 USD and 20 to 40 min per 1,000 queries (reported) | `e-ircot-hippo-hotpotqa-recall5` |
| IRCoT + ColBERTv2 | A | 82.0 | as above | `e-ircot-colbert-hotpotqa-recall5` |
| HippoRAG (ColBERTv2) | A | 77.7 | 0.1 USD and 3 min per 1,000 queries; indexing 15 USD per 10,000 passages (reported) | `e-hippo-hotpotqa-recall5` |
| Contriever | L | 75.5 | not reported | `c-h1-contriever-hotpotqa-r5` |
| GTR | L | 73.3 | not reported | `c-h1-gtr-hotpotqa-r5` |

**H7. Full Wikipedia (2018 dump), answer EM, Search-R1 paper (Qwen2.5-7B, E5, top-3 per call).**

| System | Class | EM | Cost | Id |
|---|---|---:|---|---|
| Search-R1 (PPO) | A | 43.3 | tokens per question not reported | `e-searchr1-7b-hotpotqa-em` |
| IRCoT, same 7B base | A | 13.3 | not reported | `e-searchr1-ircot-7b-hotpotqa-em` |

Search-R1 is trained on HotpotQA train, so its HotpotQA figure is in-domain; ReSearch-7B (43.52 EM) and ReSearch-32B (46.73) use another paper's setting (top-5 per call) and are not set beside it.
Our own setting has no published counterpart; the old project measured FS@2,048 on HotpotQA dev of 4,536 for Dense + BM25 and 5,922 (79.97 %) for the Phase 17 union pool under J-strong (measured, old; handover 2.3).

### 4.2 MuSiQue

**M1. Pooled 1,000 questions, 11,656 passages, passage recall@5 (HippoRAG 1 and 2 papers, same corpus).**

| System | Class | Recall@5 | Cost | Id |
|---|---|---:|---|---|
| HippoRAG 2 (Llama-3.3-70B) | A | 74.7 | 9.2M + 3.0M tokens, 99.5 min indexing on 4xH100; about 1.2 s per query (reported) | `e-hipporag2-musique-recall5` |
| NV-Embed-v2 | over L bound | 69.7 | 0.3 s per query (reported) | `c-nv-embed-v2-7b-musique-r5-hipporag2` |
| GritLM-7B | over L bound | 65.9 | not reported | `c-gritlm-7b-musique-r5-hipporag2` |
| GTE-Qwen2-7B-Instruct | over L bound | 63.6 | not reported | `c-gte-qwen2-7b-instruct-musique-r5-hipporag2` |
| RAPTOR | A | 57.8 | 1.7M + 0.2M tokens, 100.5 min (reported) | `e-raptor-hr2-musique-recall5` |
| IRCoT + HippoRAG (GPT-3.5) | A | 57.6 | 1 to 3 USD per 1,000 queries (reported) | `e-ircot-hippo-musique-recall5` |
| IRCoT + ColBERTv2 (GPT-3.5) | A | 53.7 | as above | `e-ircot-colbert-musique-recall5` |
| HippoRAG (ColBERTv2) | A | 51.9 | 0.1 USD per 1,000 queries; 15 USD per 10,000 passages indexing (reported) | `e-hippo-musique-recall5` |
| GTR (T5-base) | L | 49.1 | not reported | `c-gtr-t5-base-musique-r5-hipporag2` |
| Contriever | L | 46.6 | not reported | `c-contriever-musique-r5-hipporag2` |
| BM25 | L | 43.5 | not reported | `c-bm25-musique-r5-hipporag2` |

The rows come from two papers over the same 11,656-passage corpus and question sample; the readers and LLMs differ, so the A rows are a family, not a controlled comparison.

**M2. Same corpus, passage recall@2.**

| System | Class | Recall@2 | Id |
|---|---|---:|---|
| HippoRAG 2 | A | 56.1 | `e-hipporag2-musique-recall2` |
| IRCoT + HippoRAG | A | 45.3 | `e-ircot-hippo-musique-recall2` |
| IRCoT + ColBERTv2 | A | 41.7 | `e-ircot-colbert-musique-recall2` |
| HippoRAG | A | 40.9 | `e-hippo-musique-recall2` |

**M3. Full Wikipedia (2018 dump), answer EM, Search-R1 paper.**

| System | Class | EM | Id |
|---|---|---:|---|
| Search-R1 (Qwen2.5-7B, PPO) | A | 19.6 | `e-searchr1-7b-musique-em` |
| IRCoT, same 7B base | A | 7.2 | `e-searchr1-ircot-7b-musique-em` |

**M4. Distractor setting (20 paragraphs per question), trained.**
Beam Retrieval reaches retrieval EM 77.37 on MuSiQue-Ans (class R by cost, but fitted on MuSiQue train; context only) (`c-beam-musique-ans-em`).

No full-corpus retrieval figure was found for MuSiQue with any system; our corpus (101,962 pooled units, 2,417 validation questions) has no published counterpart.
Old measured FS@2,048: Dense + BM25 524, union pool under J-strong 809 (33.47 %) of 2,417 (measured, old).

### 4.3 2WikiMultiHopQA (context: not one of our terrain corpora)

**W1. Pooled 1,000 questions, 6,119 passages, passage recall@5 (HippoRAG 1 and 2 papers).**

| System | Class | Recall@5 | Id |
|---|---|---:|---|
| IRCoT + HippoRAG (GPT-3.5) | A | 93.9 | `e-ircot-hippo-2wiki-recall5` |
| HippoRAG 2 (Llama-3.3-70B) | A | 90.4 | `e-hipporag2-2wiki-recall5` |
| HippoRAG (ColBERTv2) | A | 89.1 | `e-hippo-2wiki-recall5` |
| NV-Embed-v2 | over L bound | 76.5 | `c-nv-embed-v2-7b-2wiki-r5-hipporag2` |
| GritLM-7B | over L bound | 76.0 | `c-gritlm-7b-2wiki-r5-hipporag2` |
| IRCoT + ColBERTv2 | A | 74.4 | `e-ircot-colbert-2wiki-recall5` |
| ColBERTv2 | L | 68.2 | `c-h1-colbertv2-2wiki-r5` |
| GTR (T5-base) | L | 67.9 | `c-gtr-t5-base-2wiki-r5-hipporag2` |
| BM25 | L | 65.3 | `c-bm25-2wiki-r5-hipporag2` |
| Contriever | L | 57.5 | `c-contriever-2wiki-r5-hipporag2` |

2Wiki is where an LLM-built graph beats every dense model by the widest margin (entity-centric questions); the gap is much smaller on HotpotQA and MuSiQue.

**W2. Full Wikipedia, answer EM, Search-R1 paper.**

| System | Class | EM | Id |
|---|---|---:|---|
| Search-R1 (Qwen2.5-7B, PPO) | A | 38.2 | `e-searchr1-7b-2wiki-em` |
| IRCoT, same 7B base | A | 14.9 | `e-searchr1-ircot-7b-2wiki-em` |

The distractor figure of Beam Retrieval (99.93 retrieval EM) is saturated and uninformative.

### 4.4 MultiHop-RAG

**R1. 609 articles in 256-token chunks, 2,255 answerable queries, Hits@10 (original definition: share of evidence in the top 10).**

| System | Class | Hits@10 | Cost | Id |
|---|---|---:|---|---|
| bge-large-en-v1.5 + bge-reranker-large (top-20) | R | 71.83 | not reported | `c-mhrag-bge-large-h10-rr` |
| bge-large-en-v1.5 | L | 67.18 | not reported | `c-mhrag-bge-large-h10` |
| instructor-large + bge-reranker-large | R | 65.9 | not reported | `c-mhrag-instructor-large-h10-rr` |
| instructor-large | L | 57.17 | not reported | `c-mhrag-instructor-large-h10` |
| e5-base-v2 + bge-reranker-large | R | 41.76 | not reported | `c-mhrag-e5-base-v2-h10-rr` |
| e5-base-v2 | L | 35.56 | not reported | `c-mhrag-e5-base-v2-h10` |

**R2. Same setting, MRR@10.**

| System | Class | MRR@10 | Id |
|---|---|---:|---|
| bge-large-en-v1.5 + bge-reranker-large | R | 56.3 | `c-mhrag-bge-large-mrr10-rr` |
| bge-large-en-v1.5 | L | 42.98 | `c-mhrag-bge-large-mrr10` |

No class-A figure on MultiHop-RAG was found; Ammann et al. 2025 use another Hits@k definition and an unstated split, so they are left out.
Old measured FS@2,048 on our 27,989 paragraph units: Dense + BM25 587, J-strong over Dense + BM25 822, union pool under J-strong 846 (37.52 %) of 2,255 (measured, old).

### 4.5 What the maps say (interpretation)

On zero-shot first stages, late interaction is the strongest family under one billion parameters on BEIR-HotpotQA (76.1 to 76.6 against 74.1 for the best dense model of that size and 69.3 for SPLADE).
The 7B dense encoders lead every table where they appear, but their offline cost on a 5.2M corpus is an order of magnitude above the rest (projection).
LLM-built graphs and LLM loops lead on pooled corpora, most clearly on 2Wiki, but every published run uses a 70B model or a closed API and corpora of 6,000 to 12,000 passages.
Search agents with an open 7B model are the only class-A systems published on full Wikipedia, and they report answers, not retrieval.

## 5. Ghost shortlist (C6)

One ghost per cost class, plus a second class-A ghost because the master plan requires an agentic ghost and the strongest class-A retrieval figures come from LLM-built graphs.
All projections assume one RTX 4090 (24 GB) on RunPod Secure Cloud at 0.74 USD/h (handover section 8), and our terrain: HotpotQA FullWiki 5,233,329 paragraphs with 7,405 dev questions, MuSiQue 101,962 units with 2,417 questions, MultiHop-RAG 27,989 units with 2,255 questions (handover 5.4).
Every hour figure below is a projection, including a setup allowance; Section 7 adds a contingency.

### G-L: light class - answerai-colbert-small-v1 (late interaction)

- Published figure: nDCG@10 76.1 on BEIR-HotpotQA, zero-shot, same corpus and questions as our HotpotQA dev (`c-answerai-colbert-small-v1-hotpotqa`, Jina-ColBERT-v2 paper Table 1).
- Why the strongest reproducible in class: it is within 0.5 points of the best sub-billion model (Jina-ColBERT-v2, 76.6, `c-jina-colbert-v2-hotpotqa`, whose weights are CC-BY-NC-4.0), above every dense model under one billion parameters (bge-large-en-v1.5, 74.1, self-reported) and every learned sparse model (69.3), and it is small enough to index 5.2M paragraphs within the light offline bound.
  NV-Embed-v2 (85.48) is stronger but over the light offline bound (section 6) and non-commercial; it is kept as an optional upper reference, not a ghost.
- Reproduction check: our nDCG@10 on HotpotQA dev must land near 76.1, because the setting matches; a miss is a finding about the copy.
- Code and weights: weights on Hugging Face (`answerdotai/answerai-colbert-small-v1`, cited by the Jina paper); code through the ColBERT or PyLate libraries.
  Gap: the ledger record has no code URL or licence, and the model card was not opened this phase; the Phase 02 spec opens it first, and if its licence or weights are not usable the pre-declared fallback is Jina-ColBERT-v2 (76.6, card read).
- A BM25 fusion of this ghost has no published figure, so it is a candidate for Phases 03 to 05, not part of the ghost.
- Projected cost: encoding about 5.23M paragraphs of roughly 70 tokens each with a small encoder takes under 0.5 GPU-h; with compressed indexing, upload and the two small corpora it is about 2.0 GPU-h in all, about 1.5 USD (projection).

### G-R: rerank class - bge-reranker-v2-m3 (J-strong) over a fused pool

- Reference figure: already measured by the old project on our terrain, FS@2,048 of the Phase 17 union pool under J-strong 5,922 / 7,405, 809 / 2,417, 846 / 2,255 (measured, old; handover 2.3); its published figure is a BEIR-17 average nDCG@10 of 53.65 (`c-bge-reranker-v2-m3-beir17`, self-reported, measured by Jina).
- Why the strongest reproducible in class: it is the strongest zero-shot cross-encoder with an Apache-2.0 licence in the ledger, it lifted all twelve system-set pairs in old Phase 17 (+5.59 to +12.59 pp, measured, old), and its query cost is measured on our harness (0.45 to 0.54 s per question, one RTX 4090).
  No per-dataset multi-hop figure was found for any modern cross-encoder (gap), so its published figure is context.
- Code and weights: `BAAI/bge-reranker-v2-m3` on Hugging Face, Apache-2.0 (card read).
- Phase 02 plan: reuse the measured scores files read-only after their digests check (0 USD), and rerank G-L's top-100 with the same judge so the class has a ghost on a literature first stage.
- Projected cost of the new reranking: 7,405 x 0.448 + 2,417 x 0.542 + 2,255 x 0.446 seconds is 1.56 GPU-h (derived from measured per-question times), about 1.9 GPU-h with setup, about 1.4 USD (projection).

### G-A1: LLM class, agentic - Search-R1 (Qwen2.5-7B, PPO)

- Published figure: answer EM 43.3 HotpotQA, 19.6 MuSiQue, 38.2 2Wiki on full Wikipedia, against 13.3, 7.2 and 14.9 for IRCoT with the same 7B base (`e-searchr1-7b-hotpotqa-em`, `e-searchr1-7b-musique-em`, `e-searchr1-7b-2wiki-em`).
- Why the strongest reproducible in class: it is the open search agent that later agents (ReSearch, R1-Searcher, ZeroSearch) take as their baseline, it has public code and checkpoints, its 7B model fits one RTX 4090, and it reads only 3 passages per search call, which keeps its evidence inside a 2,048-token budget.
  ReSearch-7B reports higher EM in its own setting (top-5 per call), which is not the same setting, so it is not ranked above.
- What we score: the passages the agent retrieves, in retrieval order, cut at 2,048 tokens (FS), with the agent's own answer EM as a fidelity check against the published figure.
  Search-R1 is trained on HotpotQA train, so HotpotQA is in-domain for it; MuSiQue and MultiHop-RAG are out-of-domain.
- Code and weights: `github.com/PeterGriffinJin/Search-R1`, checkpoints public per the repository; licence not checked (gap, to open at the Phase 02 spec).
- Scope: MuSiQue (2,417) and MultiHop-RAG (2,255) in full, and a preregistered random subsample of 1,000 HotpotQA dev questions; the retriever inside the loop is G-L's index.
- Projected cost: about 5,700 questions, about 3 search calls and 600 generated tokens each, about 1,000 output tokens per second batched with vLLM: about 1.0 GPU-h of generation, 0.5 GPU-h of prefill and retrieval, 0.5 GPU-h setup, 2.0 GPU-h, about 1.5 USD (projection).

### G-A2: LLM class, LLM-built graph - HippoRAG 2 with an open 8B model

- Published figure: passage recall@5 74.7 MuSiQue, 90.4 2Wiki, 96.3 HotpotQA on pooled corpora with Llama-3.3-70B (`e-hipporag2-musique-recall5`, `e-hipporag2-2wiki-recall5`, `e-hipporag2-hotpotqa-recall5`).
- Why the strongest reproducible in class: it is the best class-A retrieval figure on every pooled table (M1, H5; second only to IRCoT + HippoRAG on 2Wiki, which needs a retired GPT-3.5 loop), and its code is public.
- Deviation from the published recipe, stated up front: a 70B model does not fit 24 GB, so we run HippoRAG 2's code with Llama-3.1-8B-Instruct (the 8B model HippoRAG 1 used for indexing); the published 70B figure is context, and our figure is a reduced reproduction.
- Scope: MultiHop-RAG only (27,989 units); MuSiQue (101,962 units) is out of the forecast and runs only if money remains (section 7).
- Code and weights: `github.com/OSU-NLP-Group/HippoRAG`; licence not checked (gap).
- Projected cost: indexing tokens scale from the published 790 input and 260 output tokens per passage (derived from 9.2M + 3.0M over 11,656) to about 22M input and 7.2M output tokens; at about 1,500 output tokens per second batched with vLLM that is about 1.6 GPU-h, plus 0.3 GPU-h of passage encoding, 0.6 GPU-h of queries at about 1 s each and 0.5 GPU-h setup: about 3.0 GPU-h, about 2.2 USD (projection).
  MuSiQue would add about 8 GPU-h, about 5.9 USD (projection), above the class-A offline bound of section 6, which is why it is not in the forecast.

Ghost costs together for Phase 02: about 8.9 GPU-h and 6.6 USD before contingency (derived from the projections above).

## 6. Recommendations for the open decisions (C7)

These are decided by the agent under the author's delegation and recorded in the master plan's decisions table.

1. **Metrics.** FS@2,048 is the metric of record; beside it, from the same depth-100 rankings, report all-gold at k units (FS@2, FS@5, FS@20, the MDR family), per-question gold share at 5 (the HippoRAG convention) and nDCG@10 plus Recall@100 on HotpotQA dev, which match BEIR-HotpotQA exactly.
   Answer EM is reported only for the agentic ghost, as a fidelity check against its published figure; no reader is added to the tournament.
   Reason: these are computable at no cost from our rankings, and nDCG@10 on HotpotQA is the one setting where our figures meet the literature's directly.
2. **Reading budget.** 2,048 tokens of record, with 1,024 and 4,096 reported beside it.
   Reason: 2,048 is the old line's budget and MultiHop-RAG's own reader budget, and the other two cost nothing from the same rankings and show how the ranking degrades.
3. **Cost classes and bounds** (mean per question, on one RTX 4090 unless stated):
   - L, light: no generative LLM anywhere and models of at most 1B parameters; online at most 0.1 s per question on the GPU or 2 s on the laptop CPU; offline at most 2 GPU-h per million units.
   - R, rerank: L plus zero-shot cross-encoders of at most 1B parameters over at most the top-100; online at most 1 s per question; no generative LLM.
   - A, LLM: open models of at most 8B parameters that fit 24 GB, online or offline; online at most 30 s per question; offline at most 5 GPU-h per corpus.
   Reason: the bounds put the measured J-strong (0.45 to 0.54 s) inside R, keep GLiNER and the old Entity Hop inside L, and keep each class-A run inside the money cap.
4. **LLM in the heavy online loop.** Yes, in class A only, with open models of at most 8B on one RTX 4090 and no closed APIs, run on MuSiQue and MultiHop-RAG in full and on a preregistered 1,000-question HotpotQA dev subsample.
   Reason: the agentic ghost needs it either way, and with an open 7B model it costs about 1.5 USD per terrain pass (projection).
5. **Offline LLM work in the light class.** No: class L admits encoder models of at most 1B offline (such as GLiNER and the encoders) but no generative LLM; offline LLM work (graph building, document expansion by an LLM) belongs to class A.
   Reason: the light class must stay cheap on any corpus, and LLM graph building costs about 2 USD per 28,000 units here (projection) and grows with the corpus.
6. **Exam corpus.** QASPER, with the pooled-across-papers setting declared as new and a within-paper control so one published setting is reachable.
   Reason: native paragraph gold, 55.5 % of answerable text-evidence questions need more than one paragraph, CC BY 4.0 and a split by paper; LegalBench-RAG is single-document with 500-character span gold, so FS would collapse to single-evidence recall and its published figures would need re-chunking (`work/benchmarks.md` section 5).
   This reverses the master plan's provisional recommendation of LegalBench-RAG; before the exam freezes, count the QASPER test paragraphs and the share of test questions with two or more text paragraphs.

## 7. RunPod forecast for Phases 01 to 06 (C8, projection)

Constraint: the author authorized 25 USD in total and 15 USD per pod session; the account balance on 2026-10-02 is 4.22 USD (stated by the author).
Rate: 0.74 USD/h, one RTX 4090, Secure Cloud.
Every figure in this table is a projection.

| Phase | Work on the GPU | GPU-h | USD |
|---|---|---:|---:|
| 01 | harness reproduction on the laptop from cached artifacts; GPU only if a cache must be rebuilt | 0 to 0.7 | 0.5 (reserve) |
| 02 | G-L 2.0 h, G-R 1.9 h, G-A1 2.0 h, G-A2 on MultiHop-RAG 3.0 h, plus 30 % contingency for probes, downloads and reruns | 11.6 | 8.6 |
| 03 | first candidate (class set by its spec) | 3.4 | 2.5 |
| 04 | second candidate | 3.4 | 2.5 |
| 05 | third candidate | 3.4 | 2.5 |
| 06 | exam on QASPER test (about 1,451 questions; paragraph count unknown, assumed at most 30,000): G-L, G-R, G-A1, G-A2 and the chosen candidates, plus 15 % margin | 6.8 | 5.0 |
| Total | | about 29 | 21.6 |

Phases 02 to 06 come to 21.1 USD, inside the 25 USD authorized, with 3.4 USD unallocated (derived from the rows).
Recommended cap: keep the authorized 25 USD in total and 15 USD per pod session, with the per-phase figures above as soft caps that each phase's spec states; the exam's 5.0 USD is ring-fenced and no earlier phase may spend it.
The 4.22 USD balance covers Phase 01 and G-L plus G-R in Phase 02 (about 2.9 USD before contingency); the account needs a top-up within the authorized 25 USD before G-A1 and G-A2 run.

Priority order, spent top to bottom; if money runs short, items are dropped from the bottom up:
1. G-R reuse of the measured Phase 17 scores (0 USD).
2. G-L on the three terrain corpora.
3. G-R over G-L's top-100.
4. G-A1 on MuSiQue and MultiHop-RAG.
5. The exam with G-L, G-R, G-A1 and the chosen candidates (ring-fenced).
6. G-A1 on the 1,000-question HotpotQA subsample.
7. G-A2 on MultiHop-RAG, then G-A2 on the exam.
Dropped first if short: G-A2 on the exam, then G-A2 entirely, then the HotpotQA agent subsample, then the candidates' budgets shrink.
Outside the forecast, needing the author: HippoRAG 2 on MuSiQue (about 5.9 USD), NV-Embed-v2 as an upper reference (about 1 USD on MuSiQue and MultiHop-RAG, 15 to 22 USD on FullWiki), and any model above 8B.

## 8. Gaps carried forward

- No system reports Full Support at any budget; every published figure needs the metric bridge in section 2.
- Leaderboards for MuSiQue (host unreachable) and for QASPER evidence retrieval were not read; MTEB HotpotQA rows were not read.
- Licences are null in 256 of 284 records; code URLs and `weights_public` partly come from known repository names and were not opened.
- No per-dataset multi-hop figure for modern cross-encoders or LLM rerankers on MuSiQue, 2Wiki or MultiHop-RAG.
- No class-A figure on MultiHop-RAG or LegalBench-RAG; no full-corpus retrieval figure on MuSiQue.
- HippoRAG's recall@5 definition (all-gold or per-question share) is assumed, not verified.
- All GPU-hour figures for the ghosts are projections from published token counts and assumed throughputs; Phase 02 measures them.
