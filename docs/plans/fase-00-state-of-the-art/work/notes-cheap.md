# Phase 00 - cheap-and-fast survey: working notes

Companion to `ledger-cheap.json` (128 records, all read in a source this session; some column assignments were inferred and say so in `notes`).
Metrics differ across records (nDCG@10, nDCG@1000, Recall@k, Hits@k, EM, Contain-Acc), so no figure below is compared across metrics or settings.

## Strongest per family (BEIR-HotpotQA, nDCG@10, full 5.23M-paragraph corpus, zero-shot unless noted)

Dense bi-encoders: NV-Embed-v2 7B reaches 85.5 (paper Table 14 and model card), then bge-en-icl 84.98 (paper table, wrapped header), then bge-large-en-v1.5 at 74.1 and e5-large-v2 at 73.1 (model cards).
Small dense: bge-small-en-v1.5 69.9, e5-small-v2 66.6, gte-small 63.8 (all self-reported model-card metadata).
Learned sparse: SPLADE++SD 69.3 and SPLADE-v3 69.2 (SPLADE-v3 report Table 2), SPLADEv2 68.4 (ColBERTv2 paper).
Late interaction: Jina-ColBERT-v2 76.6 and answerai-colbert-small-v1 76.1 (Jina-ColBERT-v2 paper Table 1), against ColBERTv2 66.7 to 67.5.
Zero-shot cross-encoder over a first stage: BM25 + MiniLM-L6 70.7 and Contriever + MiniLM-L6 71.5 (BEIR and Contriever papers).
Lexical reference: BM25 60.3 (Recall@100 74.0).
Hybrid and fusion (depth 1000, different metric): BM25 + SPLADE convex combination 0.751, RRF 0.737, SPLADE alone 0.727, BM25 alone 0.682 (Bruch et al. Table 5).
With BM25 + all-MiniLM-L6-v2, RRF (k=60) gives 0.675 which is below BM25 alone at 0.682, while RRF with k=5 gives 0.693 and the convex combination 0.699.

## Strongest per benchmark

HotpotQA pooled (1,000 questions, 9,811 passages, recall@5): NV-Embed-v2 94.5, GritLM-7B 92.4, GTE-Qwen2-7B 89.1, BM25 74.8 (HippoRAG 2 Table 3).
2WikiMultiHopQA pooled (6,119 passages, recall@5): NV-Embed-v2 76.5 and GritLM-7B 76.0 lead the dense models, BM25 65.3 beats Contriever at 57.5 (HippoRAG 2 Table 3).
2Wiki is the one dataset where an LLM-built graph beats every dense model by a wide margin in HippoRAG 1 (89.1 recall@5 against 68.2 for ColBERTv2), which is why a graph without an LLM is worth testing there.
MuSiQue pooled (11,656 passages, recall@5): NV-Embed-v2 69.7, GritLM-7B 65.9, GTE-Qwen2-7B 63.6, BM25 43.5.
HotpotQA fullwiki (trained on HotpotQA): MDR recall@20 80.2, MDR + ELECTRA reranker SP-EM 81.2, MDR + Beam Retrieval reranker 82.2.
Distractor-style selection (trained): Beam Retrieval EM 97.29 on HotpotQA, 77.37 on MuSiQue-Ans, 99.93 on 2Wiki; the 2Wiki figure is saturated and uninformative.
MultiHop-RAG (256-token chunks, 609 articles): best cheap figure is voyage-02 or bge-large with bge-reranker-large, Hits@10 about 0.72 to 0.75; bge-large alone 0.672; e5-base-v2 only 0.356.
BEIR: SPLADE-v3 averages 51.7 over 13 datasets; mxbai-rerank-large-v1 48.8 and bge-reranker-large 45.2 on an 11-dataset average (mixedbread card); jina-reranker-v2 53.17 and bge-reranker-v2-m3 53.65 on 17 datasets (Jina card).
QASPER (in-document, recall of answer-bearing passages at top 20 percent of a paper): zero-shot ELECTRA cross-encoder 66.39, BM25 60.38, zero-shot DPR 52.61, fine-tuned ELECTRA 79.66.
LegalBench-RAG (Recall@64, text-embedding-3-large, recursive splitter): PrivacyQA 84.19, CUAD 74.70, ContractNLI 61.72, MAUD 28.28; the Cohere reranker did not help overall.
Graph without LLM: LinearRAG reports Contain-Acc 70.2 on 2Wiki, 64.3 on HotpotQA and 33.9 on MuSiQue against 48.6, 55.7 and 26.1 for vanilla top-5, with GPT-4o-mini as the generator for all.

## Costs reported

BEIR Table 3 (DBPedia 1M documents, V100 GPU and 8-core Xeon CPU): BM25 20 ms CPU with a 0.4 GB index; TAS-B 14 ms GPU, 125 ms CPU, 3 GB; ColBERT v1 350 ms GPU, 20 GB; BM25 + MiniLM cross-encoder 450 ms GPU, 6100 ms CPU.
ColBERTv2: MS MARCO index 16 or 25 GiB against 154 GiB for ColBERT v1, and 50 to 250 ms per query in a simple Python implementation.
SPLADE-v3 reports only the FLOPS regulariser (1.2 to 1.4), not milliseconds.
MDR trained on 8 V100 GPUs and is about 10 times faster than earlier multi-hop systems (estimated from BERT passes), with HNSW on 16 CPU cores.
Beam Retrieval needs 124.64 ms per question at beam 1 and trains on a single RTX 4090 for the large encoder.
LinearRAG on 2Wiki: 249.78 s indexing and 0.093 s per query retrieval with zero LLM tokens, against 936 to 4,933 s for LLM-built graphs; RTX 4090 D.
Not reported anywhere read: encoding time for HotpotQA-scale corpora for any dense model, and index size on HotpotQA.

## Code, weights and licences

Public code and weights with permissive licences: BGE (MIT), E5 v2 (MIT), GTE small and base (MIT), mxbai-embed-large-v1 (Apache-2.0), BGE-M3 (MIT), Qwen3-Embedding (Apache-2.0), bge-reranker-v2-m3 and mxbai rerankers (Apache-2.0), ColBERTv2 (MIT), MiniLM cross-encoder (Apache-2.0).
Non-commercial: NV-Embed-v2 and Jina-ColBERT-v2 and jina-reranker-v2 (CC-BY-NC-4.0); SPLADE-v3 weights are gated on Hugging Face.
Code public: ColBERT, SPLADE, MDR, Beam Retrieval, HippoRAG, LinearRAG, BEIR.
LegalBench-RAG results use closed OpenAI and Cohere APIs, so they are context only.

## Open gaps

No per-dataset HotpotQA, MuSiQue or 2Wiki figure was found for bge-reranker-v2-m3, mxbai or jina rerankers; their cards give only BEIR averages or images.
No FullWiki or full-corpus figure was found for MuSiQue or 2Wiki with a cheap retriever; all MuSiQue and 2Wiki figures are pooled-corpus (6,119 to 11,656 passages) or distractor-style.
Qwen3-Embedding and BGE-M3 give no HotpotQA, MuSiQue or 2Wiki figure in the sources read; only MMTEB retrieval averages and NarrativeQA.
No cheap published figure was found on MultiHop-RAG beyond the original paper's embedding table, and none on QASPER beyond 2021 and 2022 baselines.
No graph method without an LLM other than LinearRAG was found; its figures are end-to-end answer accuracy, not retrieval recall, and its setting and corpus sizes are unknown.
No paper reports Full Support at a token budget, so every figure here needs a metric bridge before comparison with later phases.
Not searched in this increment: MTEB leaderboard live page, LoTTE, MS MARCO, and arXiv items after mid 2025 beyond the sources listed.
HotpotQA nDCG@10 figures from model cards are self-reported MTEB metadata and were not re-run.
