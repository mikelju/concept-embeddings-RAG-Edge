# Expensive-and-slow survey: working notes

Scope: plan step 2 of Phase 00; 156 records in `ledger-expensive.json` (154 verified in the source this session).
All figures are read in the papers, none from repository READMEs; metrics and settings are never to be mixed across rows.

## Settings that must not be mixed
HippoRAG, HippoRAG 2, SiReRAG and Adaptive-RAG use small pooled corpora of about 1,000 sampled questions (MuSiQue 11,656 passages, 2Wiki 6,119, HotpotQA 9,221 or 9,811).
IRCoT uses per-dataset corpora (HotpotQA 5,233,329 paragraphs, 2Wiki 430,225, MuSiQue 139,416).
Search-R1, ReSearch and R1-Searcher use full Wikipedia dumps (2018 DPR dump, or KILT 2019 with 29M passages) and answer metrics only.
ZeroSearch and DeepResearcher evaluate on the live web, so their setting is recorded as unknown.
KG2RAG "Full" is a pool of 66,581 Wikipedia documents, not all of Wikipedia.
RankLLaMA on HotpotQA is the only reranker figure on the 5.2M-paragraph BEIR cut; the other rerankers report no multi-hop set.

## Strongest systems per family
Iterative retrieval, retrieval recall on pooled corpora (HippoRAG paper): IRCoT + HippoRAG (ColBERTv2) reaches R@5 of 57.6 MuSiQue, 93.9 2Wiki, 83.0 HotpotQA.
Iterative retrieval, answers (GPT-3.5, same paper): IRCoT + HippoRAG gets F1 33.3 MuSiQue, 62.7 2Wiki, 59.2 HotpotQA.
Iterative retrieval, full-corpus answers: IRCoT with GPT3 gets F1 60.7 HotpotQA, 68.0 2Wiki, 36.5 MuSiQue; Iter-RetGen with 2 to 4 calls gets F1 58.6 to 61.1 on HotpotQA with text-davinci-003.
Iterative retrieval, 2Wiki: FLARE direct gets EM 51.0 and F1 59.7, above the reimplemented question-decomposition baseline (47.8 / 56.4).
Agentic RL, full Wikipedia: ReSearch-32B-Instruct gets EM 46.73 HotpotQA, 44.90 2Wiki, 26.40 MuSiQue on full dev sets (7B: 43.52, 47.59, 22.30).
Agentic RL, Search-R1 7B gets EM 43.3 HotpotQA, 38.2 2Wiki, 19.6 MuSiQue, against 13.3, 14.9, 7.2 for prompt-only IRCoT with the same 7B base.
Agentic RL on the web: DeepResearcher reports F1 52.8 HotpotQA and 59.7 2Wiki; this is not comparable to Wikipedia settings.
LLM graphs, pooled corpora: HippoRAG 2 reaches R@5 of 74.7 MuSiQue, 90.4 2Wiki, 96.3 HotpotQA and QA F1 of 48.6, 71.0, 75.5 with Llama-3.3-70B.
LLM graphs, same paper: RAPTOR, GraphRAG and LightRAG score below the NV-Embed-v2 dense baseline on HotpotQA F1 (69.5, 68.6 versus 75.3 reported for the dense model, not re-recorded here).
LLM graphs with a GPT-4o reader: SiReRAG gets F1 52.08 MuSiQue, 68.20 2Wiki, 77.36 HotpotQA; GraphRAG local search gets only 20.22, 27.49, 42.74.
LLM graphs with an 8B LLM: KG2RAG reaches HotpotQA distractor response F1 0.663 (fraction), above Hybrid RAG 0.653; GraphRAG 0.400 and LightRAG 0.293 fall below plain semantic RAG (0.617).
QASPER (in-document): RAPTOR gets F1 55.7 with GPT-4, 53.1 with GPT-3, 36.6 with UnifiedQA-3B; DPR 53.0 and BM25 50.2 with GPT-4.
LLM rerankers: RankLLaMA-13B gets 76.4 nDCG@10 on BEIR HotpotQA (7B: 75.3, RepLLaMA first stage alone 68.5).
LLM rerankers, general: RankGPT gpt-4 averages 53.68 nDCG@10 on 8 BEIR sets; Qwen3-Reranker-4B gets 69.76 on MTEB-R over a Qwen3-Embedding-0.6B top-100.
Query decomposition with an LLM: Self-Ask gets F1 55.2 HotpotQA, 48.8 2Wiki, 41.5 MuSiQue (text-davinci-003).

## Reported costs
IRCoT online: about $1 to $3 and 20 to 40 min per 1,000 queries with GPT-3.5, 2 to 4 retrieval rounds (HippoRAG paper, Table 17).
HippoRAG online: about $0.1 and 3 min per 1,000 queries, which the authors present as 10 to 30 times cheaper and 6 to 13 times faster than IRCoT.
HippoRAG indexing: about $15 and 60 min per 10,000 passages with GPT-3.5; Llama-3.1-8B takes 120 min and Llama-3.1-70B 250 min on 4xH100.
HippoRAG 2 indexing on MuSiQue (11,656 passages): 9.2M input plus 3.0M output tokens, 99.5 min, about 1.2 s per query, Llama-3.3-70B on 4xH100.
Same table: RAPTOR 1.7M plus 0.2M tokens and 100.5 min; LightRAG 68.5M plus 18.3M tokens and 235 min; GraphRAG 115.5M plus 36.1M tokens and 277 min.
KG2RAG indexing per chunk: 561 input and 22 output tokens and 1 LLM call, against LightRAG 1,269 / 381 / 1 and GraphRAG 2,791 / 629 / 5.
Iter-RetGen uses 2 LLM calls and 10 paragraphs for T=2; Self-Ask uses about 3.2 calls and 16 paragraphs per question.
Adaptive-RAG relative time is 3.6x one-step against 8.8x for multi-step (FLAN-T5-XL); per query 3.08 s one-step against 27.18 s multi-step.
RankGPT per query on TREC: gpt-4 over 100 passages $0.596 and 19,890 tokens in 10 requests; gpt-3.5 $0.040; top-30 gpt-4 step $0.098.
ZeroSearch training: about $70.8 with a simulated search LLM versus $586.7 with real Google for about 64,000 requests (about 12 h).
SiReRAG query time is 2.3 s against 1.5 s for RAPTOR.
Tokens per question for Search-R1, ReSearch and R1-Searcher are not reported in the sections read; training GPU hours were not read either.

## Reproducible with open models on one 24 GB GPU (projection, not measured)
Inference with a 7B agent (Search-R1, ReSearch-7B, R1-Searcher-7B) fits in bf16 or 8-bit; public checkpoints exist per the repositories, licences not checked.
RankLLaMA-7B, RankZephyr-7B and Qwen3-Reranker-0.6B/4B fit; Qwen3-Reranker-8B fits in bf16 at about 16 GB. Qwen3 models are Apache 2.0 (stated in the paper).
IRCoT with Flan-T5-XXL (11B) fits with 8-bit weights; its F1 is 59.1 HotpotQA, 66.5 2Wiki, 30.8 MuSiQue on 500 questions.
KG2RAG-style graphs with Llama-3-8B and HippoRAG with Llama-3.1-8B indexing fit; the 8B indexing run takes 120 min per 10,000 passages on 4xH100, so a single 24 GB GPU is slower; this is a projection.
HippoRAG 2 as published needs a 70B model (4xH100); no 24 GB result with an 8B indexer was found for HippoRAG 2.
Not reproducible: RankGPT, Iter-RetGen, FLARE and IRCoT-GPT3 (retired API models), DeepResearcher and ZeroSearch evaluation (live web).

## Gaps
No expensive system reports Full Support at a token budget; every figure is passage recall, nDCG@10, EM, F1 or LLM-judge.
No GraphRAG, LightRAG or PathRAG paper figure was recorded: their papers report LLM-judged win rates, which are not retrieval or EM/F1 figures; their numbers appear only through HippoRAG 2, SiReRAG and KG2RAG reproductions.
No expensive-class figure was found or read for MultiHop-RAG or LegalBench-RAG; MultiHop-RAG and LegalBench-RAG papers were not searched for LLM-in-the-loop results this session.
QASPER has only RAPTOR in-document answer F1; no retrieval-level QASPER metric.
No LLM reranker figure on MuSiQue or 2Wiki, and RankGPT, RankZephyr and Qwen3-Reranker report no HotpotQA row in the tables read.
IRCoT retrieval recall is only plotted in its own paper; its recall numbers here come from the HippoRAG reproduction.
Not searched: PathRAG, Rank1 and other reasoning rerankers, StepSearch, later 2026 agentic RAG work, HippoRAG 2 full-corpus runs.
R1-Searcher figures are marked unverified because the table header was not machine-readable.
HippoRAG 2 Table 2 and Table 8 extractions were column-scrambled; the recorded values were matched row by row, and HotpotQA for LightRAG was left out.
Licences were left null except for Qwen3; weights_public and code_url come from known repository names and were not opened this session.
