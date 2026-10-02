# Phase 00, step 3: benchmarks and metrics

Compiled 2026-10-02 by a research subagent, for `survey.md` and the ledger.
Labels: READ means I read the figure in the cited source this session (paper PDF converted to text, README, or page).
INHERITED means copied from the previous project's `corpora_survey_2026-09-30.md` and not re-read.
SNIPPET means seen only in a search-result summary.
Nothing here is a measurement of this project.
Full Support (FS) is the metric of record: share of questions whose gold units all fit in a 2,048-token context built from our ranking, over whole paragraphs.
Our harness stores depth-100 ranked lists of whole units, so any metric that needs only the rank positions of gold units can be computed from them.

## 1. Cross-cutting findings

No benchmark below reports Full Support at a token budget over whole paragraphs.
The closest literature metrics are MDR's paragraph exact match (R@2 for two-gold questions) and QUEST-style MRecall@K (INHERITED, see the old survey).
MultiHop-RAG's response experiment retrieves top-6 chunks of 256 tokens "so that the total length of the retrieved text does not exceed 2,048" (arXiv 2401.15391, Section 4.2 experiment setup, READ), which is the same budget as ours but over chunks, with accuracy as the metric.
The name "Recall@k" hides at least three definitions: any gold unit in top-k, share of gold units in top-k (averaged over questions), and all gold units in top-k.
Two papers using "Hits@k" on MultiHop-RAG define it differently (section 2.6).
Published numbers must therefore never be compared with ours unless the definition, unit and corpus setting are all matched.

## 2. Per-benchmark sections

### 2.1 HotpotQA (distractor and FullWiki)

- Source: Yang et al. 2018, arXiv 1809.09600 (READ, Table 1 and Section 3 to 5).
- Unit: Wikipedia first paragraph (abstract) of an article; gold is 2 paragraphs plus sentence-level supporting facts.
- Splits (Table 1): train-easy 18,089; train-medium 56,814; train-hard 15,661; dev 7,405; test-distractor 7,405; test-fullwiki 7,405; total 112,779 (the dev cell is garbled in the text extraction; 7,405 is consistent with the stated total, which I checked by sum).
- Distractor setting: each question comes with 10 paragraphs, the 2 gold plus 8 distractors retrieved by bigram tf-idf with the question as query, shuffled (Section 3).
- FullWiki setting: the first paragraphs of all Wikipedia articles (the paper says "5,000,000+ wiki paragraphs"); distractor and fullwiki use different test sets so that fullwiki gold stays hidden (Table 1 caption).
- Licence: CC BY-SA 4.0 (hotpotqa.github.io, READ 2026-10-02).
- Metrics (Section 5): answer EM and F1; supporting-fact EM and F1 (sentence level); joint EM and F1 with joint P = P(ans) x P(sup), joint R likewise, joint F1 from those.
- Paragraph-level retrieval metrics come from later work: MDR's R@k and paragraph EM (section 3 below), BEIR's nDCG@10 (section 2.5).
- Leaderboard (hotpotqa.github.io, READ 2026-10-02, a table of Ans EM/F1, Sup EM/F1, Joint EM/F1):
  - Distractor: Beam Retrieval (Aug 7, 2023) 72.69/85.04, 66.25/90.09, 50.53/77.54; PipNet (Jul 7, 2022) 72.26/84.86; Smoothing R3 (Jun 27, 2022) 72.07/84.34.
  - FullWiki: AISO (May 10, 2021) 67.46/80.52, 61.17/86.02, 44.87/72.00; Chain-of-Skills (Jan 31, 2023) 67.38/80.14, 61.25/85.31, 45.65/71.65; TPRR (Feb 1, 2021) 66.95/79.50.
  - The page says the latest fullwiki entry is dated 2024-06-25, so the leaderboard is effectively frozen and does not list 2025-2026 LLM systems.
- Pooled setting used by graph-RAG papers: 1,000 dev questions with only their own passages, 9,811 passages (HippoRAG 2, arXiv 2502.14802, Table 1, READ).

### 2.2 MuSiQue (Ans and Full)

- Source: Trivedi et al. 2022, arXiv 2108.00573 (READ, Sections 4 to 5, Table 2).
- Unit: Wikipedia paragraph; each question has 20 paragraphs (supporting plus distractors, Step S6).
- Hops: 2 to 4, six reasoning-graph types; 7,676 supporting paragraphs overall in the dataset text.
- MuSiQue-Ans sizes (Table 2): train 19,938; dev 2,417; test 2,459 (24,814 total).
- MuSiQue-Full holds twice that, one unanswerable contrast question per answerable one (Table 2 caption).
- Licence: CC BY 4.0 (github.com/StonyBrookNLP/musique, READ).
- Metrics: answer F1; support F1 (paragraph-level, with the supporting paragraphs as the target set); Full adds group answer-sufficiency and group support-sufficiency F1 (GitHub README, READ).
- Human reference: answer F1 78.0, support F1 93.9 on one split of Table 3 (the table has three columns; I did not map which; quote only as order of magnitude).
- Retrieval setting in the graph-RAG literature: 1,000 dev questions, 11,656 pooled passages (HippoRAG 2, Table 1, READ).
- Leaderboards: allenai leaderboards for Ans and Full are linked from the repository; host `leaderboard.allenai.org` did not resolve from my environment on 2026-10-02, so no current top figures were read (gap).
  The README shows a dev-set sample of a Select+Answer model: Ans answer F1 0.497, support F1 0.792; Full 0.498 and 0.777 (READ, README, not a leaderboard top).

### 2.3 2WikiMultiHopQA

- Source: Ho et al. 2020, arXiv 2011.01060 (READ, Section 3 to 4, Tables 1 and 5).
- Unit: Wikipedia paragraph; context of 10 paragraphs (2 gold plus 8 distractors; bridge-comparison has 4 gold plus 6 distractors), distractors by tf-idf top-50 then entity-type filtering (Section 3).
- Splits (Table 1): train 167,454 (train-medium 154,878 plus train-hard 12,576); dev 12,576; test 12,576; total 192,606.
- Question types (Table 2): comparison 57,989; inference 7,478; compositional 86,979; bridge-comparison 40,160.
- Gold: supporting facts (title, sentence) plus Wikidata evidence triples.
- Licence: Apache-2.0 (github.com/Alab-NII/2wikimultihop, READ).
- Metrics: answer EM/F1, supporting-fact EM/F1, evidence (triple) EM/F1, joint EM/F1; baseline test: answer EM 36.53, F1 43.93; Sp Fact F1 65.26 (Table 5).
- Pooled setting of graph-RAG papers: 1,000 questions, 6,119 passages (HippoRAG 2, Table 1).
- Leaderboard: the repository README (READ) says NA-Reviewer (Oct 2021) leads with 76.73 answer EM and 52.75 joint EM and that Beam Retrieval (Aug 2023) reaches 88.47 answer EM; I did not open the leaderboard table itself, so treat the second figure as the README's claim.

### 2.4 Pooled "1,000 questions" setting (HippoRAG family)

- HippoRAG 2 (arXiv 2502.14802, Table 1, READ) samples 1,000 queries per dataset and indexes only the passages tied to those queries: NQ 9,633; PopQA 8,676; MuSiQue 11,656; 2Wiki 6,119; HotpotQA 9,811; LV-Eval 124 queries over 22,849; NarrativeQA 293 over 4,111.
- Metric: passage recall@5 (Section 4.3); the QA reader uses the top-5 passages.
- Table 3 (passage recall@5, MuSiQue / 2Wiki / HotpotQA): BM25 43.5 / 65.3 / 74.8; NV-Embed-v2 69.7 / 76.5 / 94.5; HippoRAG 2 74.7 / 90.4 / 96.3.
  The extracted table is column-scrambled; I mapped the rows by dataset order and by cross-checking Table 9.
- Table 9 gives recall@2 and recall@5 jointly, for example NV-Embed-v2 on HotpotQA 84.1/94.5 (same caveat).
- The definition of recall (share of gold passages in top-5 averaged over questions, versus all-gold) is not stated in the text I read; HippoRAG's code convention is the per-question share, but that is NOT verified here.
- Corpus sizes of 6 to 12 thousand passages are far smaller than FullWiki (5.2 million) and about a third of our previous corpora (the old project's MuSiQue corpus had 101,962 units, INHERITED), so these figures are not comparable with whole-corpus FullWiki figures.

### 2.5 BEIR-HotpotQA

- Source: Thakur et al. 2021, arXiv 2104.08663 (READ, Table 1, Table 2, Appendix licence line).
- Setting: full pooled corpus of Wikipedia abstracts; 7,405 test queries; 2.0 relevant documents per query on average; corpus 5,233,329 (the Table 1 text extraction scrambles the corpus column; 5,233,329 also appears in a Hugging Face forum post, SNIPPET).
- Unit: abstract (the HotpotQA paragraph); relevance is binary.
- Metric: nDCG@10; Recall@100 is in Table 9.
- Table 2 nDCG@10 on HotpotQA: BM25 0.603; ColBERT 0.593; BM25+cross-encoder 0.707 (best in the table).
- Current top: Nomic Embed v1.5 reported 0.7151 on HotpotQA with no reranker (Hugging Face forum post, SNIPPET, self-reported); monoT5-3B 0.759 (SNIPPET).
  I did not read the MTEB leaderboard rows for HotpotQA (gap).
- Licence: CC BY-SA 4.0 for the HotpotQA part (BEIR appendix, READ).
- Relation to FS: nDCG@10 with two gold per query rewards early rank, not co-presence; see section 4.

### 2.6 MultiHop-RAG

- Source: Tang and Yang 2024, arXiv 2401.15391 (READ, Tables 2 to 6).
- Corpus: 609 news articles chunked into 256-token chunks for retrieval; the knowledge base is pooled across all queries.
- Queries (Table 3): 2,556; inference 816; comparison 856; temporal 583; null 301.
- Evidence per query (Table 4): 0 (null) 301; 2 evidence 1,078; 3 evidence 779; 4 evidence 398.
- Gold: evidence sentences with article metadata; unit for retrieval metrics is the 256-token chunk containing evidence.
- Licence: ODC-BY (Hugging Face card `yixuantt/MultiHopRAG`, READ).
- Metrics (Section 3.3 text, READ): MAP@K (average top-K precision across queries), MRR@K (reciprocal rank of first relevant chunk), Hit@K ("the fraction of evidence that appears in the top-K retrieved set", so a recall over evidence); null queries excluded from retrieval evaluation.
- Table 5 (best rows, with bge-reranker-large over top-20): voyage-02 MRR@10 0.586, MAP@10 0.4795, Hits@10 0.7467, Hits@4 0.6625; bge-large-en-v1.5 Hits@10 0.7183.
- Table 6 (end-to-end accuracy, top-6 chunks within 2,048 tokens): GPT-4 0.56 with retrieved chunks, 0.89 with ground-truth chunks.
- Definition conflict: Ammann et al. 2025 (arXiv 2507.00355, Section 4.3, READ) define Hits@k as the percentage of questions with at least one gold evidence in the top-k, which is not the original definition; their Table 1 reports Hits@10 0.872, MRR@10 0.635, MAP@10 0.322 for question decomposition plus reranking (bge-large-en-v1.5, bge-reranker-large) on an "eval split" of MultiHop-RAG.
  Their figures are therefore not comparable with Table 5 of the original paper without recomputation.
- Leaderboard: none found; the top figures in the literature are the ones above plus unread SNIPPET claims.

### 2.7 QASPER (exam candidate)

- Source: Dasigi et al. 2021, arXiv 2105.03011 (READ, Sections 1 to 4, Table 3); Hugging Face card `allenai/qasper` (READ).
- Corpus: 1,585 NLP papers (arXiv, full text parsed by S2ORC; 18K papers were collected for annotation, 1,585 annotated); questions 5,049; on average 3.2 questions per paper, at most 12.
- Splits (HF card): train 888 papers and 2,593 questions; validation 281 papers and 1,005 questions; test 416 papers; the test question count is not on the card and follows from the totals as 1,451 (DERIVED: 5,049 - 2,593 - 1,005).
  Each paper appears in only one split (paper, Section 4.1).
- Unit: paragraph of the paper (also figures and tables as evidence); gold evidence is a minimal set of paragraphs, figures or tables chosen by annotators.
- Evidence per question: 55.5% of the answerable questions with text-only evidence need more than one paragraph; 13% need a table or figure; unanswerable questions have no evidence (paper, Sections 1 and 3).
  Mean number of evidence paragraphs per question: not reported in the text I read (gap).
- Multiple annotators: dev and test questions have multiple reference answers (98% in test, 74% in validation), and scores take the max over references.
- Licence: CC BY 4.0 (HF card, READ).
- Metrics: Answer-F1 (span F1 with max over references); Evidence-F1 (F1 over the set of paragraphs, figures and tables chosen against the reference evidence, max over references) (Section 3.2 and 4.1).
- Evidence-F1 baselines (Table 3): LED-large 31.25 dev, 39.37 test; LED-base 23.94 dev, 29.85 test; TF-IDF 10.68 dev, 9.20 test; random paragraph 2.09 and 1.30; human lower bound 71.62.
- Retrieval setting of published work: within one paper (the system reads or retrieves from the single paper the question was asked about); the pooled-across-papers setting is not used by the paper and, among the sources I opened, by none.
  RAPTOR and SKETCH use QASPER within a paper (found by search, not opened: SNIPPET).
- Leaderboard: a Papers-with-Code or Hugging Face leaderboard is mentioned in a SNIPPET (answer metrics only, UnifiedQA-large 61.39); no 2025-2026 evidence-retrieval top figures were read (gap).

### 2.8 LegalBench-RAG (exam candidate)

- Source: Pipitone and Alami 2024, arXiv 2408.10343 (READ, Tables 1 to 7, Sections 3 to 5); repository `zeroentropy-ai/legalbenchrag` (READ).
- Corpus (Table 2): 4 sub-corpora, 714 documents, 79,704,214 characters, 6,889 queries.
  ContractNLI: 95 docs, 1,013,969 chars, 977 queries.
  MAUD: 150 docs, 52,721,337 chars, 1,676 queries.
  CUAD: 462 docs, 25,792,044 chars, 4,042 queries.
  PrivacyQA: 7 docs, 176,864 chars, 194 queries.
  The paper's Table 1 gives 6,858 total (ContractNLI 946) and Table 2 gives 6,889 (977): an internal inconsistency of 31 queries.
- Mini version (Table 3): 72 documents (18, 18, 29, 7), 8,682,104 characters, 776 queries (194 per sub-corpus).
  All published baseline experiments use only the mini version (Section 4).
- Unit: character spans over raw document text; gold is a list of relevant spans (character index ranges) per query.
- Evidence per query: spans per query are not tabulated; the paper states that each query is answered by exactly one document, so this is a single-document benchmark, and that "several queries" need multi-hop reasoning (Section 3.3).
- Queries are derived from LegalBench annotations (clauses) and their context spans (Section 3.1).
- Retrieval setting: the paper evaluates each sub-corpus separately and averages the four with equal weight (Section 4.2 "we weight the metrics equally on each dataset"); the retrieval is a pooled search over that sub-corpus's documents (my reading of Section 2 to 4; the repository README does not state it, so mark as INFERRED).
- Metrics: character-level Precision@k and Recall@k, k from 1 to 64: precision is the share of retrieved characters inside gold spans, recall is the share of gold characters covered by the retrieved chunks (repository README, "precision and recall even at the exact character level", READ; the paper's formal definition is in a part I did not locate).
- Baseline setup (Section 4.2): text-embedding-3-large, SQLite Vec, Cohere rerank-english-v3.0; chunking either naive fixed size 500 characters without overlap or a Recursive Character Text Splitter (RCTS).
- Baseline figures, all-datasets row (equal weight over the four), percent:
  - RCTS without reranker (Table 5): Precision@1 6.41, @2 6.16, @4 5.76, @8 4.36, @64 1.45; Recall@1 4.94, @4 16.90, @16 37.06, @64 62.22.
  - RCTS with Cohere reranker (Table 7): Precision@1 6.13; Recall@64 61.06.
  - Per sub-corpus, RCTS without reranker, Recall@64: PrivacyQA 84.19, ContractNLI 61.72, MAUD 28.28, CUAD 74.70 (Table 5).
  - I rechecked Recall@64 as the mean of the four sub-corpus values: 62.2225 for Table 5 and 61.0575 for Table 7, both matching.
  - Anomalies: the all-datasets row of Table 4 (naive chunking) gives Recall@64 76.39 while the mean of its four sub-corpus values is 63.49, so that row is inconsistent; Table 6 (naive plus Cohere) reproduces Table 5 nearly digit for digit, which looks like a copy error.
  The old survey quoted "P@1 6.41 and R@64 62.22" and these match Table 5.
- Open-source follow-up: arXiv 2508.13107 (READ via fetch summary) reports on the mini version that RCTS + SBERT (all-mpnet-base-v2) + cosine is best, improving Recall@K by 30 to 95% and Precision@K by about 2.5 times for K>4 over the original baseline; figures are in plots (Figure 3, Appendix C Figure 7), not in a table I could read.
- Vendor claim: Ragie (blog, 2024-12-06, SNIPPET-level read, self-reported) reports precision@1 53.675 against 6.41 and recall 99.4% at k=64; the setting and the gold used are not verified, so it is context only.
- Licence: repository code MIT; the data derive from ContractNLI, CUAD, MAUD and PrivacyQA, and the repository requires accepting each source's usage terms to regenerate (README, READ); the old survey records CC BY 4.0, which I did not re-verify (so cite the repository for code and per-source terms for data).
- Leaderboard: none in the repository.

### 2.9 Briefly: FRAMES, BRIGHT, BrowseComp-Plus and 2025-2026 multi-evidence retrieval benchmarks

All figures here are INHERITED from `corpora_survey_2026-09-30.md` unless marked, and none was re-read this session.
- FRAMES (arXiv 2409.12941): 824 questions, 2 to 15 gold Wikipedia articles, CC BY 4.0, gold is whole articles, main metric is end-to-end accuracy; article-level gold is too coarse for paragraph Full Support, so it matters only as a comparability note.
- BRIGHT (arXiv 2407.12883, ICLR 2025): 1,398 queries over 12 tasks, nDCG@10; the ICLR abstract (SNIPPET, 2026-10-02 search) says the best model scores 22.1 and the MTEB leader scores 18.0 on it, while the old survey quotes 24.3 in the paper and 46.8 for DIVER; these conflict, so cite none until reread.
  Relevant documents are mostly few per query, so it is a poor fit for Full Support.
- BrowseComp-Plus (arXiv 2508.06600): 100,195 documents, 830 queries, 2.9 gold and 6.1 evidence documents per query; Recall@5/100/1000 for BM25 1.2/4.7/13.7 and Qwen3-Embedding-8B 14.5/47.7/76.7; document gold only; relevant to search-agent comparison and to large-K recall, not to paragraph Full Support.
- QUEST (MRecall@K, share of queries with all gold documents in top K) and QAMPARI (ERecall@K over evidence passages): these are the community's closest "all gold" metrics (old survey).
- I did not survey 2026 search-agent benchmarks beyond the above (gap).

## 3. Metrics definition table

"Computable" means computable from depth-100 ranked lists of whole units with gold labels at unit level.
FS means Full Support @2,048 tokens as defined in the handover glossary.
Metrics that live on a different unit (chunks, characters, sentences) need a mapping and are never numerically comparable with published figures.

| Metric | Definition | Unit | Setting where reported | Computable from our depth-100 rankings | Relation to Full Support |
|---|---|---|---|---|---|
| Full Support @B tokens (ours) | share of questions whose every gold unit fits in a B-token context built from the ranking | whole paragraph | ours | yes (it is the metric) | itself |
| FS@k units (ours) | every gold unit in first k | whole paragraph | ours | yes | FS with a rank cut instead of a token cut |
| MDR R@k (HotpotQA, Xiong 2021 Table 1) | recall at top k paragraphs; its R@2 of 65.9 equals the paper's SP EM of 65.9 for direct MDR, so R@k appears to be both-gold-in-top-k (INFERRED from the two tables) | paragraph | FullWiki, 5.2M corpus | yes, if the ranking is paragraphs | for 2-gold questions it equals FS@k units; differs from FS by cutting on rank not tokens |
| Paragraph EM (P-EM) in MDR code | 1 if all gold titles are in retrieved titles (code of `eval_mhop_retrieval.py`, READ via fetch) | paragraph | HotpotQA | yes | equals FS@k units |
| Paragraph recall (PR) in MDR code | 1 if any gold title is retrieved (same file) | paragraph | HotpotQA | yes | weaker than FS (any instead of all); gives an upper bound on FS |
| Support Passage EM (SP EM) | whether the final selected pair covers both gold passages (MDR paper footnote 5) | paragraph, after reranking | HotpotQA FullWiki | yes, as FS@2 on a reranked list | equals FS@2 units |
| Passage recall@k (HippoRAG family) | share of gold passages in top k (average over queries; definition not stated in the text read, per-query share assumed, NOT verified) | passage | pooled 1,000-question corpora of 6k to 12k passages | yes | per-query average of gold share, so a soft version of FS; upper bounds FS@k |
| Recall@k (BEIR) | share of relevant documents in top k, averaged; Recall@100 in Table 9 | abstract/document | full pooled corpus | yes | soft version of FS@100 units |
| nDCG@10 (BEIR) | normalised DCG over the first 10 with binary relevance | abstract/document | BEIR-HotpotQA 5.2M | yes | rank-quality; not a co-presence measure, can be high while a gold paragraph is missing |
| MRR@k | mean of reciprocal rank of first relevant item in top k | chunk (MultiHop-RAG) | MultiHop-RAG | yes | measures only the first gold; unrelated to FS |
| MAP@k | mean over queries of average precision at each rank where gold appears (truncated at k) | chunk | MultiHop-RAG | yes | rank-weighted precision of all gold; weakly related |
| Hits@k, original (MultiHop-RAG) | fraction of the evidence found in the top k | chunk | MultiHop-RAG | yes | per-query gold share; soft FS |
| Hits@k, Ammann 2025 | share of questions with at least one gold evidence in top k | chunk | MultiHop-RAG eval split | yes | equals "any gold", upper bound on FS |
| Precision@k | share of retrieved items that are gold | chunk or paragraph | MultiHop-RAG, LegalBench-RAG | yes | tells density of the context; FS ignores noise inside the budget |
| Character precision/recall @k (LegalBench-RAG) | precision: share of retrieved characters inside gold spans; recall: share of gold characters covered (README, paper definition not located) | character span; chunks of 500 chars | LegalBench-RAG-mini per sub-corpus | only after mapping spans to paragraphs; numbers would not be comparable (our units are paragraphs, far longer than 500-char chunks) | for single-evidence queries, recall at 100% of characters is close to FS; units differ |
| Evidence-F1 (QASPER) | F1 of the predicted set of paragraphs, figures and tables against the reference set, max over references | paragraph (and float) | within one paper | partly: needs a predicted set (threshold or top-k) and within-paper retrieval; text-only questions only | set-overlap measure; FS asks for all gold in a budget |
| Support F1 (HotpotQA, MuSiQue) | F1 of predicted supporting sentences (HotpotQA) or paragraphs (MuSiQue) vs gold | sentence or paragraph | distractor or Ans/Full | MuSiQue paragraph level yes if a set is chosen; HotpotQA sentence level no | like Evidence-F1 |
| Joint EM/F1 (HotpotQA) | answer and supporting facts both correct; joint P and R are products | answer plus sentences | distractor and FullWiki | no (needs a reader) | not retrieval |
| Answer EM/F1 | token overlap with the gold answer | answer string | all QA | no (needs a reader) | reader quality; FS is its necessary retrieval condition only loosely |
| LLM-judge accuracy (MultiHop-RAG Table 6) | accuracy of an LLM answer against the gold answer with retrieved or ground-truth chunks | answer | MultiHop-RAG | no | the end-to-end result FS is meant to predict; setting there is top-6 chunks of 256 tokens within 2,048 tokens |
| MRecall@K (QUEST, INHERITED) | share of queries with all gold documents in top K | document | QUEST 325k entities | yes, if units are documents | the literature's FS at a rank cut |

## 4. How the metrics relate to Full Support

- FS at a token budget is the all-gold criterion with a budget cut, so FS@k units and MDR's paragraph EM are the same family; they differ only in what limits the list (rank or tokens).
- Any-gold and per-query share metrics (Hits@k of Ammann 2025, PR, recall@k of HippoRAG, Recall@100 of BEIR) are weaker than FS: for the same ranking and cut, FS is at most the any-gold rate and at most the mean gold share.
- Rank-position metrics (nDCG@10, MRR, MAP) do not test whether all gold are present; two systems with equal nDCG@10 can differ widely in FS.
- Metrics needing a reader (answer EM/F1, joint, LLM-judge accuracy) cannot be computed from rankings.
- Because every metric above is a function of the gold-unit rank positions, one depth-100 list per question yields all the unit-level metrics at once; the only extra input is the token length of each unit for the token cut.
- Reporting rule (consistent with spec C4): report FS beside the literature metric recomputed on the same units and corpus, and compare with a published number only when unit, definition and corpus setting match.

## 5. Exam candidate comparison

| Aspect | QASPER | LegalBench-RAG |
|---|---|---|
| Domain | NLP papers | contracts, M&A agreements, privacy policies |
| Corpus (full) | 1,585 papers; paragraph count not read | 714 docs, 79.7M characters |
| Size used by published baselines | whole-paper reading or retrieval inside one paper | mini: 72 docs, 8.7M characters, 776 queries |
| Questions | 5,049 (test about 1,451 over 416 papers, derived) | 6,889 (mini 776) |
| Evidence per question | 55.5% of text-evidence answerable questions need more than one paragraph; 13% need tables or figures | single document by construction; spans per query not tabulated |
| Gold unit | paragraph, native | character span, mapped losslessly onto paragraphs |
| Published setting | within one paper | pooled per sub-corpus, four sub-corpora averaged equally |
| Published baselines | Evidence-F1: LED-large 39.37 test, TF-IDF 9.20 | RCTS + text-embedding-3-large: P@1 6.41, R@64 62.22; naive and Cohere variants in Tables 4 to 7 |
| Metric match with FS | Evidence-F1 needs a set; pooled retrieval is a new setting | character recall at k chunks; our unit is a paragraph |
| Licence | CC BY 4.0 | code MIT; data under the four source datasets' terms |
| Multi-evidence fit for Full Support | good (multi-paragraph questions) | poor (mostly one evidence span, so FS degenerates to recall of a single gold) |
| Leaderboard | answer metrics only found | none |

What makes a whole-corpus result comparable with published figures:
- LegalBench-RAG is the only one of the two whose published baselines are a pooled retrieval over many documents, so a whole-sub-corpus retrieval is the same setting as the paper, but only for characters and 500-character chunks, so comparison requires re-chunking, which defeats the whole-paragraph unit.
- QASPER's published retrieval is inside one paper; a whole-corpus (pooled) result is a new setting without a published counterpart, and no figure from the literature could be compared with it.
- A within-paper control on QASPER, run with the same harness, would be comparable in setting to the published Evidence-F1 only after choosing a set rule; this was not verified.

Recommendation: use QASPER as the exam corpus.
Reason: Full Support is a multi-evidence metric and QASPER has native paragraph gold with a majority of multi-paragraph questions, a clean CC BY 4.0 licence and its own train, validation and test split by paper, whereas LegalBench-RAG questions are single-document with span gold at a 500-character scale, so Full Support there collapses to single-evidence recall and its published numbers could be matched only by giving up whole paragraphs.
Declare the pooled-across-papers setting as new, and add a within-paper control so that at least one published setting (Evidence-F1 style) is reachable; consider LegalBench-RAG-mini (72 documents, 776 queries) only as an optional secondary check of domain transfer if comparison with Table 5 of its paper is wanted, recorded with the caveats above.
Before the exam is frozen, measure the QASPER test paragraph count and the share of test questions with at least two text paragraphs, which I did not.

## 6. Gaps

- Current leaderboard tops: MuSiQue (host not reachable), QASPER evidence retrieval, MultiHop-RAG, BRIGHT, FRAMES, BrowseComp-Plus, MTEB HotpotQA; only HotpotQA and the 2Wiki README were read.
- LegalBench-RAG paper's formal precision and recall definitions and spans-per-query statistics.
- QASPER mean evidence paragraphs per question and the paragraph count of the corpus.
- HippoRAG's exact recall@5 definition (all-gold or per-query share).
- MDR R@k equals P-EM only by inference from two tables, not from a stated definition.
- Ammann et al. 2025 "eval split" size for MultiHop-RAG.
- 2025 to 2026 search-agent benchmarks beyond those the old survey named.
