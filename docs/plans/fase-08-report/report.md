<!--
Phase 08 report. Prose is hand-written; every table between BEGIN GENERATED and END GENERATED
is written by code from tournament.json and must not be edited by hand.
Regenerate the export and the tables: uv run edge-rag report
Check only (writes nothing): uv run edge-rag report --check
Checker rules (spec C3, C4): outside generated blocks, comments and inline code, every figure
(single digits and zero-padded phase numbers excepted) must equal a value in tournament.json at
the precision written, or sit on a line carrying a pointer of the form [src: committed/path.md]
whose page contains that figure; pointers into data/ are refused. A line with figures must also
carry one of the labels measured, derived, exploratory, interpretation or projection.
-->

# The tournament: report

Status: final write-up of the project (Phase 08); the project closes with it.
Machine-readable figures: `tournament.json` next to this page, with the SHA-256 of every phase run file it was built from.

## Question and answer

The question: does a cheap, fast retrieval strategy sit on the quality-cost frontier against the literature's best recipes (the ghosts) on the terrain, and keep its place on a locked exam corpus?

The answer, as the outcome code wrote it, without promotion:
- No strategy designed in this project (Phases 03-05) advanced to the exam; every verdict is in the first table below.
- On the terrain, the project's best own system wins 8 of 9 set-class cells against the strongest measured ghost and loses MuSiQue class A to G-A1 (Search-R1) (measured, Phase 06).
- The HotpotQA wins are an upper bound (interpretation): the P10-C and P14 weights are in-sample there and BGE-small, inside the fused own systems, was fine-tuned on HotpotQA train.
- The class A bar stands on G-A1 alone: G-A2 (HippoRAG 2) was not run (cost gate).
- On the exam (QASPER), class L `rrf4` loses to G-L; classes R and A `j-rrf4` tie G-R, which is not losing and not a win (measured, Phase 07).
- QASPER has few multi-evidence questions, so the exam says little about the multi-evidence case the project targets.

So no cheap strategy is shown to beat the literature outside the terrain: the rerank-class system holds level with its ghost on the exam, and the light-class system falls behind its ghost there.

<!-- BEGIN GENERATED: verdicts -->
| Phase | Field | Value | Label |
|---|---|---|---|
| 03 | exam_entrant | none from this phase | written by code under the frozen selection rule |
| 03 | f3_verdict | advances, no entrant (no win against RRF3; a loss to RRF3; a loss to the best light-class system so far) | written by code under the frozen selection rule |
| 03 | rrf3_context_verdict | advances | written by code under the frozen selection rule |
| 04 | exam_entrant | none from this phase | written by code under the frozen selection rule |
| 04 | j-rrf3_context_verdict | advances | written by code under the frozen selection rule |
| 04 | verdict | advances, no entrant (a loss to j-rrf3) | written by code under the frozen selection rule |
| 05 | exam_entrant | none from this phase | written by code under the frozen selection rule |
| 05 | rrf-prf_context_verdict | does not advance | written by code under the frozen selection rule |
| 05 | verdict | does not advance | written by code under the frozen selection rule |
| 06 | beats_the_literature | ['class L on hotpotqa-dev', 'class R on hotpotqa-dev', 'class A on hotpotqa-dev', 'class L on multihop-rag', 'class R on multihop-rag', 'class A on multihop-rag', 'class L on musique', 'class R on musique'] | written by code under the frozen rule (spec, comparisons and the bar) |
<!-- END GENERATED -->

## Setup

- Terrain: HotpotQA FullWiki dev, MuSiQue and MultiHop-RAG, with 7,405 / 2,417 / 2,255 questions (measured) [src: docs/plans/fase-02-ghosts/spec.md].
- Exam: QASPER test, pooled across papers, opened once under a rule frozen before the download (Phase 07 spec).
- Metric of record: Full Support at the reading budget of record (FS@2,048), the count of questions whose every gold evidence unit is inside that token budget of the ranking; the other metrics each phase measured are on the phase pages, and no common column is invented where a phase did not measure it.
- Cost classes (master plan decisions table): L light, no generative LLM and models up to 1B; R rerank, L plus zero-shot cross-encoders over a bounded top list, no text generation; A LLM, open models up to 8B on one GPU in the online loop.
- Ghosts reproduced on this harness (Phases 02 and 06): G-L (answerai-colbert-small-v1 with a PLAID index), G-R (bge-reranker-v2-m3 over the top list of G-L), G-R2 (Qwen3-Reranker-0.6B), G-A1 (Search-R1) and G-A2 (HippoRAG 2, not run).
- The old project's systems (`p10-a`, `p10-b`, `p10-c`, `p14`, `j-p10b`, `j-union`) appear labelled context; `p10-b` and `p14` are the best light-class own systems on two sets.
- Every comparison is paired on the same questions with an exact test; the states win, loss and tie are written by each phase's outcome code under its frozen rule.
- Laptop cost counts 0 USD by assumption: energy is not counted.

The verdict states of every comparison, per set:

<!-- BEGIN GENERATED: states -->
| Phase | Comparison | hotpotqa-dev | multihop-rag | musique |
|---|---|---|---|---|
| 03 | f3 vs best light-class so far | loss | loss | loss |
| 03 | f3 vs best so far | loss | loss | loss |
| 03 | f3 vs g-l | win | win | tie |
| 03 | f3 vs rrf3 | tie | loss | loss |
| 03 | rrf3 vs best light-class so far | loss | tie | loss |
| 03 | rrf3 vs best so far | loss | loss | loss |
| 03 | rrf3 vs g-l | win | win | win |
| 04 | j-rrf3 vs best so far | loss | win | loss |
| 04 | j-rrf3 vs g-r | tie | win | win |
| 04 | j-rrf4 vs best in bound | win | tie | win |
| 04 | j-rrf4 vs best so far | tie | tie | loss |
| 04 | j-rrf4 vs f3 | win | win | win |
| 04 | j-rrf4 vs g-r | win | win | win |
| 04 | j-rrf4 vs j-rrf3 | win | loss | win |
| 04 | j-rrf4 vs j-union | tie | tie | win |
| 05 | mch vs best light-class so far | loss | loss | loss |
| 05 | mch vs best so far | loss | loss | loss |
| 05 | mch vs g-l | loss | win | tie |
| 05 | mch vs rrf-1s | loss | loss | loss |
| 05 | mch vs rrf-prf | win | loss | win |
| 05 | rrf-prf vs best so far | loss | loss | loss |
| 05 | rrf-prf vs g-l | loss | win | tie |
| 06 | g-a1 vs best own | loss | not run | not run |
| 06 | g-a1 vs best so far | loss | not run | not run |
| 06 | g-a1 vs g-a2 | not run | not run | not run |
| 06 | g-a2 vs best own | not run | not run | not run |
| 06 | g-a2 vs best so far | not run | not run | not run |
| 06 | g-a2 vs g-a1 | not run | not run | not run |
| 06 | g-r2 vs best own | loss | loss | loss |
| 06 | g-r2 vs best so far | loss | loss | loss |
| 06 | g-r2 vs g-r | loss | tie | tie |
<!-- END GENERATED -->

FS@2,048 of record per phase and system:

<!-- BEGIN GENERATED: fs -->
| Phase | System | Role | hotpotqa-dev | multihop-rag | musique | Label |
|---|---|---|---|---|---|---|
| 02 | g-a1 | ghost | - | 557 | 1101 | measured |
| 02 | g-l | ghost | 5000 | 187 | 533 | measured |
| 02 | g-r | ghost | 5652 | 425 | 661 | measured |
| 02 | j-p10b | context (old project) | 5198 | 822 | 704 | measured |
| 02 | j-union | context (old project) | 5922 | 846 | 809 | measured |
| 02 | p10-a | context (old project) | 4125 | 338 | 436 | measured |
| 02 | p10-b | context (old project) | 4536 | 587 | 524 | measured |
| 02 | p10-c | context (old project) | 4801 | 522 | 669 | measured |
| 02 | p14 | context (old project) | 5224 | 513 | 761 | measured |
| 03 | f3 | own | 5080 | 490 | 537 | measured |
| 03 | g-a1 | ghost | - | - | 1101 | measured |
| 03 | g-l | ghost | 5000 | 187 | 533 | measured |
| 03 | j-union | context (old project) | 5922 | 846 | - | measured |
| 03 | p10-b | context (old project) | - | 587 | - | measured |
| 03 | p14 | context (old project) | 5224 | - | 761 | measured |
| 03 | rrf3 | own | 5083 | 556 | 594 | measured |
| 04 | f3 | own | 5080 | 490 | 537 | measured |
| 04 | g-a1 | ghost | - | - | 1101 | measured |
| 04 | g-r | ghost | 5652 | 425 | 661 | measured |
| 04 | j-p10b | context (old project) | - | 822 | 704 | measured |
| 04 | j-rrf3 | own | 5673 | 873 | 737 | measured |
| 04 | j-rrf4 | own | 5950 | 843 | 831 | measured |
| 04 | j-union | context (old project) | 5922 | 846 | 809 | measured |
| 04 | rrf4 | own | 5291 | 485 | 665 | measured |
| 05 | dense-prf | own | 4149 | 266 | 445 | measured |
| 05 | g-a1 | ghost | - | - | 1101 | measured |
| 05 | g-l | ghost | 5000 | 187 | 533 | measured |
| 05 | hop-ms | own | 10 | 15 | 8 | measured |
| 05 | j-rrf3 | own | - | 873 | - | measured |
| 05 | j-rrf4 | own | 5950 | - | - | measured |
| 05 | mch | own | 4528 | 366 | 528 | measured |
| 05 | p10-b | context (old project) | - | 587 | - | measured |
| 05 | p14 | context (old project) | - | - | 761 | measured |
| 05 | rrf-1s | own | 4717 | 419 | 565 | measured |
| 05 | rrf-prf | own | 4481 | 460 | 502 | measured |
| 05 | rrf4 | own | 5291 | - | - | measured |
| 06 | g-a1 | ghost | 746 (of 1000) | 557 | 1101 | measured |
| 06 | g-l | ghost | 5000 | 187 | 533 | measured |
| 06 | g-r | ghost | 5652 | 425 | 661 | measured |
| 06 | g-r2 | ghost | 715 (of 1000) | 438 | 666 | measured |
| 06 | j-rrf3 | own | - | 873 | - | measured |
| 06 | j-rrf4 | own | 5950 | - | 831 | measured |
| 06 | p10-b | context (old project) | - | 587 | - | measured |
| 06 | p14 | context (old project) | - | - | 761 | measured |
| 06 | rrf4 | own | 5291 | - | - | measured |
<!-- END GENERATED -->

The paired tests behind the states, as each phase recorded them:

<!-- BEGIN GENERATED: paired -->
| Phase | Set | System | Against | Wins | Losses | Ties | Exact p | State | On or reason | Label |
|---|---|---|---|---|---|---|---|---|---|---|
| 02 | hotpotqa-dev | g-l | j-union | 188 | 1110 | 6107 | 2.1e-159 | - | - | measured |
| 02 | hotpotqa-dev | g-r | j-union | 294 | 564 | 6547 | 2.03e-20 | - | - | measured |
| 02 | hotpotqa-dev | g-r | g-l | 719 | 67 | 6619 | 8.08e-139 | - | - | measured |
| 02 | multihop-rag | g-l | j-union | 32 | 691 | 1532 | 2.8e-162 | - | - | measured |
| 02 | multihop-rag | g-r | j-union | 49 | 470 | 1736 | 2.27e-87 | - | - | measured |
| 02 | multihop-rag | g-r | g-l | 264 | 26 | 1965 | 9.16e-51 | - | - | measured |
| 02 | multihop-rag | g-a1 | j-union | 110 | 399 | 1746 | 1.68e-39 | - | - | measured |
| 02 | multihop-rag | g-a1 | g-l | 448 | 78 | 1729 | 4.11e-64 | - | - | measured |
| 02 | multihop-rag | g-a1 | g-r | 337 | 205 | 1713 | 1.57e-08 | - | - | measured |
| 02 | multihop-rag | g-a1 | g-l (searched index) | 447 | 77 | 1731 | 2.07e-64 | - | - | measured |
| 02 | musique | g-l | j-union | 61 | 337 | 2019 | 2.28e-47 | - | - | measured |
| 02 | musique | g-r | j-union | 52 | 200 | 2165 | 1.19e-21 | - | - | measured |
| 02 | musique | g-r | g-l | 203 | 75 | 2139 | 8.62e-15 | - | - | measured |
| 02 | musique | g-a1 | j-union | 524 | 232 | 1661 | 8.15e-27 | - | - | measured |
| 02 | musique | g-a1 | g-l | 682 | 114 | 1621 | 2.32e-99 | - | - | measured |
| 02 | musique | g-a1 | g-r | 631 | 191 | 1595 | 1.14e-55 | - | - | measured |
| 02 | musique | g-a1 | g-l (searched index) | 683 | 109 | 1625 | 2.38e-102 | - | - | measured |
| 03 | hotpotqa-dev | f3 | g-l | 390 | 310 | 6705 | 0.0028 | win | - | measured |
| 03 | hotpotqa-dev | rrf3 | g-l | 387 | 304 | 6714 | 0.00179 | win | - | measured |
| 03 | hotpotqa-dev | f3 | rrf3 | 3 | 6 | 7396 | 0.508 | tie | - | measured |
| 03 | hotpotqa-dev | f3 | p14 | 420 | 564 | 6421 | 4.96e-06 | loss | - | measured |
| 03 | hotpotqa-dev | rrf3 | p14 | 420 | 561 | 6424 | 7.57e-06 | loss | - | measured |
| 03 | hotpotqa-dev | f3 | j-union | 102 | 944 | 6359 | 1.87e-171 | loss | - | measured |
| 03 | hotpotqa-dev | rrf3 | j-union | 101 | 940 | 6364 | 3.87e-171 | loss | - | measured |
| 03 | multihop-rag | f3 | g-l | 355 | 52 | 1848 | 1.46e-56 | win | - | measured |
| 03 | multihop-rag | rrf3 | g-l | 384 | 15 | 1856 | 9.76e-94 | win | - | measured |
| 03 | multihop-rag | f3 | rrf3 | 144 | 210 | 1901 | 0.000533 | loss | - | measured |
| 03 | multihop-rag | f3 | p10-b | 140 | 237 | 1878 | 6.68e-07 | loss | - | measured |
| 03 | multihop-rag | rrf3 | p10-b | 107 | 138 | 2010 | 0.0551 | tie | - | measured |
| 03 | multihop-rag | f3 | j-union | 85 | 441 | 1729 | 5.84e-59 | loss | - | measured |
| 03 | multihop-rag | rrf3 | j-union | 75 | 365 | 1815 | 8e-47 | loss | - | measured |
| 03 | musique | f3 | g-l | 121 | 117 | 2179 | 0.846 | tie | - | measured |
| 03 | musique | rrf3 | g-l | 137 | 76 | 2204 | 3.51e-05 | win | - | measured |
| 03 | musique | f3 | rrf3 | 21 | 78 | 2318 | 6.88e-09 | loss | - | measured |
| 03 | musique | f3 | p14 | 90 | 314 | 2013 | 3.7e-30 | loss | - | measured |
| 03 | musique | rrf3 | p14 | 105 | 272 | 2040 | 3.43e-18 | loss | - | measured |
| 03 | musique | f3 | g-a1 | 107 | 671 | 1639 | 1.26e-100 | loss | - | measured |
| 03 | musique | rrf3 | g-a1 | 137 | 644 | 1636 | 2.41e-79 | loss | - | measured |
| 04 | hotpotqa-dev | j-rrf4 | g-r | 438 | 140 | 6827 | 1.07e-36 | win | - | measured |
| 04 | hotpotqa-dev | j-rrf4 | j-rrf3 | 331 | 54 | 7020 | 1.09e-49 | win | - | measured |
| 04 | hotpotqa-dev | j-rrf4 | g-r | 438 | 140 | 6827 | 1.07e-36 | win | - | measured |
| 04 | hotpotqa-dev | j-rrf4 | j-union | 244 | 216 | 6945 | 0.208 | tie | - | measured |
| 04 | hotpotqa-dev | j-rrf4 | j-union | 244 | 216 | 6945 | 0.208 | tie | - | measured |
| 04 | hotpotqa-dev | j-rrf4 | f3 | 928 | 58 | 6419 | 1.11e-202 | win | - | measured |
| 04 | hotpotqa-dev | j-rrf3 | g-r | 189 | 168 | 7048 | 0.29 | tie | - | measured |
| 04 | hotpotqa-dev | j-rrf3 | j-union | 210 | 459 | 6736 | 2.94e-22 | loss | - | measured |
| 04 | multihop-rag | j-rrf4 | g-r | 453 | 35 | 1767 | 9.36e-94 | win | - | measured |
| 04 | multihop-rag | j-rrf4 | j-rrf3 | 21 | 51 | 2183 | 0.000535 | loss | - | measured |
| 04 | multihop-rag | j-rrf4 | j-p10b | 85 | 64 | 2106 | 0.101 | tie | - | measured |
| 04 | multihop-rag | j-rrf4 | j-union | 59 | 62 | 2134 | 0.856 | tie | - | measured |
| 04 | multihop-rag | j-rrf4 | j-union | 59 | 62 | 2134 | 0.856 | tie | - | measured |
| 04 | multihop-rag | j-rrf4 | f3 | 438 | 85 | 1732 | 2.75e-58 | win | - | measured |
| 04 | multihop-rag | j-rrf3 | g-r | 475 | 27 | 1753 | 6.06e-107 | win | - | measured |
| 04 | multihop-rag | j-rrf3 | j-union | 69 | 42 | 2144 | 0.0132 | win | - | measured |
| 04 | musique | j-rrf4 | g-r | 197 | 27 | 2193 | 4.41e-33 | win | - | measured |
| 04 | musique | j-rrf4 | j-rrf3 | 109 | 15 | 2293 | 8.67e-19 | win | - | measured |
| 04 | musique | j-rrf4 | j-p10b | 183 | 56 | 2178 | 6.38e-17 | win | - | measured |
| 04 | musique | j-rrf4 | j-union | 66 | 44 | 2307 | 0.0448 | win | - | measured |
| 04 | musique | j-rrf4 | g-a1 | 238 | 508 | 1671 | 2.29e-23 | loss | - | measured |
| 04 | musique | j-rrf4 | f3 | 363 | 69 | 1985 | 2.94e-49 | win | - | measured |
| 04 | musique | j-rrf3 | g-r | 106 | 30 | 2281 | 3.82e-11 | win | - | measured |
| 04 | musique | j-rrf3 | g-a1 | 220 | 584 | 1613 | 7.65e-39 | loss | - | measured |
| 05 | hotpotqa-dev | mch | g-l | 366 | 838 | 6201 | 4.66e-43 | loss | - | measured |
| 05 | hotpotqa-dev | mch | rrf-prf | 183 | 136 | 7086 | 0.0099 | win | - | measured |
| 05 | hotpotqa-dev | mch | rrf-1s | 73 | 262 | 7070 | 3.9e-26 | loss | - | measured |
| 05 | hotpotqa-dev | mch | rrf4 | 90 | 853 | 6462 | 1.28e-156 | loss | - | measured |
| 05 | hotpotqa-dev | mch | j-rrf4 | 64 | 1486 | 5855 | 0 | loss | - | measured |
| 05 | hotpotqa-dev | rrf-prf | g-l | 334 | 853 | 6218 | 8.83e-53 | loss | - | measured |
| 05 | hotpotqa-dev | rrf-prf | j-rrf4 | 57 | 1526 | 5822 | 0 | loss | - | measured |
| 05 | multihop-rag | mch | g-l | 250 | 71 | 1934 | 1.61e-24 | win | - | measured |
| 05 | multihop-rag | mch | rrf-prf | 27 | 121 | 2107 | 2.08e-15 | loss | - | measured |
| 05 | multihop-rag | mch | rrf-1s | 25 | 78 | 2152 | 1.61e-07 | loss | - | measured |
| 05 | multihop-rag | mch | p10-b | 31 | 252 | 1972 | 3.28e-44 | loss | - | measured |
| 05 | multihop-rag | mch | j-rrf3 | 39 | 546 | 1670 | 1.89e-115 | loss | - | measured |
| 05 | multihop-rag | rrf-prf | g-l | 328 | 55 | 1872 | 1.94e-48 | win | - | measured |
| 05 | multihop-rag | rrf-prf | j-rrf3 | 50 | 463 | 1742 | 7.46e-85 | loss | - | measured |
| 05 | musique | mch | g-l | 135 | 140 | 2142 | 0.809 | tie | - | measured |
| 05 | musique | mch | rrf-prf | 90 | 64 | 2263 | 0.0436 | win | - | measured |
| 05 | musique | mch | rrf-1s | 48 | 85 | 2284 | 0.00169 | loss | - | measured |
| 05 | musique | mch | p14 | 80 | 313 | 2024 | 1.16e-33 | loss | - | measured |
| 05 | musique | mch | g-a1 | 131 | 704 | 1582 | 1.46e-95 | loss | - | measured |
| 05 | musique | rrf-prf | g-l | 105 | 136 | 2176 | 0.0531 | tie | - | measured |
| 05 | musique | rrf-prf | g-a1 | 113 | 712 | 1592 | 5.53e-107 | loss | - | measured |
| 06 | hotpotqa-dev | g-r2 | j-rrf4 | 20 | 98 | 882 | 1.53e-13 | loss | the 1,000 preregistered qids (D11) | measured; state written by code |
| 06 | hotpotqa-dev | g-r2 | j-rrf4 | 20 | 98 | 882 | 1.53e-13 | loss | the 1,000 preregistered qids (D11) | measured; state written by code |
| 06 | hotpotqa-dev | g-r2 | g-r | 11 | 54 | 935 | 6.03e-08 | loss | the 1,000 preregistered qids (D11) | measured; state written by code |
| 06 | hotpotqa-dev | g-a1 | j-rrf4 | 111 | 158 | 731 | 0.00494 | loss | the 1,000 preregistered qids (D11) | measured; state written by code |
| 06 | hotpotqa-dev | g-a1 | j-rrf4 | 111 | 158 | 731 | 0.00494 | loss | the 1,000 preregistered qids (D11) | measured; state written by code |
| 06 | hotpotqa-dev | g-a1 | g-a2 | - | - | - | - | not run | g-a2 is outside the spec's scope on this set | - |
| 06 | hotpotqa-dev | g-a2 | j-rrf4 | - | - | - | - | not run | g-a2 is outside the spec's scope on this set | - |
| 06 | hotpotqa-dev | g-a2 | j-rrf4 | - | - | - | - | not run | g-a2 is outside the spec's scope on this set | - |
| 06 | hotpotqa-dev | g-a2 | g-a1 | - | - | - | - | not run | g-a2 is outside the spec's scope on this set | - |
| 06 | multihop-rag | g-r2 | j-rrf3 | 80 | 515 | 1660 | 8.93e-79 | loss | full set | measured; state written by code |
| 06 | multihop-rag | g-r2 | j-rrf3 | 80 | 515 | 1660 | 8.93e-79 | loss | full set | measured; state written by code |
| 06 | multihop-rag | g-r2 | g-r | 67 | 54 | 2134 | 0.275 | tie | full set | measured; state written by code |
| 06 | multihop-rag | g-a1 | j-rrf3 | - | - | - | - | not run | g-a1 is outside the spec's scope on this set | - |
| 06 | multihop-rag | g-a1 | j-rrf3 | - | - | - | - | not run | g-a1 is outside the spec's scope on this set | - |
| 06 | multihop-rag | g-a1 | g-a2 | - | - | - | - | not run | g-a1 is outside the spec's scope on this set | - |
| 06 | multihop-rag | g-a2 | j-rrf3 | - | - | - | - | not run | cost gate: the probe projected 7.10 USD against G-A2's 4.5 USD hard cut, so the session stopped before the full index (plan increment 4, deviation 06.3, D12) | - |
| 06 | multihop-rag | g-a2 | j-rrf3 | - | - | - | - | not run | cost gate: the probe projected 7.10 USD against G-A2's 4.5 USD hard cut, so the session stopped before the full index (plan increment 4, deviation 06.3, D12) | - |
| 06 | multihop-rag | g-a2 | g-a1 | - | - | - | - | not run | cost gate: the probe projected 7.10 USD against G-A2's 4.5 USD hard cut, so the session stopped before the full index (plan increment 4, deviation 06.3, D12) | - |
| 06 | musique | g-r2 | g-a1 | 173 | 608 | 1636 | 1.75e-57 | loss | full set | measured; state written by code |
| 06 | musique | g-r2 | j-rrf4 | 99 | 264 | 2054 | 1.88e-18 | loss | full set | measured; state written by code |
| 06 | musique | g-r2 | g-r | 119 | 114 | 2184 | 0.793 | tie | full set | measured; state written by code |
| 06 | musique | g-a1 | g-a1 | - | - | - | - | not run | g-a1 is itself the best system so far on this set; no self-comparison | - |
| 06 | musique | g-a1 | j-rrf4 | - | - | - | - | not run | g-a1 is outside the spec's scope on this set | - |
| 06 | musique | g-a1 | g-a2 | - | - | - | - | not run | g-a1 is outside the spec's scope on this set | - |
| 06 | musique | g-a2 | g-a1 | - | - | - | - | not run | g-a2 is outside the spec's scope on this set | - |
| 06 | musique | g-a2 | j-rrf4 | - | - | - | - | not run | g-a2 is outside the spec's scope on this set | - |
| 06 | musique | g-a2 | g-a1 | - | - | - | - | not run | g-a2 is outside the spec's scope on this set | - |
<!-- END GENERATED -->

## Candidate phases

### Phase 03 - weight-free fusion (class L)

The candidate `f3` (reciprocal rank fusion with a source-diversity pass) advances against G-L but has no win and two losses against plain RRF3, and loses to the best light-class system on all three sets, so it sends no entrant.
The light-class gate G-L is weaker than the old fitted fusions; the strongest cheap literature recipe, a convex combination, is represented only by those old fusions.
Money: no pod; no clean balance delta exists for the phase (deviation 03.1, interpretation) [src: docs/plans/fase-03-fusion/plan.md].

### Phase 04 - cross-encoder judge over a pooled bag (class R)

The candidate `j-rrf4` (the literature control `j-rrf3` plus the entity hop in the pool) wins against G-R on all three sets but loses to `j-rrf3` on MultiHop-RAG, so it advances with no entrant from the phase.
The phase rented one short RTX 4090 pod (its measured money is in the money table).

### Phase 05 - multi-seed entity hop (class L)

The candidate `mch` does not advance: it loses to G-L on HotpotQA, to `rrf-prf` on MultiHop-RAG and to the best light-class system so far on every set.
The working control `rrf-prf` is weaker than MDR, the strongest light-class literature recipe for multi-hop retrieval, which was not reproduced.
Money: no pod.

Cost cells of the candidates, with each phase's label:

<!-- BEGIN GENERATED: cost -->
| Phase | System | Set | Laptop online s/question | GPU online s/question | Inside class | Label |
|---|---|---|---|---|---|---|
| 03 | f3 | hotpotqa-dev | 0.26 | NVIDIA A100-SXM4-80GB: 0.358 | inside_light_class: no | derived |
| 03 | f3 | multihop-rag | 0.0545 | NVIDIA GeForce RTX 4090: 0.0118 | inside_light_class: yes | derived |
| 03 | f3 | musique | 0.0713 | NVIDIA GeForce RTX 4090: 0.0247 | inside_light_class: yes | derived |
| 03 | rrf3 | hotpotqa-dev | 0.221 | NVIDIA A100-SXM4-80GB: 0.358 | inside_light_class: no | derived |
| 03 | rrf3 | multihop-rag | 0.0431 | NVIDIA GeForce RTX 4090: 0.0118 | inside_light_class: yes | derived |
| 03 | rrf3 | musique | 0.025 | NVIDIA GeForce RTX 4090: 0.0247 | inside_light_class: yes | derived |
| 04 | j-rrf3 | hotpotqa-dev | 0.221 | NVIDIA A100-SXM4-80GB: 0.358; NVIDIA GeForce RTX 4090: 0.274 | inside_rerank_class: yes | derived; GPU seconds are checked per hardware, never summed across |
| 04 | j-rrf3 | multihop-rag | 0.0431 | NVIDIA GeForce RTX 4090: 0.353 | inside_rerank_class: yes | derived; GPU seconds are checked per hardware, never summed across |
| 04 | j-rrf3 | musique | 0.025 | NVIDIA GeForce RTX 4090: 0.346 | inside_rerank_class: yes | derived; GPU seconds are checked per hardware, never summed across |
| 04 | j-rrf4 | hotpotqa-dev | 0.246 | NVIDIA A100-SXM4-80GB: 0.358; NVIDIA GeForce RTX 4090: 0.274 | inside_rerank_class: yes | derived; GPU seconds are checked per hardware, never summed across |
| 04 | j-rrf4 | multihop-rag | 0.0436 | NVIDIA GeForce RTX 4090: 0.353 | inside_rerank_class: yes | derived; GPU seconds are checked per hardware, never summed across |
| 04 | j-rrf4 | musique | 0.0257 | NVIDIA GeForce RTX 4090: 0.346 | inside_rerank_class: yes | derived; GPU seconds are checked per hardware, never summed across |
| 05 | mch | hotpotqa-dev | 0.554 | - | inside_light_class: yes | derived; GPU and laptop seconds are checked apart, never summed |
| 05 | mch | multihop-rag | 0.0483 | - | inside_light_class: yes | derived; GPU and laptop seconds are checked apart, never summed |
| 05 | mch | musique | 0.0357 | - | inside_light_class: yes | derived; GPU and laptop seconds are checked apart, never summed |
| 05 | rrf-prf | hotpotqa-dev | 0.363 | - | inside_light_class: yes | derived; GPU and laptop seconds are checked apart, never summed |
| 05 | rrf-prf | multihop-rag | 0.0443 | - | inside_light_class: yes | derived; GPU and laptop seconds are checked apart, never summed |
| 05 | rrf-prf | musique | 0.0297 | - | inside_light_class: yes | derived; GPU and laptop seconds are checked apart, never summed |
| 06 | g-a1 | hotpotqa-dev | - | NVIDIA A100-SXM4-80GB: 1.1 | - | seconds measured (Phase 06 manifest); USD derived (time x rate) |
| 06 | g-a1 | multihop-rag | - | NVIDIA A100-SXM4-80GB: 0.258 | - | seconds measured (pod manifests); USD derived (time x rate) |
| 06 | g-a1 | musique | - | NVIDIA A100-SXM4-80GB: 0.377 | - | seconds measured (pod manifests); USD derived (time x rate) |
| 06 | g-r2 | hotpotqa-dev | - | NVIDIA GeForce RTX 4090: 1.25 | - | seconds measured (Phase 06 manifest); USD derived (time x rate) |
| 06 | g-r2 | multihop-rag | - | NVIDIA GeForce RTX 4090: 1.62 | - | seconds measured (Phase 06 manifest); USD derived (time x rate) |
| 06 | g-r2 | musique | - | NVIDIA GeForce RTX 4090: 1.63 | - | seconds measured (Phase 06 manifest); USD derived (time x rate) |
<!-- END GENERATED -->

Cost cells never recorded, kept as written:

<!-- BEGIN GENERATED: unrecorded -->
| Phase | Cost cell | Label as written |
|---|---|---|
| 03 | Dense corpus embeddings | not measured in this phase; handover: not recorded |
| 03 | Dense corpus embeddings | not measured in this phase; handover: BGE-small encoding 41.20 s on the old pod (measured (recorded time)); 0.0404 USD attributable to GLiNER + BGE, the BGE encoding inside it (derived (time x rate)); the session invoiced 0.165 USD (measured (invoice)); the session's balance delta 0.1029 USD (measured (balance delta)) |
| 04 | Dense corpus embeddings | inherited, not measured here: not recorded |
| 04 | GLiNER entity index | inherited: GLiNER over all 5,233,329 FullWiki paragraphs 6.18 h, 4.57 USD attributable (derived (time x rate); successor_project_handover.md, cost table (research_summary.md point 2)); the whole old Phase 9 session invoiced 12.71 USD, GLiNER and other work (measured (invoice); successor_project_handover.md, cost table (research_summary.md point 2)) |
| 04 | Dense corpus embeddings | inherited, not measured here: BGE-small encoding 41.20 s on the old pod (measured (recorded time); successor_project_handover.md, cost table (16.results.md section 3)); 0.0404 USD attributable to GLiNER + BGE, the BGE encoding inside it (derived (time x rate); successor_project_handover.md, cost table (16.results.md section 3)); the session invoiced 0.165 USD (measured (invoice); successor_project_handover.md, cost table (16.results.md section 3)); the session's balance delta 0.1029 USD (measured (balance delta); successor_project_handover.md, RunPod know-how) |
| 04 | GLiNER entity index | inherited: GLiNER 154.77 s extraction, BGE 41.20 s encoding, 0.0404 USD attributable (derived (time x rate); successor_project_handover.md, cost table (16.results.md section 3)); the session invoiced 0.165 USD (measured (invoice); successor_project_handover.md, cost table (16.results.md section 3)); the session's balance delta 0.1029 USD (measured (balance delta); successor_project_handover.md, RunPod know-how) |
| 04 | GLiNER entity index | inherited: GLiNER 10.3 min, 0.13 USD attributable (derived (time x rate); successor_project_handover.md, cost table (research_summary.md point 7)); the session invoiced 0.356 USD (measured (invoice); successor_project_handover.md, cost table (research_summary.md point 7)) |
| 05 | Dense corpus embeddings | inherited, not measured here: not recorded |
| 05 | GLiNER entity index | inherited: GLiNER over all 5,233,329 FullWiki paragraphs 6.18 h, 4.57 USD attributable (derived (time x rate); successor_project_handover.md, cost table (research_summary.md point 2)); the whole old Phase 9 session invoiced 12.71 USD, GLiNER and other work (measured (invoice); successor_project_handover.md, cost table (research_summary.md point 2)) |
| 05 | Dense corpus embeddings | inherited, not measured here: BGE-small encoding 41.20 s on the old pod (measured (recorded time); successor_project_handover.md, cost table (16.results.md section 3)); 0.0404 USD attributable to GLiNER + BGE, the BGE encoding inside it (derived (time x rate); successor_project_handover.md, cost table (16.results.md section 3)); the session invoiced 0.165 USD (measured (invoice); successor_project_handover.md, cost table (16.results.md section 3)); the session's balance delta 0.1029 USD (measured (balance delta); successor_project_handover.md, RunPod know-how) |
| 06 | Dense corpus embeddings | inherited, not measured here: not recorded |
| 06 | GLiNER entity index | inherited: GLiNER over all 5,233,329 FullWiki paragraphs 6.18 h, 4.57 USD attributable (derived (time x rate); successor_project_handover.md, cost table (research_summary.md point 2)); the whole old Phase 9 session invoiced 12.71 USD, GLiNER and other work (measured (invoice); successor_project_handover.md, cost table (research_summary.md point 2)) |
| 06 | Dense corpus embeddings | inherited, not measured here: BGE-small encoding 41.20 s on the old pod (measured (recorded time); successor_project_handover.md, cost table (16.results.md section 3)); 0.0404 USD attributable to GLiNER + BGE, the BGE encoding inside it (derived (time x rate); successor_project_handover.md, cost table (16.results.md section 3)); the session invoiced 0.165 USD (measured (invoice); successor_project_handover.md, cost table (16.results.md section 3)); the session's balance delta 0.1029 USD (measured (balance delta); successor_project_handover.md, RunPod know-how) |
| 06 | GLiNER entity index | inherited: GLiNER 10.3 min, 0.13 USD attributable (derived (time x rate); successor_project_handover.md, cost table (research_summary.md point 7)); the session invoiced 0.356 USD (measured (invoice); successor_project_handover.md, cost table (research_summary.md point 7)) |
| 07 | G-A1: G-L PLAID index the G-A1 retriever served on the A100 pod | not recorded (the g-a1 manifest times model load and run only) |
| 07 | G-R: first-stage fusion on the pod | not recorded (not timed apart in the g-r manifest (first-stage inputs as the manifest records them: bm25, dense, g-l, hop)) |
| 07 | G-R2: first-stage fusion on the pod | not recorded (not timed apart in the g-r2 manifest (first-stage inputs as the manifest records them: bm25, dense, g-l, hop)) |
| 07 | j-rrf3: first-stage fusion on the pod | not recorded (not timed apart in the j-rrf3 manifest (first-stage inputs as the manifest records them: bm25, dense, g-l, hop)) |
| 07 | j-rrf4: first-stage fusion on the pod | not recorded (not timed apart in the j-rrf4 manifest (first-stage inputs as the manifest records them: bm25, dense, g-l, hop)) |
| 07 | rrf4: rrf4 fusion on the laptop | not recorded (built by tools/assemble.py (deviation 07.4) with no timing) |
<!-- END GENERATED -->

## The literature bar (Phase 06)

Phase 06 ran the rivals Phase 02 had not: G-R2 on all three sets and G-A1 on the 1,000 preregistered HotpotQA qids (measured); G-A2 stayed not run (cost gate; the author decided no top-up).
The bar compares the best own system of each cell with the strongest measured ghost of its class:

<!-- BEGIN GENERATED: bar -->
| Set | Class | Own | Own FS | Strongest ghost | Ghosts FS | Wins | Losses | Ties | Exact p | State | On | Label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa-dev | L | rrf4 | 5291 | g-l | g-l: 5000 | 564 | 273 | 6568 | 4.16e-24 | win | full set | FS and paired test measured; strongest ghost and state written by code |
| hotpotqa-dev | R | j-rrf4 | 793 | g-r | g-r: 758; g-r2: 715 | 57 | 22 | 921 | 0.000103 | win | the 1,000 preregistered qids (D11) | FS and paired test measured; strongest ghost and state written by code |
| hotpotqa-dev | A | j-rrf4 | 793 | g-a1 | g-a1: 746; g-a2: - | 158 | 111 | 731 | 0.00494 | win | the 1,000 preregistered qids (D11) | FS and paired test measured; strongest ghost and state written by code |
| multihop-rag | L | p10-b | 587 | g-l | g-l: 187 | 442 | 42 | 1771 | 2.92e-85 | win | full set | FS and paired test measured; strongest ghost and state written by code |
| multihop-rag | R | j-rrf3 | 873 | g-r2 | g-r: 425; g-r2: 438 | 515 | 80 | 1660 | 8.93e-79 | win | full set | FS and paired test measured; strongest ghost and state written by code |
| multihop-rag | A | j-rrf3 | 873 | g-a1 | g-a1: 557; g-a2: - | 415 | 99 | 1741 | 5.19e-47 | win | full set | FS and paired test measured; strongest ghost and state written by code |
| musique | L | p14 | 761 | g-l | g-l: 533 | 331 | 103 | 1983 | 5.65e-29 | win | full set | FS and paired test measured; strongest ghost and state written by code |
| musique | R | j-rrf4 | 831 | g-r2 | g-r: 661; g-r2: 666 | 264 | 99 | 2054 | 1.88e-18 | win | full set | FS and paired test measured; strongest ghost and state written by code |
| musique | A | j-rrf4 | 831 | g-a1 | g-a1: 1101; g-a2: - | 238 | 508 | 1671 | 2.29e-23 | loss | full set | FS and paired test measured; strongest ghost and state written by code |
<!-- END GENERATED -->

Reading it (interpretation):
- The one loss is MuSiQue class A, to G-A1, an agentic LLM ghost.
- The HotpotQA row is an upper bound for the own systems: in-sample P10-C and P14 weights, BGE-small fine-tuned on HotpotQA train; G-L and G-A1 are in-domain there too.
- The class R and A cells on HotpotQA cover the 1,000 preregistered qids only (measured on that subsample).
- The class A bar is G-A1 alone; without G-A2 a class A win means a win over the ghosts that ran.
- The light-class bar is weak: G-L is weaker than the light systems already on disk, and MDR is not reproduced.

## The exam (Phase 07)

The author named the own systems from the terrain figures (Phase 06); nothing was chosen from the exam.
The class L entrant `rrf4` was rebuilt on the laptop with the frozen code (deviation 07.4, interpretation) [src: docs/plans/fase-07-exam/plan.md].

<!-- BEGIN GENERATED: exam -->
In scope: 1337 questions (1106 single-evidence, 231 multi-evidence); FS and paired tests measured; states and verdict written by code.

| System | Role | FS@2,048 | Single | Multi | Within-paper | Cost |
|---|---|---|---|---|---|---|
| G-A1 | ghost | 591 | 568 | 23 | - | not recorded |
| G-L | ghost | 1058 | 948 | 110 | - | not recorded |
| G-R | ghost | 1102 | 982 | 120 | - | not recorded |
| G-R2 | ghost | 1056 | 928 | 128 | - | not recorded |
| bm25 | component | 770 | 683 | 87 | 969 | not recorded |
| dense | component | 775 | 716 | 59 | 1038 | not recorded |
| hop | component | 68 | 64 | 4 | - | not recorded |
| j-rrf3 | control | 1105 | 983 | 122 | - | not recorded |
| j-rrf4 | entrant | 1107 | 983 | 124 | - | not recorded |
| p10-b | context | 838 | 755 | 83 | - | not recorded |
| p14 | context | 811 | 734 | 77 | - | not recorded |
| rrf4 | entrant | 928 | 832 | 96 | - | not recorded |
| within-j-strong | within-paper control only | - | - | - | 1129 | not recorded |

| Class | Own | Own FS | Strongest ghost | Ghosts FS | Wins | Losses | Ties | Exact p | State | Against j-rrf3 | Label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L | rrf4 | 928 | G-L | G-L: 1058 | 55 | 185 | 1097 | 1.24e-17 | loss | - | FS and paired test measured; strongest ghost and state written by code |
| R | j-rrf4 | 1107 | G-R | G-L: 1058; G-R: 1102; G-R2: 1056 | 8 | 3 | 1326 | 0.227 | tie | tie | FS and paired test measured; strongest ghost and state written by code |
| A | j-rrf4 | 1107 | G-R | G-A1: 591; G-L: 1058; G-R: 1102; G-R2: 1056 | 8 | 3 | 1326 | 0.227 | tie | tie | FS and paired test measured; strongest ghost and state written by code |
<!-- END GENERATED -->

Reading it (interpretation):
- Class L: `rrf4` loses to G-L on the exam.
- Classes R and A: `j-rrf4` ties G-R and ties the literature control `j-rrf3`; a tie is not losing, not a win.
- Only 231 of the 1,337 questions in scope are multi-evidence (measured), so the exam tests mostly single-evidence retrieval, which the project does not target.
- The entity hop returns an empty ranking on 301 of 1,451 exam questions (measured) [src: docs/plans/fase-07-exam/annex.md].

Exam cost per system:

<!-- BEGIN GENERATED: exam-cost -->
| System | Offline s | Online s/question | Unsplit s | USD | Hardware | Label |
|---|---|---|---|---|---|---|
| G-A1 | 311 | 0.227 | 0 | 0.283 | NVIDIA A100-SXM4-80GB | derived (sum of measured components) |
| G-L | 0 | 0 | 69.4 | 0.0143 | NVIDIA GeForce RTX 4090 | derived (sum of measured components) |
| G-R | 27432 | 0.525 | 69.4 | 0.165 | NVIDIA GeForce RTX 4090, laptop CPU (ARM64, Windows) | derived (sum of measured components) |
| G-R2 | 27428 | 0.661 | 69.4 | 0.205 | NVIDIA GeForce RTX 4090, laptop CPU (ARM64, Windows) | derived (sum of measured components) |
| j-rrf3 | 27432 | 0.467 | 69.4 | 0.148 | NVIDIA GeForce RTX 4090, laptop CPU (ARM64, Windows) | derived (sum of measured components) |
| j-rrf4 | 27432 | 0.466 | 69.4 | 0.147 | NVIDIA GeForce RTX 4090, laptop CPU (ARM64, Windows) | derived (sum of measured components) |
| p10-b | 4593 | 0.0439 | 0 | 0 | laptop CPU (ARM64, Windows) | derived (sum of measured components) |
| p14 | 27397 | 0.044 | 0 | 0 | laptop CPU (ARM64, Windows) | derived (sum of measured components) |
| rrf4 | 27397 | 0.0439 | 69.4 | 0.0143 | NVIDIA GeForce RTX 4090, laptop CPU (ARM64, Windows) | derived (sum of measured components) |
<!-- END GENERATED -->

## Money

Each row carries the three figures where they exist: time x rate (derived), balance delta (measured) and invoice (measured), with the file it is read from.

<!-- BEGIN GENERATED: money -->
| Phase | Time x rate USD, derived (time x rate) | Balance delta USD, measured (balance delta) | Invoice USD, measured (invoice) | Note | Source |
|---|---|---|---|---|---|
| 00 | - | - | - | no pod rented | docs/plans/fase-06-rivals/spec.md |
| 01 | - | - | - | no pod rented | docs/plans/fase-06-rivals/spec.md |
| 02 | 14.48 | 14.73 | 14.78 | over the 11.5 USD phase cap (deviation 02.1) | docs/plans/fase-02-ghosts/plan.md |
| 03 | - | - | 0 | no pod; the author's billing reading shows no charge inside the phase (deviation 03.1) | docs/plans/fase-03-fusion/plan.md; docs/plans/fase-06-rivals/spec.md |
| 04 | 0.32 | 0.33 | 0.33 |  | docs/plans/fase-04-judge/plan.md |
| 05 | - | 0 | - | no pod rented | docs/plans/fase-05-hop/plan.md; docs/plans/fase-06-rivals/spec.md |
| 06 | 7.490 | 7.63 | - | invoice pending when the phase closed | docs/plans/fase-06-rivals/plan.md |
| 07 | 1.464 | 1.489 | 1.489 | balance delta 1.4894290670 USD in the source | docs/plans/fase-07-exam/plan.md |
| total | 23.754 | 24.179 | - | derived sums; not available: the Phase 06 invoice was pending at its close; balance first to last 24.225 (derived: 29.22 USD before Phase 02 (docs/plans/fase-02-ghosts/plan.md) minus 4.995 USD at the Phase 07 close (docs/plans/0_plan_maestro.md)) | - |
<!-- END GENERATED -->

Gaps, declared instead of estimated:
- Phases 00 and 01 rented no pod.
- Phase 03 has no clean balance delta (deviation 03.1, interpretation) [src: docs/plans/fase-03-fusion/plan.md]; the author's billing reading shows no charge inside it.
- The Phase 06 invoice was pending when the phase closed, so the invoice total is not available.

This phase spent no money and measured nothing.

## Limits and biases

- One exam corpus, run once: a tie or a loss there is one draw.
- QASPER offers few multi-evidence questions, so the exam is weak on the case the project was built for.
- The terrain was looked at while designing and naming the systems; terrain wins show fit to the terrain, and the exam is the out-of-sample check.
- HotpotQA is partly in-domain for several components, so its wins are an upper bound.
- The light-class bar is weak (G-L, no MDR) and the class A bar is G-A1 alone (G-A2 not run).
- Some costs were never recorded (inherited old-pod Dense embeddings in Phases 03-05, several Phase 07 first-stage timings); the cost comparison is partial there.
- Metrics differ by phase (Phase 02 lacks some, Phase 07 reports FS@2,048 only).

## What the result shows and does not show

It shows (interpretation):
- On the terrain, fusions and a zero-shot judge over a pooled bag beat the measured ghosts of their class in every cell except MuSiQue class A.
- On the exam, the rerank-class system does not lose to its ghost or to the literature control; the light-class system loses to its ghost.

It does not show:
- That any own strategy beats the literature outside the terrain: the best exam outcome is a tie.
- That the entity hop helps once a judge orders the pool: on the exam `j-rrf4` and `j-rrf3` tie.
- Anything about class A against HippoRAG 2, which never ran.
- How the systems compare on an exam rich in multi-evidence questions.

## What a successor could try

- An exam corpus with many multi-evidence questions and native paragraph gold, opened once under a frozen rule.
- Reproduce MDR as the light-class bar and run G-A2 (HippoRAG 2) with enough budget, so both bars stand on the strongest recipe.
- Record every first-stage and embedding cost at run time, so no cost cell is left `not recorded`.
- Refit or replace the in-sample HotpotQA components before claiming wins there.
