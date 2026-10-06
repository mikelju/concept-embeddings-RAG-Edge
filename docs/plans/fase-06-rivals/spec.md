# Phase 06 - Literature rivals never measured on the bench: specification

Status: approved (frozen)
Approved: 2026-10-06, by the author
Master plan: `../0_plan_maestro.md`; ghosts and forecast: `../fase-00-state-of-the-art/survey.md` sections 5 and 7; ghost code and pod recipe: `../fase-02-ghosts/` (`spec.md`, `plan.md`, `pod_recipe.md`, `results.md`, `02.1-money.md`).
Numbering: the rivals are Phase 06, the exam becomes Phase 07 and the game Phase 08 (decision 1); earlier frozen documents that say "Phase 06" for the exam are not edited.
Delegation: the author delegated every further decision of this phase to the agent ("decide everything needed, deliver the whole phase"), within the frozen criteria and the money cap below.

## Objective
Measure on this harness the three literature rivals that Phase 00 named and Phase 02 never ran: G-R2 (Qwen3-Reranker-0.6B), G-A2 (HippoRAG 2) and G-A1 (Search-R1) on the HotpotQA subsample.
Until they run, the project cannot say its best system beats the literature in any cost class: it can only say it beats the ghosts that happen to have been measured.
Said first and plainly, the current position is weak: no candidate of Phases 03-05 entered the exam, and on MuSiQue the best system so far is a ghost, G-A1 (FS@2,048 1,101 of 2,417 against 809 for the old union pool under J-strong, paired 524 wins and 232 losses, exact p 8.2e-27; measured, Phase 02 `results.md`; named best system so far in Phase 05 `results.md`).
On MultiHop-RAG G-A1 loses to the old union pool (557 against 846, measured), and on HotpotQA G-A1 was never run (Phase 02 optional item, not run under deviation 02.1).
The phase adds no candidate; it raises or confirms the bar every claim must clear, and it fixes, by code, which claims of "beats the literature" the project may make on the terrain.

## Question
On the terrain, what Full Support @2,048 and what offline and online cost do G-R2, G-A2 and G-A1 (HotpotQA subsample) reach, and, paired on the same questions, does the project's best own system still win, tie or lose against the strongest measured ghost of each cost class?

## Inputs measured on disk (2026-10-06)
Read while writing this draft; every figure is measured unless labelled otherwise; no gold was read.
- G-L depth-100 rankings on the three sets: `data/phase02/rankings/<set>/g-l.jsonl.gz` (MuSiQue and MultiHop-RAG, session 2 builds) and `data/phase02/session6/s6/rankings/hotpotqa-dev/g-l.jsonl.gz` (HotpotQA); digests in Phase 02 `plan.md`.
- The G-L HotpotQA index (35 GB on the pod, Phase 02 F9) was not downloaded: `data/phase02/session6/` holds 30 MB of rankings, manifests and logs, and no network volume exists (Phase 04 and 05 balance readings).
  G-A1 on HotpotQA therefore needs the index rebuilt (cost below).
- Best figures so far by FS@2,048 (Phase 05 `results.md`): HotpotQA `j-rrf4` 5,950 of 7,405; MuSiQue `g-a1` 1,101 of 2,417; MultiHop-RAG `j-rrf3` 873 of 2,255.
- G-A1's driver, retrieval server and its 50-question check (C4 of Phase 02: same first search 49 of 50, same answer 46 of 50, measured) exist and are reused unchanged.

## What stays fixed
- Corpora, units, questions, gold and token counts: the old project's files, digest-checked through `OldData` (Phase 01).
- Metrics, as in Phases 02-05: FS @1,024 / @2,048 (of record) / @4,096, FS@2/5/20 units, gold share at 5, Recall@k, nDCG@10.
- Paired test: exact McNemar on per-question FS@2,048, two-sided, alpha 0.05 per comparison, no correction (stated as a limitation).
- The rankings and results of earlier phases are read as they are, against their digests; none is rebuilt or rewritten, except that the G-L HotpotQA index is rebuilt as a new, separately digested artifact for G-A1 (its rankings are not substituted for the reported G-L build; Phase 02 F5 says G-L varies a little across builds).
- No rival is tuned on the terrain: published settings, or the settings fixed here before anything runs; a fidelity miss is reported, never closed by changing a setting.
- The exam corpus (QASPER) is not opened, downloaded or read in this phase; the contamination check below reads only model cards, papers and dataset documentation about it.

## Frozen definition of the rivals
- **G-R2** (class R): `Qwen/Qwen3-Reranker-0.6B`, pinned revision, used as a pointwise scorer over G-L's top-100 with the model card's prompt format, default instruction and yes/no logit, no text generated; order by score descending, then G-L rank.
  Sets: MuSiQue (2,417), MultiHop-RAG (2,255) and HotpotQA dev (7,405), all in full (Phase 00 scoped the first two; decision 3 adds HotpotQA).
  Pairs are prepared on the laptop (unit texts of G-L's top-100), as Phase 04's `judge-pairs` did, so the pod needs no corpus.
- **G-A2** (class A): HippoRAG 2 from `github.com/OSU-NLP-Group/HippoRAG` at a pinned commit, with Llama-3.1-8B-Instruct served by vLLM for extraction and online steps and GritLM-7B as the encoder, as fixed in Phase 00 and Phase 02; staged on one GPU (extraction, then encoding), every other setting at the repository default.
  Set: MultiHop-RAG only (27,989 units, 2,255 questions); MuSiQue stays out (about 5.9 USD, projection, Phase 00 section 7).
  Its ranking is HippoRAG 2's passage ranking at depth 100; the code diff against the pinned commit is limited to model names, endpoints, retrieval depth and the input/output adapters, and is recorded.
- **G-A1 on HotpotQA** (class A): the Phase 02 Search-R1 driver, checkpoint, prompt, greedy decoding, top-3 retrieval, at most 8 search calls and 1,024 new tokens per turn, unchanged, searching a G-L HotpotQA index rebuilt with the Phase 02 D17 settings (one `create` call, fp16 encode batches).
  Questions: the preregistered 1,000-question HotpotQA dev subsample, `random.Random(20261002).sample` over the sorted qids (master plan, 2026-10-02; Phase 02 spec); the qid list and its sha256 are written before the pod opens.
  Evidence list: retrieved passages in retrieval order, duplicates dropped, filled to the budget, as in Phase 02; answer EM is context only.

## Comparisons and the literature bar
The rule below is applied by code to terrain figures only; rivals are references, never exam entrants from this phase.
- Each rival is paired, per set, against: the best system so far (Phase 02-05 results, by FS@2,048), the project's best own system (best of the old systems and Phase 03-05 systems, ghosts excluded), and the other ghost of its class (G-R2 against G-R; G-A2 against G-A1).
- On the HotpotQA subsample every comparison is restricted to the 1,000 qids; the full-set figures stay as they are.
- State per comparison: `win` when wins > losses and p < 0.05, `loss` when losses > wins and p < 0.05, `tie` otherwise, `not run` when the rival was not run on that set.
- **Literature bar**: per set and cost class (L, R, A), the strongest measured ghost by FS@2,048 (G-L; G-R or G-R2; G-A1 or G-A2), and the state of the project's best own system of that class or a cheaper one against it.
  "Beats the literature in class X on set S" may be written only where that state is `win`; `not run` is never a win.

## Pending checks before any money (licences and training data)
Read and recorded in `plan.md`, with source and date, before the first pod; "inferred" marks a conclusion the source does not state.
Training rule (Phase 00 section 5): a terrain train split in a rival's training mix makes that set in-domain (labelled); our dev questions or any QASPER data in it make the rival an upper reference, not a ghost, and it is then not run unless the author says so.

| Rival | Licence (state today) | Still to check |
|---|---|---|
| G-R2 Qwen3-Reranker-0.6B | Apache-2.0 (card, read in Phase 02) | Reranker training data is not named on the card; Phase 02 labelled HotpotQA in-domain from the Qwen3-Embedding data (inferred). Check the Qwen3-Embedding report for HotpotQA, MuSiQue, MultiHop-RAG and QASPER (train or dev); base Qwen3 pretraining is undisclosed, so dev-set leakage cannot be excluded and is labelled unknown |
| G-A2 HippoRAG 2 | code MIT (Phase 02); Llama-3.1-8B community licence, eligible (Phase 02); GritLM-7B Apache-2.0 (Phase 00) | Re-read the licence at the pinned commit (HippoRAG 2 lives in the same repository; confirm); Llama-3.1 is gated on Hugging Face, so access with the author's account is needed; GritLM's E5S and MEDI2 data for HotpotQA, MuSiQue, MultiHop-RAG and QASPER (Phase 02 inferred HotpotQA); Llama-3.1's pretraining cutoff against MultiHop-RAG's news articles (late 2023): whether the corpus text itself may be in pretraining (interpretation until checked) |
| G-A1 Search-R1 | code Apache-2.0; checkpoint licence unstated, used and labelled, weights not redistributed (Phase 02 decision); base Qwen2.5-7B Apache-2.0 | Trained on NQ plus HotpotQA train: HotpotQA in-domain (Phase 00); confirm from the repository's data script that HotpotQA dev is not in its training data; base Qwen2.5-7B pretraining undisclosed, labelled unknown; QASPER not in its stated training data (check) |

## Cost projection (projection, not measurement)
Basis: RunPod Secure Cloud prices recorded in earlier phases, re-read from the API before each pod: RTX 4090 24 GB at 0.74 USD/h (measured, Phase 04 pod, 2026-10-04) and A100-SXM4-80GB at 1.59 USD/h (measured, Phase 02 D15 and D17, 2026-10-03, the 250 GB RAM offer for HotpotQA).
Phase 00 projected G-R2 at 1.0 USD and G-A2 on MultiHop-RAG at 2.4 USD on the terrain (3.4 USD together), G-A1 on the HotpotQA subsample at 0.3 USD (assuming the index existed), and 2.6 USD for G-R2 and G-A2 on the exam (survey section 7).
Phase 02 measured that these projections run short: G-R on HotpotQA took 84 min against a projected 51 (F9), and the phase overran its cap (deviation 02.1); so each figure below adds 25 % contingency.

| Rival | GPU and basis | GPU-h | USD |
|---|---|---:|---:|
| G-R2, MuSiQue and MultiHop-RAG | 4090; 4,672 questions at about 0.6 s (1.5 x G-R's measured 0.44 / 0.39 s per question on the 4090, assumed), 0.4 h setup, + 25 % | 1.5 | 1.1 |
| G-R2 on HotpotQA (decision 3) | 4090; 7,405 questions at about 0.6 s, no extra setup, + 25 % | 1.5 | 1.1 |
| G-A2, MultiHop-RAG | 4090; Phase 00's 3.2 GPU-h (indexing 1.6, encoding 0.3, swaps 0.2, queries 0.6, setup 0.5), + 25 % | 4.0 | 3.0 |
| G-A1, HotpotQA 1,000 | A100 80 GB, 250 GB RAM; setup and upload 0.4 h, index rebuild 1.3 h (session 6 measured encode 3,877 s plus index 685 s), 1,000 questions about 0.5 h (about 3 searches at G-L's measured 0.358 s plus generation), download 0.1 h, + 25 % | 2.9 | 4.6 |
| Total | | 9.9 | 9.8 |

Every figure in this table is a projection; the totals are derived from the rows.

## Money state and stopping rule
- Money state before this phase: 15.11 USD spent on the project (derived: Phase 02 invoice 14.78 plus Phase 04 invoice 0.33; Phases 00, 01, 03 and 05 rented no pod); the RunPod account balance is 14.11 USD (measured, `clientBalance` 14.1147504042, 2026-10-05T06:57:23Z, Phase 05 `plan.md`).
- Authorization (decision 2): the author authorizes spending the remaining account balance on this phase, in place of the earlier 25 USD total and the 7.49 USD that was left above the exam core reserve.
- **Phase cap** = the `clientBalance` read before the first pod of this phase minus 1.0 USD of safety buffer, because RunPod stops pods when the balance reaches zero; at the 2026-10-05 reading that is about 13.1 USD (derived), and the cap is fixed, as a derived figure, from the reading actually taken before the first pod.
- Total spend authorized for the project becomes 15.11 USD (spent) plus the phase cap (derived).
- The exam core reserve (2.4 USD, projection, Phase 00 section 7) is dropped: **the exam (Phase 07) is unfunded** after this phase and needs a top-up that the author decides at that time.
- The projected total (9.8 USD) fits under the phase cap; the run order is G-R2 (three sets), then G-A2 on MultiHop-RAG, then G-A1 on the HotpotQA subsample (decision 4), and the agent may swap G-A2 and G-A1 if Llama-3.1 access is not ready when G-A2 would start; an item runs only if the spend so far plus its re-projection stays within the phase cap, and an item that does not fit is recorded as not run.
- Before each pod: read `clientBalance` and `myself { pods }`, record the rate offered, and re-project the item at that rate; at the end, close with a reading at least 2 h after the last termination.
- Hard cut per item at 1.5 times its projection, time x rate (G-R2 on the three sets 3.3 USD, G-A2 4.5 USD, G-A1 6.9 USD), never past the phase cap; a stage whose measured pace projects past its cut stops and is recorded (deviation 02.1's lesson: project time and RAM at full scale from a probe at that scale before committing).
- Llama-3.1 access (decision 5): the author accepts the licence on Hugging Face and sets a user-level Windows environment variable `HF_TOKEN` (read token); agents check only its presence, with `env | grep -o "^HF_TOKEN="`, never print it, and pass it to the pod as an environment variable; if it is still absent when every other item is done, G-A2 is recorded `not run` (reason: gated access), with no swap to another LLM.
- Abort an item, without retry under other settings, when: its fidelity or determinism check fails (C3); gated model access is refused; the training-data check makes it an upper reference; or it runs out of memory once on the projected GPU (one resize to the next GPU tier only if the re-projection fits the cut).
- No pod is left running; per-pod session cap 15 USD (master plan) is never approached.

## Scope
In, in run order:
1. Licence and training-data checks for the three rivals, recorded before any pod.
2. G-R2 pair preparation on the laptop, pod scoring with fidelity and determinism checks, depth-100 rankings.
3. G-A2 on MultiHop-RAG.
4. G-A1 on the HotpotQA subsample, with the G-L index rebuild.
5. Results table and literature bar, written by the shared results module (Phase 05).

Out: new candidates; any fitting or tuning on the terrain; HippoRAG 2 on MuSiQue or HotpotQA; Search-R1 with its own retriever or corpus; G-A1 on the full HotpotQA dev; any model above 8B or any closed API; the exam corpus; rival runs on the exam (they belong to the exam phase).

## Acceptance criteria
Frozen on approval. Changing them requires a deviation approved by the author.

| ID | Observable criterion | How it is checked |
|---|---|---|
| C1 | Before the first pod, `plan.md` records for each rival its licence and, for HotpotQA, MuSiQue, MultiHop-RAG and QASPER, whether train or dev data are in its training mix (yes, no, unknown), each with source and date; each rival is labelled ghost or upper reference by the training rule | the record in `plan.md`, its commit earlier than the first pod's creation time |
| C2 | The HotpotQA subsample qid list (1,000 qids) is written once with its sha256 before any G-A1 run; a test recomputes it from the frozen rule | the file, its sha256 in `plan.md`, the test |
| C3 | Fidelity: G-R2's batched scorer reproduces the model card's reference code on 300 pairs (the first 100 per set in pair-file order, on each of the three sets) within 1e-3 absolute, and rescoring the same pairs in reversed batch order differs by at most 1e-3 (bf16 determinism gate, research protocol); G-A2's diff against the pinned HippoRAG commit touches only model names, endpoints, retrieval depth and adapters; G-A1 runs the Phase 02 driver at an unchanged code path | saved check JSON; the recorded diff; `git diff` of the driver against Phase 02's commit |
| C4 | Each rival that runs has depth-100 rankings (G-R2, G-A2) or evidence lists (G-A1) for every question in its scope, written once, with a manifest pinning model ids and revisions, code commit, pod type and `costPerHr`, offline and online seconds, and, for G-A2, LLM input and output tokens | manifests under `data/phase06/`, digests in `plan.md` |
| C5 | The results table, written by the shared results module, lists per set every rival run, G-L, G-R, G-A1, the best system so far and the project's best own system, with every metric of record, a label per value and offline and online time and USD; a rival not run appears as `not run` with its reason | `data/phase06/results.json` and `results.md` generated from it |
| C6 | The same table gives exact McNemar on FS@2,048 for each rival against the comparisons listed above, per set (on the 1,000 qids for HotpotQA G-A1), with wins, losses, ties, p and state, and the literature bar per set and class with its state, all written by code | tests on a hand-built example for the state and the bar; the generated page |
| C7 | The exam corpus is not touched: no QASPER file is downloaded, and no command, manifest or log of this phase names a QASPER data path | `git grep` and a search of `data/phase06/` manifests and logs for `qasper`, output recorded |
| C8 | Money: `clientBalance` read before each pod and at least 2 h after the last termination, with UTC time; spend as time x rate (derived), balance delta (measured) and invoice when the author copies it; met when the phase spend is at most the phase cap (the `clientBalance` read before the first pod minus 1.0 USD, derived), no item passed its hard cut, and `myself { pods }` is empty at the close | readings and figures in `plan.md` |
| C9 | `npm run check` passes | exit 0 |

## Assumptions
- Qwen3-Reranker-0.6B scores at about 1.5 times bge-reranker-v2-m3's per-question time on our pairs (assumed, not measured).
- HippoRAG 2's code runs with an OpenAI-compatible vLLM endpoint and a local GritLM encoder without changes beyond those C3 allows (read from Phase 00, not checked against the current repository).
- An A100 80 GB offer with about 250 GB RAM is still available near 1.59 USD/h (measured 2026-10-03; re-read before the pod).

## Risks
- Projections ran short in Phase 02 by up to 1.6 times on G-R and far more on the HotpotQA index; the 25 % contingency may not cover it, which is why each item has a hard cut.
- MultiHop-RAG's news articles may be in Llama-3.1's pretraining; G-A2's figure would then be partly memorised (labelled, not testable here).
- Search-R1's checkpoint licence is unstated; results are reported, weights are not redistributed.
- The phase may spend the account down to about 1.0 USD, which leaves the exam (Phase 07) without money until the author tops the account up.
- The rebuilt G-L HotpotQA index differs slightly from the reported build (F5); G-A1 is reported with the index it searched, as in Phase 02.

## Anti-goals
- No second try of a rival with other settings to close a gap; no new candidate inside this phase.
- No reader, no LLM judge; no per-phase stage code beyond the pod scripts, the pair preparation and rows in the shared results module.

## Decisions taken
Taken by the author on 2026-10-06, when approving this spec.
1. **Numbering**: option A. The rivals are Phase 06 (`fase-06-rivals`), the exam becomes Phase 07 and the game Phase 08; earlier frozen documents that say "Phase 06" for the exam are not edited, and a master plan decision row records the mapping.
2. **Money**: the author authorizes spending the remaining RunPod account balance (14.11 USD measured on 2026-10-05, `clientBalance` 14.1147504042) on this phase, instead of the 7.49 USD cap.
   The phase cap is the `clientBalance` read before the first pod minus a 1.0 USD safety buffer (about 13.1 USD, derived); the exam core reserve is dropped and the exam (Phase 07) is unfunded until the author decides a top-up; the hard cut per item stays at 1.5 times its projection.
   Total spend authorized for the project: 15.11 USD spent plus the phase cap (derived).
3. **G-R2 on HotpotQA**: added; G-R2 runs on all three sets.
4. **Run order**: G-R2, then G-A2 on MultiHop-RAG, then G-A1 on the HotpotQA subsample; the agent may swap G-A2 and G-A1 if Llama-3.1 access is not ready when G-A2 would start.
5. **Llama-3.1 access**: the author accepts the licence on Hugging Face and sets a user-level Windows environment variable `HF_TOKEN` (read token); agents check its presence only with `env | grep -o "^HF_TOKEN="`, never print it, and pass it to the pod as an environment variable.
   If it is still absent when every other item is done, G-A2 is recorded `not run` (reason: gated access); no swap to another LLM, no change of recipe.
6. **Delegation**: the author delegated every further decision of this phase to the agent ("decide everything needed, deliver the whole phase"); merging stays the author's.
