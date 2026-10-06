# Phase 06 - Literature rivals never measured on the bench: plan

Status: in progress
Spec: `spec.md` (approved by the author, 2026-10-06; frozen)
Base: branch `fase-06-rivals` from `fase-05-hop` at 5b735e5

## Decisions
Taken by the agent under the author's delegation for this phase (spec, decision 6).
- D1. Phase 06 outputs live in `data/phase06/` (`pairs/<set>/`, `scores/<set>/`, `rankings/<set>/`, `checks/`, `logs/`, `subsample/`, `results.json`); `results.md` is generated into this folder, as in Phases 03-05.
- D2. G-R2 pair preparation reuses Phase 04's `judge-pairs` code path (`src/edge_rag/judge.py`: `pair_row`, `write_once`) over G-L's pinned depth-100 rankings; the pod scorer is a new `src/edge_rag/pod/` module beside `rerank.py`, and the rankings are assembled on the laptop, as Phase 04's `assemble` did.
- D3. G-A1 reuses the Phase 02 driver `src/edge_rag/pod/searchr1.py` and `scripts/pod_run.sh` unchanged; only the question list (the 1,000 qids) and the index path differ.
- D4. G-A2 code runs on the pod from a pinned HippoRAG commit; the adapter (our units in, a depth-100 ranking out) is the only code of ours on that path, and the diff against the pinned commit is saved for C3.
- D5. Money readings and pods follow `.agents/skills/remote-gpu/SKILL.md`, taken by the main session; `HF_TOKEN` is checked for presence only with `env | grep -o "^HF_TOKEN="` and never printed or written to a file.
- D6. A rival that the training-data check makes an upper reference is not run (spec, pending checks); the agent does not override that rule under the delegation.

## Increments
Each increment leaves the product working and covers named criteria.
This file is the durable state: a new session resumes from here and from Git.

| # | Increment | Criteria | Files | Status | Evidence |
|---|---|---|---|---|---|
| 1 | Licence and training-data checks for G-R2, G-A2 and G-A1 (model cards, papers, repositories, dataset docs), each answer yes, no or unknown per set with source and date, each rival labelled ghost or upper reference; HotpotQA 1,000-qid subsample written once from `random.Random(20261002).sample` over the sorted qids, sha256 recorded here, test recomputing it | C1, C2 | `plan.md`, `data/phase06/subsample/`, tests | partial: C2 done, C1 pending | C2 (2026-10-06, commit 6a9683c, `uv run edge-rag rivals-subsample`): `data/phase06/subsample/hotpotqa-dev-1000.txt`, 1,000 of 7,405 qids, one per line in the order `random.Random(20261002).sample(sorted(qids), 1000)` returns, sha256 6cebd41ff9e5e9f02e27ebb019620de0e37b77794fc6a783a2ff0d80fb3a62bd; manifest sha256 c87182fd1dd9b5de; no earlier qid list existed (Phase 04 only projected it, D8 there); `tests/test_rivals.py::test_subsample_file_recomputes_from_the_rule` recomputes it from the questions file and the pinned sha256 (pass). |
| 2 | G-R2 pair files prepared on the laptop for the three sets from G-L's pinned top-100 (reusing the Phase 04 pair code), scorer code (model card prompt, default instruction, yes/no logit) and assembly into depth-100 rankings, with local unit tests on a hand-built example | C3, C4 | `src/edge_rag/`, `src/edge_rag/pod/`, `scripts/`, tests | done | Code at commit 6a9683c: `src/edge_rag/rivals.py` (`rivals-pairs`, `rivals-rank`), `src/edge_rag/pod/qwen_rerank.py` (`--check`, `--score`), `scripts/pod_gr2.sh`, `tests/test_rivals.py` (subsample rule, ranking order and tie break, refusals, manifest, C3 comparison: 5 pass). Pairs (`uv run edge-rag rivals-pairs --set <set>`, G-L top 100 in question-file order, read after G-L matched `config.GL_SHA256`; manifests at 6a9683c with no uncommitted source change; the first manifests, written at 019557d with the code uncommitted, kept as stamped copies, same bytes): musique 2,417 questions, 241,700 pairs, 65,269,438 bytes, sha256 01de92b59a506addbcc749ed764612d581834089b5e3efd380b8800292f90639 (manifest 638528fbd34d76ac); multihop-rag 2,255, 225,500, 40,913,525 bytes, 41fc4353e2e4461beed721744dfbd6d17e523f01077f62f1386795d74bd713f1 (manifest 8e94dc6e2a98b41d); hotpotqa-dev 7,405, 740,500, 142,059,291 bytes, 7c778d1d964b89cad6c72d7b7112cb7a836971a29ba05e827f3a35bfc613a3d7 (manifest bc3b55800988f58e; 64.8 s, peak RSS 1,744 MB). `npm run check` exit 0. |
| 3 | G-R2 pod run: fidelity on 300 pairs against the model card's reference code (within 1e-3), reversed-batch determinism (within 1e-3), then scoring the three sets; rankings and manifests written once | C3, C4, C8 | `data/phase06/`, `plan.md` | pending | |
| 4 | G-A2 pod run on MultiHop-RAG: pinned HippoRAG commit, Llama-3.1-8B-Instruct through vLLM, GritLM-7B; recorded diff; depth-100 rankings and manifest with LLM tokens; `not run` if `HF_TOKEN` is still absent after every other item | C3, C4, C8 | `data/phase06/`, `plan.md` | pending | |
| 5 | G-A1 pod run on the HotpotQA 1,000: G-L HotpotQA index rebuilt with the Phase 02 D17 settings (new, separately digested artifact), Phase 02 driver unchanged (`git diff` recorded), evidence lists and manifest | C3, C4, C8 | `data/phase06/`, `plan.md` | pending | |
| 6 | Phase 06 rows in the shared results module `src/edge_rag/phase_results.py` (rivals, G-L, G-R, G-A1, best so far, best own system; McNemar per comparison; literature bar per set and class); tests of the state and the bar on a hand-built example; generated `data/phase06/results.json` and `docs/plans/fase-06-rivals/results.md` | C5, C6 | `src/edge_rag/phase_results.py`, `src/edge_rag/cli.py`, tests, `results.md` | pending | |
| 7 | Close: C7 `git grep` and search of `data/phase06/` manifests and logs for `qasper`; C8 closing `clientBalance` reading at least 2 h after the last termination and the three money figures; C9 `npm run check`; adversarial review; delivery (push and PR) | C7, C8, C9 | `plan.md`, master plan | pending | |

Run order of the pod items: 3, 4, 5; items 4 and 5 swap if Llama-3.1 access is not ready when item 4 would start (spec, decision 4).

## G-R2 pod recipe (increment 3)
Written before the pod; follows `.agents/skills/remote-gpu/SKILL.md`; money rule in the section below.
- Pod: 1x RTX 4090 Secure Cloud (`podFindAndDeployOnDemand`, `cloudType: SECURE`), image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, container disk 40 GB, volume 30 GB, `HF_HOME=/workspace/hf`, ports `22/tcp`; record `costPerHr`.
- Before: read `clientBalance` and `myself { pods }`; phase cap = balance minus 1.0 USD; re-project at the offered rate.
- Freeze and push the commit the pod runs (this recipe's commit or later); the pod reads the pairs only after their sha256 equals the laptop manifests', so the pod commit need not be 6a9683c, and its C3 check and scoring must share one commit with no uncommitted source change (the scorer refuses otherwise).
- Laptop: `(cd data/phase06/pairs && sha256sum *.jsonl.gz > ../pairs.sha256)`; upload `data/phase06/pairs/{musique,multihop-rag,hotpotqa-dev}.{jsonl.gz,manifest.json}` (about 248 MB) to `/workspace/<repo>/data/phase06/pairs/` and `pairs.sha256` to `/workspace/pairs.sha256`. No corpus, no old data.
- Pod: clone into `/workspace`, check out the frozen commit, install `uv`, then `POD_COST_PER_HR=<rate> nohup bash scripts/pod_gr2.sh > /workspace/pod_gr2.out 2>&1 &`; wait with a background `until grep -q "STAGE_DONE all\|STAGE_FAIL" /workspace/pod_gr2.log` loop, in waits shorter than 1,000 s per subagent.
- Stages: pair sha256 check; `uv sync --frozen --group pod`; CUDA kernel check; C3 (`qwen_rerank --check`, 300 pairs, writes `data/phase06/checks/c3-g-r2.json`; a fail ends the session and G-R2 is recorded aborted, no retry); scoring of musique, multihop-rag, hotpotqa-dev (`qwen_rerank --score <set>`, writes `data/phase06/scores/<set>.{jsonl.gz,manifest.json}`); `sha256sums.txt`.
- Download before terminating: `data/phase06/checks/`, `data/phase06/scores/` (with `sha256sums.txt`) and `/workspace/pod_gr2.{log,out,samples,progress}` to `data/phase06/logs/`; check every file against `sha256sums.txt` on the laptop; then `podTerminate` and `myself { pods }` empty.
- Laptop after: `uv run edge-rag rivals-rank --set <set>` for the three sets writes `data/phase06/rankings/<set>/g-r2.{jsonl.gz,manifest.json}`; digests here.
- Time (projection): setup 0.4 h; 12,077 questions (1,207,700 pairs) at about 0.6 s per question (spec assumption, 1.5 x G-R) 2.0 h; C3 and download under 0.1 h; + 25 % about 3.0 h, about 2.2 USD at 0.74 USD/h. Hard cut 3.3 USD (about 4.5 h at 0.74 USD/h): if the first set's measured pace in `pod_gr2.progress` projects the three sets past it, stop after the current set and record the rest `not run`.

## Licence and training-data record (C1)
Filled by increment 1, before the first pod.

## Money
Rule (spec, decision 2): phase cap = `clientBalance` read before the first pod minus 1.0 USD safety buffer (derived; about 13.1 USD at the 2026-10-05 reading of 14.1147504042 USD).
An item starts only if the spend so far plus its re-projection at the offered rate stays within the cap; hard cut per item at 1.5 times its projection (G-R2 3.3 USD, G-A2 4.5 USD, G-A1 6.9 USD), never past the cap.
The exam (Phase 07) is unfunded after this phase; it needs a top-up decided by the author.
- Before the first pod: pending (UTC time, `clientBalance`, `myself { pods }`, phase cap derived from it).
- Per pod: pending (rate offered, re-projection, start and end UTC, time x rate).
- Close: pending (reading at least 2 h after the last termination; balance delta; invoice when the author copies it).

## Decisions taken during execution
- D7. G-R2 runs in bfloat16 (research protocol: a bfloat16 model needs a determinism gate). With the pinned `transformers==5.16.1` the model card's own load call (no dtype) resolves the checkpoint's bfloat16, so the reference and the scorer share the dtype; the reference dtype is recorded in the check JSON. Batch 16 and maximum length 8,192 are Phase 02's fixed `QwenReranker` settings, unchanged.
- D8. C3 compares the model card's score, P(yes), within 1e-3 absolute: fidelity against the card's reference code run one pair per call (no padding), determinism against the same 300 pairs scored in reversed order (other batch composition and padding). The stored score is log P(yes) (Phase 02 code, free of saturation ties); its reversed-order difference is recorded, not gated, because a log near 0 probability magnifies bfloat16 noise without changing the order of the pairs that matter. Reason: the spec names no space and the card's output is P(yes).
- D9. The subsample file keeps the order `sample` returns, one qid per line, so its bytes are the rule's output; consumers read it as a set.


## Deviations
| ID | Summary | Affects criteria | Status |
|---|---|---|---|

## Adversarial review
| Round | Backend | Range | Lenses | Findings | Status |
|---|---|---|---|---|---|

## Results by criterion
Pending: filled at the close, per criterion, with the command run, the observed result and the evidence.

## Candidate learnings
