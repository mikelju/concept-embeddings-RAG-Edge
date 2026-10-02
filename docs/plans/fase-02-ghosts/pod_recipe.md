# Phase 02 - Pod recipe (increment 4)

Written before the pod is opened, as `.agents/skills/remote-gpu/SKILL.md` step 1 requires.
Spec: `spec.md` (money rule and stop states); plan: `plan.md` (D1, D5-D7).

## Machine
- One RunPod pod, 1x NVIDIA GeForce RTX 4090, Secure Cloud, on-demand.
- Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, ports `22/tcp,8888/http`, SSH key `~/.ssh/runpod_ed25519`.
- Container disk 40 GB, volume 100 GB at `/workspace` (PLAID index of HotpotQA, Search-R1 7B weights, two environments).
- Rate read before opening: `securePrice` 0.74 USD/h (measured, API, 2026-10-02); the `costPerHr` returned at creation is the one recorded.
- Balance before the phase: 29.22 USD (measured, API, 2026-10-02).

## Frozen inputs
- Code: the commit of `fase-02-ghosts` noted in `plan.md` before the pod runs; any fix found by a probe is a new commit on the same branch, recorded in the plan, and the pod moves to it.
- Old data, uploaded by `scp` and checked by `OldData` against the digests in `config.py` on the pod:
  `phase9`, `phase15`, `phase16`: `corpus.json`, `corpus.jsonl.gz`, `questions.json`;
  `phase17`: `pairs-<set>.jsonl.gz` and `scores-strong-<set>.jsonl.gz` (pinned in `rerank.py`).
- Models by pinned Hub revision (in the pod modules).

## Order
1. Create the pod, record `costPerHr` and the start time.
2. Clone, check out the frozen commit, install `uv`, upload the old data, `sha256sum` on both ends.
3. Probes before any full pass, each a few minutes:
   `sync` and `cuda` stages; G-L on 1,000 MuSiQue units and 100 questions; the C3 check (300 old pairs) as G-R's probe;
   Search-R1 on 5 MuSiQue questions (`--limit 5 --tag probe`) once G-L's MuSiQue index exists.
   Every failure is fixed on the laptop, committed, pushed, and the pod pulls the new commit; each fix is a row in the plan.
4. `scripts/pod_run.sh` in full, under `nohup`, watched from the laptop with a background wait on its stage markers.
5. `sha256sum` every output on the pod, download, check the digests on the laptop.
6. `podTerminate`, then `myself { pods { id } }` must list nothing; record the balance.

## Stop conditions
- Session cap: the pod is terminated when time x rate reaches 8 USD (about 10.8 h at 0.74 USD/h), whatever is running; downloaded outputs are kept, the rest is recorded as not run.
- Core cap: before each full stage, the rate measured on its probe (or on the previous set) projects the stage; a stage whose projection takes the phase past 7.9 USD is not started and is recorded.
- Two failed attempts on the same problem stop that stage: it is recorded as failed with its log, and the session moves on to stages that do not depend on it (G-R and G-A1 depend on G-L; G-A1 on the retrieval server).
- Idle pod: no stage running and no fix in progress for 20 minutes means terminate.
- Nothing is left on the pod at the end: outputs are downloaded before `podTerminate`.

## Projection (labelled projection, not measured)
- Setup and upload about 0.5 h; G-L on the three sets about 1 h (HotpotQA, 5.23 M units, dominates); G-R about 1.5 h (about 1.2 M pairs); C3 and C4 checks about 0.3 h; G-A1 about 1 h.
- About 4.3 h, about 3.2 USD at 0.74 USD/h; the probes replace this with measured rates before each full stage.
