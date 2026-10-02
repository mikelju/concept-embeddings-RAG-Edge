---
name: remote-gpu
description: How to run work the Windows ARM64 laptop cannot do (GPU extraction, large encodes, judges, packages without win_arm64 wheels) on a rented RunPod RTX 4090 driven through its API and SSH - setup, uploads, downloads, cost recording and traps. Load when a task needs CUDA, a large embedding or scoring pass, or a rented machine.
---

# Remote GPU runs

Carried from `concept-embeddings-RAG` (skill `remote-gpu`, handover section 8, old `docs/plans/phase_17/17.runpod_recipe.md`).
Renting compute spends money: ask the author first, with the estimate, and state the session cap.

## Why

The laptop is Windows 11 ARM64, 12 cores, about 32 GB RAM, no GPU. That is a development constraint, not a scientific one: never distort an experiment to fit it.
- spaCy cannot install (`blis` has no `win_arm64` wheel).
- GLiNER measured 0.558 paragraphs/s on the laptop CPU against 162.178 on an RTX 4090 (about 290x).
- Before adding a model dependency, check whether it installs here, whether another backend exists, and whether the work belongs on the pod.

## Environment facts

- Python `>=3.12,<3.13`; `pyproject.toml` resolves `torch` from `pytorch-cu126` on linux/x86_64 and from `pytorch-cpu` elsewhere: respect it.
- `uv run` re-syncs the venv from the lock, so a pod-local `uv pip install` fix is undone; change `pyproject.toml` and `uv.lock` and move the pod to that commit.
- `tool.uv.sources` only routes packages the project declares directly.

## API and SSH flow

- The API key is the Windows user-level environment variable `RUNPOD_API_KEY`; after setting it, restart VS Code.
  Check presence with `env | grep -o "^RUNPOD_API_KEY="`; **never print the value**. SSH key: `~/.ssh/runpod_ed25519`.
- `curl` to `https://api.runpod.io/graphql` with `Authorization: Bearer $RUNPOD_API_KEY`:
  `myself { clientBalance pubKey }` (read-only check); `gpuTypes(input:{id:"NVIDIA GeForce RTX 4090"}) { securePrice }`;
  `podFindAndDeployOnDemand` with `cloudType: SECURE`, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
  ports `22/tcp,8888/http`, `startSsh: true`, `supportPublicIp: true`, volume and container disk sized for the job.
- Poll `pod(input:{podId}) { runtime { ports } }` for the public port 22; then `ssh -i ~/.ssh/runpod_ed25519 -p <port> root@<ip>`, `scp` for files.
- End with `podTerminate` (**terminate, not stop**: a stopped pod bills its volume) and `myself { clientBalance pods { id } }` to confirm nothing is left.
- Reference rate: 1x RTX 4090 Secure Cloud, 0.74 USD/h (2026-10-01). Record `costPerHr` at creation.

## Recipe

1. A phase that rents compute writes its own recipe, with its stop conditions and cost cap, before the pod is opened.
2. Freeze and push the exact commit the pod will run; note its hash.
3. On the pod: clone into `/workspace`, check out the frozen commit, `uv sync` with the groups the job needs.
4. Upload only frozen inputs the repo does not contain; check their SHA-256 on the pod.
5. Verify CUDA with a real kernel, not only `torch.cuda.is_available()`.
6. Probe a small sample before the full pass.
7. Write `pod_setup.sh` in the scratchpad, `scp` it, run it with `nohup`, wait with a background `until grep` loop.
   Subagents hit a wall-clock limit near 1,000 s: never brief one with a long stage plus its wait.
8. `sha256sum` every expensive artifact on both ends and **download before terminating**: `/workspace` dies with the pod.
9. Decide on the laptop, not on the pod. Stop and ask the author if the session nears its cap.

## Traps

- The manifest `hardware` block reports the host (for example 256 cores, 1.08 TB RAM), not the container; record the allocation separately (cgroup v2).
- Billing lags about an hour behind a terminated pod: record time x rate, balance delta and invoice as three labelled figures; the author copies the invoice later.
- PyPI `vllm` 0.30.0 is a CUDA 13 build against the pinned `torch==2.13.0+cu126`: pin `torchvision==0.28.0` and `torchaudio==2.11.0`
  to `pytorch-cu126` via `tool.uv.sources`, and add `<venv>/lib/python3.12/site-packages/nvidia/cu13/lib` to `LD_LIBRARY_PATH`.
- `pkill -f "vllm serve"` kills the SSH session whose command line holds the pattern; use `pgrep -f "[v]llm serve"`.
- A float32 8 B model does not fit a 24 GB card.
