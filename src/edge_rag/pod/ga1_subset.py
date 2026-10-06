"""G-A1 on the HotpotQA 1,000 (Phase 06, plan increment 5, D3): the Phase 02 driver
`edge_rag.pod.searchr1`, byte-unchanged, restricted to the preregistered qids.

The driver reads its questions through `common.set_questions`, which checks every digest and
the count of the whole question file (7,405), so a smaller question file cannot pass it. This
wrapper replaces that one function for the run: the replacement calls the original (every check
still runs) and keeps the 1,000 qids in question-file order. It can also set one run setting,
vLLM's share of GPU memory (the driver's `GPU_MEMORY_UTILIZATION`, 0.80): the retrieval server
holds the HotpotQA index (35 GB on disk, Phase 02 F9) on the same card, which the small MuSiQue
and MultiHop-RAG indexes did not (recipe). Then it calls the driver's own `main`, and records
what it changed in `subset.json` beside the driver's trace.

    uv run --group pod python -m edge_rag.pod.ga1_subset --qids <file> --qids-sha256 <hex> \
        [--gpu-memory-utilization 0.45]
"""

import argparse
import hashlib
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from edge_rag.artifacts import ArtifactError, sha256_file, write_json
from edge_rag.pod import common, searchr1

SET = "hotpotqa-dev"
SIZE = 1000
SetQuestions = Callable[..., tuple[list[Any], dict[str, str]]]


def read_qids(path: Path, expected_sha256: str) -> list[str]:
    """The qid list, one per line, refused unless its sha256 is the recorded one."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ArtifactError(f"{path}: sha256 {digest}, expected {expected_sha256}")
    qids = path.read_text("utf-8").split()
    if len(qids) != SIZE or len(set(qids)) != SIZE:
        raise ArtifactError(
            f"{path}: {len(qids)} qids ({len(set(qids))} distinct), expected {SIZE}"
        )
    return qids


def restricted(original: SetQuestions, qids: Sequence[str]) -> SetQuestions:
    """`set_questions` that runs the original and keeps only `qids`, in question-file order."""
    wanted = set(qids)

    def set_questions(*args: Any, **kwargs: Any) -> tuple[list[Any], dict[str, str]]:
        questions, answers = original(*args, **kwargs)
        kept = [q for q in questions if q.qid in wanted]
        if len(kept) != len(wanted):
            raise ArtifactError(f"{len(wanted) - len(kept)} listed qids not in the question file")
        return kept, answers

    return set_questions


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qids", type=Path, required=True)
    parser.add_argument("--qids-sha256", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    args = parser.parse_args(argv)
    driver_args = ["--set", SET]  # the driver's defaults otherwise: vLLM, G-L server on :8000
    qids = read_qids(args.qids, args.qids_sha256)
    record: dict[str, Any] = {
        "qids_file": args.qids.name,
        "qids_sha256": args.qids_sha256,
        "qids": len(qids),
        "order": "question file order (the driver's), restricted to the listed qids",
        "driver_args": driver_args,
        "driver_gpu_memory_utilization": searchr1.GPU_MEMORY_UTILIZATION,
        "gpu_memory_utilization": args.gpu_memory_utilization or searchr1.GPU_MEMORY_UTILIZATION,
        "driver_sha256": {
            Path(m.__file__ or "").name: sha256_file(Path(m.__file__ or ""))
            for m in (searchr1, common)
        },
        **common.git_provenance(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    common.set_questions = restricted(common.set_questions, qids)
    if args.gpu_memory_utilization is not None:
        searchr1.GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
    sys.argv = [searchr1.__file__ or "searchr1", *driver_args]
    searchr1.main()
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(searchr1.output_dir(SET, searchr1.SYSTEM) / "subset.json", record)


if __name__ == "__main__":
    main()
