"""Phase 07 pod stage rules (plan increment 5b): the fixed order, the per-item probe and the
per-system completion marker the laptop downloads on (spec "Money and stopping rule", plan D9).
Imports only the standard library, so it runs in every pod environment and on the laptop.
`scripts/pod_exam.sh` reads the order from `order`, asks `decide` after each probe and calls
`mark` when an item ends; `scripts/exam_fetch.sh` on the laptop downloads an item as soon as
its marker exists and accepts it only when `check` passes on the downloaded copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Hard-coded before the exam opens (spec): verdict items first, then the within-paper control
# and the context rows; the A100 pod runs G-A1 alone.
ORDER = (
    "setup",
    "gliner",
    "g-l",
    "c3",
    "j-strong",  # G-R and `j-rrf4`
    "g-r2",
    "j-rrf3",
    "within-j-strong",
)
GA1_ORDER = ("setup", "g-a1")
PROBE_QUESTIONS = 50
PROBE_UNITS = 1_000
# Per-question items probe on their first 50 questions, corpus passes on their first 1,000
# units; setup and the C3 checks are fixed-size and run without a probe.
PROBE = {
    "gliner": PROBE_UNITS,
    "g-l": PROBE_UNITS,
    "j-strong": PROBE_QUESTIONS,
    "g-r2": PROBE_QUESTIONS,
    "j-rrf3": PROBE_QUESTIONS,
    "within-j-strong": PROBE_QUESTIONS,
    "g-a1": PROBE_QUESTIONS,
}
MARKER = "{name}.complete.json"
SKIP_EXIT = 3


def decide(
    probe_seconds: float,
    probe_count: int,
    total_count: int,
    *,
    cut_usd: float,
    spent_seconds: float,
    rate_per_hr: float,
) -> dict[str, Any]:
    """Project the full run from the probe's pace; run only if it fits the remaining cut
    (the cut minus the time x rate already spent). A skip is `not run (cut)`."""
    if probe_count <= 0 or total_count <= 0:
        raise ValueError("probe and total counts must be positive")
    projected_seconds = probe_seconds / probe_count * total_count
    projected_usd = projected_seconds * rate_per_hr / 3600
    remaining_usd = cut_usd - spent_seconds * rate_per_hr / 3600
    run = projected_usd <= remaining_usd
    return {
        "run": run,
        "state": "run" if run else "not run (cut)",
        "probe_seconds": round(probe_seconds, 3),
        "probe_count": probe_count,
        "total_count": total_count,
        "projected_seconds": round(projected_seconds, 3),
        "projected_usd": round(projected_usd, 4),
        "remaining_usd": round(remaining_usd, 4),
    }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mark(
    directory: Path, name: str, files: Sequence[str], extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Written once, after every output of `name` exists: the sha256 of each file, relative to
    `directory`. The marker is the last write, so its presence means the item ended."""
    marker_path = directory / MARKER.format(name=name)
    if marker_path.exists():
        raise FileExistsError(f"{marker_path} exists: an item is marked complete once")
    if not files:
        raise ValueError(f"{name}: a completion marker needs at least one file")
    body = {
        "system": name,
        "files": {rel: sha256_of(directory / rel) for rel in sorted(files)},
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", "utf-8")
    temporary.replace(marker_path)
    return body


def is_complete(directory: Path, name: str) -> bool:
    """The marker exists and every file it names is present with that sha256."""
    marker_path = directory / MARKER.format(name=name)
    if not marker_path.exists():
        return False
    files = json.loads(marker_path.read_text("utf-8")).get("files") or {}
    return bool(files) and all(
        (directory / rel).is_file() and sha256_of(directory / rel) == sha256
        for rel, sha256 in files.items()
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m edge_rag.pod.exam_stage")
    commands = parser.add_subparsers(dest="command", required=True)
    order = commands.add_parser("order", help="print the fixed order, one item per line")
    order.add_argument("--pod", choices=("4090", "a100"), default="4090")
    probe = commands.add_parser("probe-size", help="print an item's probe size, or nothing")
    probe.add_argument("item")
    dec = commands.add_parser("decide", help=f"exit 0 to run, {SKIP_EXIT} for not run (cut)")
    dec.add_argument("--probe-seconds", type=float, required=True)
    dec.add_argument("--probe-count", type=int, required=True)
    dec.add_argument("--total", type=int, required=True)
    dec.add_argument("--cut-usd", type=float, required=True)
    dec.add_argument("--spent-seconds", type=float, required=True)
    dec.add_argument("--rate", type=float, required=True)
    mk = commands.add_parser("mark", help="write <name>.complete.json over the given files")
    mk.add_argument("directory", type=Path)
    mk.add_argument("name")
    mk.add_argument("files", nargs="+")
    ck = commands.add_parser("check", help="exit 0 when <name> is complete in directory")
    ck.add_argument("directory", type=Path)
    ck.add_argument("name")
    args = parser.parse_args(argv)
    if args.command == "order":
        print("\n".join(ORDER if args.pod == "4090" else GA1_ORDER))
    elif args.command == "probe-size":
        if args.item in PROBE:
            print(PROBE[args.item])
    elif args.command == "decide":
        result = decide(
            args.probe_seconds,
            args.probe_count,
            args.total,
            cut_usd=args.cut_usd,
            spent_seconds=args.spent_seconds,
            rate_per_hr=args.rate,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["run"] else SKIP_EXIT
    elif args.command == "mark":
        mark(args.directory, args.name, args.files)
    elif args.command == "check":
        return 0 if is_complete(args.directory, args.name) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
