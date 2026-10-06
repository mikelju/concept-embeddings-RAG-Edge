"""Pace gate of the G-A1 HotpotQA session (Phase 06 spec, hard cut 6.9 USD; deviation 02.1's
lesson: project the whole session from the measured pace before committing the hours).

Standard library only, run as a script by `scripts/pod_ga1.sh` once a minute:

    python3 src/edge_rag/pod/ga1_pace.py --stage g-l --progress P --pace S --spent 1234 \
        --rate 1.59 --cut 6.9

`g-l`: the encode pace from `S` (lines `<epoch> <units encoded>`, sampled by the script) gives
the rest of the encode; the stages after it are added at their recipe projection. `g-a1`: the
mean seconds per driver turn so far (`[TURN]` lines in `P`) times the turns left (at most
`TURNS`) is an upper bound, since every turn has no more active questions than the one before.
Prints one `PACE` line; exit 1 when the spend so far or the projection passes the cut.
"""

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

UNITS = 5_233_329  # HotpotQA dev units (Phase 02 F9)
TURNS = 9  # the driver's MAX_SEARCHES (8) search turns plus one answer turn
MIN_PROBE_S = 600.0  # project the encode only from 10 minutes of samples
# Recipe projection of what follows each gated stage, in seconds (labelled projection):
INDEX_S = 685 * 1.25  # PLAID create after the encode (F9 measured 685 s) + 25 %
SERVER_S = 900.0  # retrieval server start: 5.2 M units read and the index loaded on the GPU
GA1_S = 1800.0  # spec: 1,000 questions about 0.5 h
COLLECT_S = 900.0  # index digests, output sums, download
TURN_RE = re.compile(r"\[TURN\] (\d+) .* elapsed_s ([0-9.]+)")


def encode_remaining(samples: Sequence[tuple[float, int]]) -> float | None:
    """Seconds left in the encode at the pace between the first and last sample, or None
    before `MIN_PROBE_S` of samples or while no unit has been added."""
    if len(samples) < 2:
        return None
    (t0, u0), (t1, u1) = samples[0], samples[-1]
    if t1 - t0 < MIN_PROBE_S or u1 <= u0:
        return None
    return (UNITS - u1) / ((u1 - u0) / (t1 - t0))


def turns_remaining(lines: Sequence[str]) -> float | None:
    """Upper bound on the seconds left in the turn loop, from the last `[TURN]` line."""
    last = None
    for line in lines:
        match = TURN_RE.search(line)
        if match:
            last = (int(match.group(1)), float(match.group(2)))
    if last is None:
        return None
    turn, elapsed = last
    return elapsed / turn * max(TURNS - turn, 0)


def projected_seconds(
    stage: str, progress: Sequence[str], samples: Sequence[tuple[float, int]]
) -> float | None:
    """Seconds left in the session from the measured pace, or None when not yet measured."""
    if stage == "g-l":
        left = encode_remaining(samples)
        return None if left is None else left + INDEX_S + SERVER_S + GA1_S + COLLECT_S
    if stage == "g-a1":
        left = turns_remaining(progress)
        return None if left is None else left + COLLECT_S
    return None


def read_samples(path: Path) -> list[tuple[float, int]]:
    if not path.exists():
        return []
    rows = [line.split() for line in path.read_text("utf-8").splitlines()]
    return [(float(r[0]), int(r[1])) for r in rows if len(r) == 2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--pace", type=Path, required=True)
    parser.add_argument("--spent", type=float, required=True, help="pod seconds so far")
    parser.add_argument("--rate", type=float, required=True, help="costPerHr, USD")
    parser.add_argument("--cut", type=float, required=True, help="hard cut, USD")
    args = parser.parse_args(argv)
    progress = (
        args.progress.read_text("utf-8", errors="replace").splitlines()
        if args.progress.exists()
        else []
    )
    left = projected_seconds(args.stage, progress, read_samples(args.pace))
    spent_usd = args.spent / 3600 * args.rate
    total_usd = None if left is None else (args.spent + left) / 3600 * args.rate
    shown = "unmeasured" if total_usd is None else f"{total_usd:.2f}"
    print(
        f"PACE stage {args.stage} spent_usd {spent_usd:.2f} projected_usd {shown} cut {args.cut}"
        " (projection from measured pace)",
        flush=True,
    )
    over = spent_usd >= args.cut or (total_usd is not None and total_usd > args.cut)
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
