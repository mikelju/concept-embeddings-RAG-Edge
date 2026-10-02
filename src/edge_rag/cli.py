"""`uv run edge-rag <command>`: one generic runner, ASCII-only output."""

import argparse
import sys

from edge_rag import config


def _say(line: str) -> None:
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edge-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    reproduce = commands.add_parser("reproduce", help="run the reproduction gate on one set")
    reproduce.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    args = parser.parse_args(argv)
    if args.command == "reproduce":
        from edge_rag.reproduce import run

        result = run(args.set_name, say=_say)
        return 0 if result["gate_pass"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
