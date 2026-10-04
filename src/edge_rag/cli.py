"""`uv run edge-rag <command>`: one generic runner, ASCII-only output."""

import argparse
import sys
from pathlib import Path

from edge_rag import config


def _say(line: str) -> None:
    print(line.encode("ascii", "replace").decode("ascii"), flush=True)


def _report(scored: dict) -> None:
    _say(f"[{scored['set']}] {scored['rankings']} n={scored['n']}")
    for budget, count in scored["full_support_at_budget"].items():
        _say(f"  FS@{budget} tokens: {count}")
    for k, count in scored["full_support_at_k"].items():
        _say(f"  FS@{k} units: {count}")
    share = config.GOLD_SHARE_K
    _say(f"  gold share at {share}: {scored[f'gold_share_at_{share}']:.4f}")
    for k, value in scored["recall_at_k"].items():
        _say(f"  Recall@{k}: {value:.4f}")
    _say(f"  nDCG@{config.NDCG_K}: {scored[f'ndcg_at_{config.NDCG_K}']:.4f}")
    if "against" in scored:
        against = scored["against"]
        test = against[f"paired_full_support_at_{config.BUDGET}"]
        _say(
            f"  against {against['rankings']} (FS@{config.BUDGET} "
            f"{against[f'full_support_at_{config.BUDGET}']}): wins {test['wins']} "
            f"losses {test['losses']} ties {test['ties']} exact p {test['p']:.4g}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edge-rag")
    commands = parser.add_subparsers(dest="command", required=True)
    reproduce = commands.add_parser("reproduce", help="run the reproduction gate on one set")
    reproduce.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    reproduce.add_argument("--out", type=Path, help="write rankings and result under this folder")
    components = commands.add_parser("components", help="Phase 03 Dense and BM25 lists")
    components.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    fuse = commands.add_parser("fuse", help="Phase 03 RRF3 and F3 lists")
    fuse.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    pool = commands.add_parser("pool", help="Phase 04 entity hop and RRF4 lists")
    pool.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    score = commands.add_parser("score", help="score one ranking file on one set")
    score.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    score.add_argument("--rankings", required=True, type=Path)
    score.add_argument("--against", type=Path, help="paired exact McNemar on FS@2,048 vs this")
    score.add_argument("--out", type=Path, help="write the result JSON here")
    refs = commands.add_parser("old-refs", help="convert old Phase 17 J(P10-B) and union lists")
    refs.add_argument(
        "--set", dest="set_name", required=True, choices=sorted(config.OLD_REFERENCES)
    )
    results_cmd = commands.add_parser("results", help="write the Phase 02 results table")
    results_cmd.add_argument(
        "--set", dest="set_names", action="append", choices=sorted(config.OLD_REFERENCES)
    )
    fusion_cmd = commands.add_parser("fusion-results", help="write the Phase 03 results table")
    fusion_cmd.add_argument(
        "--from-json",
        action="store_true",
        help="only check that results.md renders byte-equal from results.json as written",
    )
    args = parser.parse_args(argv)
    if args.command == "reproduce":
        from edge_rag.reproduce import run

        result = run(args.set_name, say=_say, out=args.out)
        return 0 if result["gate_pass"] else 1
    if args.command == "components":
        from edge_rag import components as components_module

        body = components_module.run(args.set_name, say=_say)
        equal = body["equality"].values()
        return 0 if all(entry["equal"] == entry["of"] for entry in equal) else 1
    if args.command == "fuse":
        from edge_rag import fuse as fuse_module

        fuse_module.run(args.set_name, say=_say)
        return 0
    if args.command == "pool":
        from edge_rag import pool as pool_module

        body = pool_module.run(args.set_name, say=_say)
        entry = body["equality"]["fusion_equals_p10-c"]
        return 0 if entry["equal"] == entry["of"] else 1
    if args.command == "score":
        from edge_rag import scoring
        from edge_rag.artifacts import write_json

        scored = scoring.run(args.set_name, args.rankings, args.against)
        _report(scored)
        if args.out is not None:
            _say(f"result written to {write_json(args.out, scored)}")
        return 0
    if args.command == "old-refs":
        from edge_rag import scoring

        for name, digest in scoring.old_references(args.set_name).items():
            _say(f"[{args.set_name}] {name}.jsonl.gz sha256 {digest}")
        return 0
    if args.command == "results":
        from edge_rag import results

        table = results.run(args.set_names or sorted(config.OLD_REFERENCES))
        for entry in table["sets"]:
            _say(f"[{entry['set']}] systems {', '.join(entry['systems'])}")
        return 0
    if args.command == "fusion-results":
        from edge_rag import fusion_results

        if args.from_json:
            _say(f"results.md regenerates byte-equal, sha256 {fusion_results.regenerate()}")
            return 0
        table = fusion_results.run(sorted(config.SETS))
        outcome = table["outcome"]
        _say(f"F3 verdict: {outcome['f3_verdict']}; exam entrant: {outcome['exam_entrant']}")
        for set_name, reason in table["not_run"].items():
            _say(f"[{set_name}] not run ({reason})")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
