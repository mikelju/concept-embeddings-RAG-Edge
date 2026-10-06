"""`uv run edge-rag <command>`: one generic runner, ASCII-only output."""

import argparse
import json
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
    converge = commands.add_parser("converge", help="Phase 05 multi-seed hop and its fusions")
    converge.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    pairs = commands.add_parser("judge-pairs", help="Phase 04 uncached judge pairs per set")
    pairs.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    commands.add_parser("rivals-subsample", help="Phase 06 HotpotQA 1,000-qid subsample")
    rivals_pairs = commands.add_parser("rivals-pairs", help="Phase 06 G-R2 pairs per set")
    rivals_pairs.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    rivals_rank = commands.add_parser("rivals-rank", help="Phase 06 G-R2 lists from pod scores")
    rivals_rank.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    ga2_export = commands.add_parser("ga2-export", help="Phase 06 G-A2 MultiHop-RAG bundle")
    ga2_export.add_argument("--out", type=Path, default=config.PHASE06_DIR / "ga2")
    ga2_rank = commands.add_parser("ga2-rank", help="Phase 06 G-A2 lists from the pod output")
    ga2_rank.add_argument("--bundle", type=Path, default=config.PHASE06_DIR / "ga2")
    ga2_rank.add_argument("--pod", type=Path, default=config.PHASE06_DIR / "ga2" / "pod")
    ga2_rank.add_argument("--rankings", type=Path, default=config.PHASE06_RANKINGS_DIR)
    ga1_export = commands.add_parser("ga1-export", help="Phase 06 G-A1 HotpotQA 1,000 bundle")
    ga1_export.add_argument("--out", type=Path, default=config.PHASE06_DIR / "ga1")
    ga1_rank = commands.add_parser("ga1-rank", help="Phase 06 G-A1 lists from the pod output")
    ga1_rank.add_argument("--bundle", type=Path, default=config.PHASE06_DIR / "ga1")
    ga1_rank.add_argument("--pod", type=Path, default=config.PHASE06_DIR / "ga1" / "pod")
    ga1_rank.add_argument("--rankings", type=Path, default=config.PHASE06_RANKINGS_DIR)
    judge_cmd = commands.add_parser("judge", help="Phase 04 j-rrf3 and j-rrf4 lists")
    judge_cmd.add_argument("--set", dest="set_name", required=True, choices=sorted(config.SETS))
    judge_results_cmd = commands.add_parser("judge-results", help="write the Phase 04 results")
    judge_results_cmd.add_argument(
        "--from-json", action="store_true", help="only check results.md regenerates byte-equal"
    )
    judge_results_cmd.add_argument(
        "--out", type=Path, help="write results.json and results.md into this folder instead"
    )
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
    fusion_cmd.add_argument(
        "--out", type=Path, help="write results.json and results.md into this folder instead"
    )
    hop_cmd = commands.add_parser("hop-results", help="write the Phase 05 results table")
    hop_cmd.add_argument(
        "--from-json", action="store_true", help="only check results.md regenerates byte-equal"
    )
    hop_cmd.add_argument(
        "--out", type=Path, help="write results.json and results.md into this folder instead"
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
    if args.command == "converge":
        from edge_rag import converge as converge_module

        body = converge_module.run(args.set_name, say=_say)
        equal = body["equality"].values()
        return 0 if all(entry["equal"] == entry["of"] for entry in equal) else 1
    if args.command == "judge-pairs":
        from edge_rag import judge

        body = judge.run_pairs(args.set_name, say=_say)
        check = body["question_text_check"]
        return 0 if check["equal"] == check["of"] else 1
    if args.command in ("rivals-subsample", "rivals-pairs", "rivals-rank"):
        from edge_rag import rivals

        if args.command == "rivals-subsample":
            rivals.run_subsample(say=_say)
        elif args.command == "rivals-pairs":
            rivals.run_pairs(args.set_name, say=_say)
        else:
            rivals.run_rank(args.set_name, say=_say)
        return 0
    if args.command in ("ga2-export", "ga2-rank"):
        from edge_rag import ga2

        if args.command == "ga2-export":
            ga2.run_export(args.out, say=_say)
        else:
            ga2.run_rank(args.bundle, args.pod, args.rankings, say=_say)
        return 0
    if args.command in ("ga1-export", "ga1-rank"):
        from edge_rag import ga1_hotpot

        if args.command == "ga1-export":
            ga1_hotpot.run_export(args.out, say=_say)
        else:
            ga1_hotpot.run_rank(args.bundle, args.pod, args.rankings, say=_say)
        return 0
    if args.command == "judge-results":
        from edge_rag.phase_results import Phase04

        if args.from_json:
            _say(f"results.md regenerates byte-equal, sha256 {Phase04.regenerate()}")
            return 0
        _say(json.dumps(Phase04.run(sorted(config.SETS), args.out)["outcome"], indent=2))
        return 0
    if args.command == "hop-results":
        from edge_rag.phase_results import Phase05

        if args.from_json:
            _say(f"results.md regenerates byte-equal, sha256 {Phase05.regenerate()}")
            return 0
        _say(json.dumps(Phase05.run(sorted(config.SETS), args.out)["outcome"], indent=2))
        return 0
    if args.command == "judge":
        from edge_rag import judge

        counts = judge.run_judge(args.set_name, say=_say)["counts"]
        whole = all(n == counts["questions"] for n in counts["permutation_of_pool"].values())
        return 0 if whole else 1
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
        from edge_rag.phase_results import Phase03

        if args.from_json:
            _say(f"results.md regenerates byte-equal, sha256 {Phase03.regenerate()}")
            return 0
        table = Phase03.run(sorted(config.SETS), args.out)
        outcome = table["outcome"]
        _say(f"F3 verdict: {outcome['f3_verdict']}; exam entrant: {outcome['exam_entrant']}")
        for set_name, reason in table["not_run"].items():
            _say(f"[{set_name}] not run ({reason})")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
