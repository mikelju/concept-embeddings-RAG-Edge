"""Phase 02 results table (C6): every system scored per set, with labels, costs and paired tests."""

import hashlib
import json
from pathlib import Path
from typing import Any

from edge_rag import config, metrics, scoring
from edge_rag.artifacts import Checks, OldData, write_bytes, write_json

OLD_SYSTEMS = ("p10-a", "p10-b", "p10-c", "p14", "j-p10b", "j-union")
GHOSTS = ("g-l", "g-r", "g-a1", "g-r2")
OLD_COST = "not measured in this phase (old project)"
PHASE_DIR = config.RANKINGS_DIR.parent
# G-A1 searched the session 4 G-L index, not the reported session 2 build (plan F7):
# its offline cost and provenance come from that index's G-L manifest.
GHOST_INDEX = {"g-a1": PHASE_DIR / "session4" / "s4A" / "rankings"}
# G-L rebuilds kept beside the reported build (plan F5, F7): build label -> ranking file.
G_L_REBUILDS = {
    "multihop-rag": {
        "session 3": "session3/rankings/multihop-rag/g-l.jsonl.gz",
        "session 4": "session4/s4A/rankings/multihop-rag/g-l.jsonl.gz",
        "session 4 same-host rebuild": "session4/s4B/g-l.jsonl.gz",
    },
    "musique": {
        "session 3": "session3/rankings/musique/g-l.jsonl.gz",
        "session 4": "session4/s4A/rankings/musique/g-l.jsonl.gz",
    },
}


def _manifest(directory: Path, system: str) -> dict[str, Any] | None:
    path = directory / f"{system}.manifest.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def _usd(seconds: float, rate: float) -> float:
    return seconds * rate / 3600


def ghost_cost(system: str, manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Offline (G-L encode + index, shared by every ghost on its index) and online seconds per
    question, each with USD as time x the pod's costPerHr (derived)."""
    g_l = manifests["g-l"]
    offline = g_l["seconds"]["encode"] + g_l["seconds"]["index"]
    online = g_l["search_seconds_per_question"]
    if system in ("g-r", "g-r2"):
        online += manifests[system]["rerank_seconds_per_question"]
    elif system == "g-a1":
        online = manifests[system]["seconds_per_question"]  # retrieval calls included
    rate = manifests[system]["cost_per_hr_usd"]
    return {
        "offline_seconds": offline,
        "offline_usd": _usd(offline, rate),
        "online_seconds_per_question": online,
        "online_usd_per_question": _usd(online, rate),
        "cost_per_hr_usd": rate,
        "label": "seconds measured (pod manifests); USD derived (time x rate)",
    }


def set_table(set_name: str) -> dict[str, Any]:
    spec = config.SETS[set_name]
    checks = Checks()
    old = OldData(spec.directory, spec.file_sha256, checks)
    questions = scoring.questions_of(old, checks, spec)
    counts = scoring.token_counts(old, checks, spec)
    directory = config.RANKINGS_DIR / set_name
    rows: dict[str, Any] = {}
    records: dict[str, list[int]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for system in OLD_SYSTEMS + GHOSTS:
        path = directory / f"{system}.jsonl.gz"
        if not path.exists():
            continue
        summary, records[system] = scoring.score(questions, scoring.read_rankings(path), counts)
        row: dict[str, Any] = {
            "rankings_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metrics": summary,
            "metrics_label": "measured",
        }
        manifest = _manifest(directory, system)
        if manifest is not None:
            manifests[system] = manifest
            row["model"] = manifest.get("model")
            row["revision"] = manifest.get("revision")
            row["git_commit"] = manifest.get("git_commit")
        rows[system] = row
    for system in rows:
        if system in GHOST_INDEX:
            index_dir = GHOST_INDEX[system] / set_name
            index_manifest = _manifest(index_dir, "g-l")
            if index_manifest is None:
                raise FileNotFoundError(f"G-L manifest of the index {system} searched: {index_dir}")
            rows[system]["searched_g_l_index"] = {
                "manifest": (index_dir / "g-l.manifest.json").relative_to(PHASE_DIR).as_posix(),
                "manifest_sha256": hashlib.sha256(
                    (index_dir / "g-l.manifest.json").read_bytes()
                ).hexdigest(),
            }
            rows[system]["cost"] = ghost_cost(system, {**manifests, "g-l": index_manifest})
        elif system in GHOSTS:
            rows[system]["cost"] = ghost_cost(system, manifests)
        else:
            rows[system]["cost"] = {"label": OLD_COST}
    budget = str(config.BUDGET)
    old_present = [s for s in OLD_SYSTEMS if s in rows]
    best_old = max(
        old_present, key=lambda s: rows[s]["metrics"]["full_support_at_budget"][budget]
    )
    ghosts = [s for s in GHOSTS if s in rows]
    paired = []
    for i, ghost in enumerate(ghosts):
        for base in [best_old, *ghosts[:i]]:
            test = metrics.paired(records[base], records[ghost])
            paired.append({"system": ghost, "against": base, **test, "label": "measured"})
    spread = []
    for build, relative in G_L_REBUILDS.get(set_name, {}).items():
        path = PHASE_DIR / relative
        summary, record = scoring.score(questions, scoring.read_rankings(path), counts)
        test = metrics.paired(records["g-l"], record)
        spread.append(
            {
                "build": build,
                "rankings": relative,
                "rankings_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "full_support_at_budget": summary["full_support_at_budget"][budget],
                "against_reported_g_l": {**test, "label": "measured"},
            }
        )
    return {
        "set": set_name,
        "n": len(questions),
        "best_old_by_fs_at_budget": best_old,
        "systems": rows,
        f"paired_full_support_at_{budget}": paired,
        f"g_l_build_spread_at_{budget}": spread,
        "digests_checked": checks.records,
    }


def markdown(table: dict[str, Any]) -> str:
    budget = str(config.BUDGET)
    lines = [
        "# Phase 02 - Results",
        "",
        "Generated by `uv run edge-rag results` from `results.json`; do not edit by hand.",
        "Metrics are measured; seconds are measured on the pod; USD is time x rate (derived).",
        "Offline cost is G-L's encode and index, shared by every ghost that searches its index.",
        "G-L and G-R are the session 2 builds; G-A1 searched the session 4 G-L index (plan F7),",
        "and its offline cost is that index's.",
        "",
    ]
    for entry in table["sets"]:
        lines += [
            f"## {entry['set']} (n = {entry['n']})",
            "",
            "| System | FS@1,024 | FS@2,048 | FS@4,096 | FS@2 | FS@5 | FS@20 | R@5 | R@100 "
            "| nDCG@10 | Offline s | Offline USD | Online s/q | Online USD/q |",
            "|---|" + "---:|" * 13,
        ]
        for system, row in entry["systems"].items():
            m, c = row["metrics"], row["cost"]
            fs, at_k = m["full_support_at_budget"], m["full_support_at_k"]
            cost = (
                f"{c['offline_seconds']:.0f} | {c['offline_usd']:.3f} | "
                f"{c['online_seconds_per_question']:.4f} | {c['online_usd_per_question']:.6f}"
                if "offline_seconds" in c
                else "- | - | - | -"
            )
            lines.append(
                f"| {system} | {fs['1024']} | {fs['2048']} | {fs['4096']} | {at_k['2']} "
                f"| {at_k['5']} | {at_k['20']} | {m['recall_at_k']['5']:.4f} "
                f"| {m['recall_at_k']['100']:.4f} | {m['ndcg_at_10']:.4f} | {cost} |"
            )
        lines += [
            "",
            f"Paired exact McNemar on FS@{budget} (best old system by FS@{budget}: "
            f"{entry['best_old_by_fs_at_budget']}):",
            "",
            "| System | Against | Wins | Losses | Ties | Exact p |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for test in entry[f"paired_full_support_at_{budget}"]:
            lines.append(
                f"| {test['system']} | {test['against']} | {test['wins']} | {test['losses']} "
                f"| {test['ties']} | {test['p']:.4g} |"
            )
        spread = entry[f"g_l_build_spread_at_{budget}"]
        if spread:
            reported = entry["systems"]["g-l"]["metrics"]["full_support_at_budget"][budget]
            lines += [
                "",
                f"G-L build-to-build spread on FS@{budget} (plan F5, F7), each rebuild paired "
                f"against the reported session 2 build ({reported}):",
                "",
                f"| Build | FS@{budget} | Wins | Losses | Ties | Exact p |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for build in spread:
                test = build["against_reported_g_l"]
                lines.append(
                    f"| {build['build']} | {build['full_support_at_budget']} | {test['wins']} "
                    f"| {test['losses']} | {test['ties']} | {test['p']:.4g} |"
                )
        lines.append("")
    return "\n".join(lines)


def run(set_names: list[str]) -> dict[str, Any]:
    table = {"phase": "02", "sets": [set_table(name) for name in set_names]}
    out = config.RANKINGS_DIR.parent
    write_json(out / "results.json", table)
    write_bytes(
        config.REPO_ROOT / "docs" / "plans" / "fase-02-ghosts" / "results.md",
        markdown(table).encode("utf-8"),
    )
    return table
