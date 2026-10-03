"""Phase 03 results table (spec C4-C6): RRF3 and F3 against G-L and the best systems so far.

Every figure is read from run files: Phase 03 rankings and manifests, Phase 02 `results.json`
and ranking files (read only, each checked against the sha256 recorded there). States, verdicts
and the class check are written here by code under the spec's frozen selection rule.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from edge_rag import components, config, fuse, metrics, scoring
from edge_rag.artifacts import ArtifactError, Checks, OldData, write_bytes, write_json

PHASE02_RESULTS = config.DATA_DIR / "phase02" / "results.json"
RESULTS_JSON = config.PHASE03_DIR / "results.json"
RESULTS_MD = config.REPO_ROOT / "docs" / "plans" / "fase-03-fusion" / "results.md"
LIGHT = ("p10-a", "p10-b", "p10-c", "p14", "g-l")
CANDIDATES = ("rrf3", "f3")
ALPHA = 0.05
LAPTOP_BOUND, GPU_BOUND = 2.0, 0.1
NOT_RUN = "not run"
# Costs the handover records for the old references, with their source (spec C4).
HANDOVER = "successor_project_handover.md, cost table"
DENSE_EMBEDDING = {
    "multihop-rag": "BGE-small encoding 41.20 s on the old pod, inside 0.0404 USD attributable "
    "to GLiNER + BGE (16.results.md section 3)",
}
GLINER = {
    "hotpotqa-dev": "GLiNER 6.18 h, 4.57 USD attributable (derived; research_summary.md point 2)",
    "musique": "GLiNER 10.3 min, 0.13 USD attributable (research_summary.md point 7)",
    "multihop-rag": "GLiNER 154.77 s extraction, BGE 41.20 s encoding, 0.0404 USD attributable "
    "(16.results.md section 3)",
}
JUDGE = "J-strong 1.1576 USD attributable over the three sets (derived; 17.results.md section 4)"


def state(test: Mapping[str, Any] | None) -> str:
    if test is None:
        return NOT_RUN
    if test["wins"] > test["losses"] and test["p"] < ALPHA:
        return "win"
    if test["losses"] > test["wins"] and test["p"] < ALPHA:
        return "loss"
    return "tie"


def advances(against_g_l: Sequence[str]) -> bool:
    return against_g_l.count("win") >= 2 and "loss" not in against_g_l


def verdict(
    against_g_l: Sequence[str], against_rrf3: Sequence[str], against_best_light: Sequence[str]
) -> str:
    """The selection rule: states per set (`not run` counts as neither a win nor a loss)."""
    if not advances(against_g_l):
        return "does not advance"
    failed = []
    if "win" not in against_rrf3:
        failed.append("no win against RRF3")
    if "loss" in against_rrf3:
        failed.append("a loss to RRF3")
    if "loss" in against_best_light:
        failed.append("a loss to the best light-class system so far")
    return "entrant" if not failed else f"advances, no entrant ({'; '.join(failed)})"


def handover_cost(set_name: str, system: str) -> str | None:
    if system in ("p10-a", "p10-b"):
        return DENSE_EMBEDDING.get(set_name)
    if system in ("p10-c", "p14"):
        return GLINER[set_name]
    if system in ("j-p10b", "j-union"):
        return JUDGE
    return None


def candidate_cost(
    system: str,
    comp: Mapping[str, Any],
    fused: Mapping[str, Any],
    g_l_cost: Mapping[str, Any],
    gpu: str,
    set_name: str,
) -> dict[str, Any]:
    """Each component on its own hardware, the per-hardware totals and the class check."""
    laptop = comp["hardware"]
    dense_note = DENSE_EMBEDDING.get(set_name, "not recorded")
    measured = "measured (Phase 03 manifests)"
    pod = "seconds measured (Phase 02 G-L manifest); USD derived (time x rate)"
    rows: list[dict[str, Any]] = [
        {"component": "BM25 build", "kind": "offline", "hardware": laptop,
         "seconds": comp["offline_seconds"]["bm25_build"], "usd": 0.0, "label": measured},
        {"component": "Dense corpus embeddings", "kind": "offline", "hardware": "old pod",
         "seconds": None, "usd": None,
         "label": f"not measured in this phase; handover: {dense_note}"},
        {"component": "G-L encode and index", "kind": "offline", "hardware": gpu,
         "seconds": g_l_cost["offline_seconds"], "usd": g_l_cost["offline_usd"], "label": pod},
        {"component": "BGE-small question encoding", "kind": "online", "hardware": laptop,
         "seconds": comp["question_encoding"]["seconds_per_question"], "usd": 0.0,
         "label": f"measured on the first {comp['question_encoding']['questions']} questions"},
        {"component": "Dense retrieval", "kind": "online", "hardware": laptop,
         "seconds": comp["retrieval_seconds_per_question"]["dense"], "usd": 0.0, "label": measured},
        {"component": "BM25 retrieval", "kind": "online", "hardware": laptop,
         "seconds": comp["retrieval_seconds_per_question"]["bm25"], "usd": 0.0, "label": measured},
        {"component": "G-L search", "kind": "online", "hardware": gpu,
         "seconds": g_l_cost["online_seconds_per_question"],
         "usd": g_l_cost["online_usd_per_question"], "label": pod},
        {"component": "RRF", "kind": "online", "hardware": laptop,
         "seconds": fused["online_seconds_per_question"]["rrf"], "usd": 0.0, "label": measured},
    ]  # fmt: skip
    if system == "f3":
        rows.insert(
            1,
            {"component": "Boilerplate flags", "kind": "offline", "hardware": laptop,
             "seconds": fused["offline_seconds"]["boilerplate_flags"], "usd": 0.0,
             "label": measured},
        )  # fmt: skip
        rows.append(
            {"component": "Diversity pass", "kind": "online", "hardware": laptop,
             "seconds": fused["online_seconds_per_question"]["diversity"], "usd": 0.0,
             "label": measured},
        )  # fmt: skip
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["seconds"] is None:
            continue
        entry = totals.setdefault(f"{row['kind']} {row['hardware']}", {"seconds": 0.0, "usd": 0.0})
        entry["seconds"] += row["seconds"]
        entry["usd"] += row["usd"]
    laptop_online = totals[f"online {laptop}"]["seconds"]
    gpu_online = totals[f"online {gpu}"]["seconds"]
    return {
        "components": rows,
        "totals_per_hardware": totals,
        "totals_label": "derived (sums of the components on one hardware; the Dense corpus "
        "embeddings are not in the offline totals)",
        "cross_hardware_online_seconds_per_question": laptop_online + gpu_online,
        "cross_hardware_label": "derived, context only: laptop CPU and GPU seconds summed",
        "class_check": {
            "laptop_online_seconds_per_question": laptop_online,
            "laptop_bound": LAPTOP_BOUND,
            "laptop_inside": laptop_online <= LAPTOP_BOUND,
            "gpu": gpu,
            "gpu_online_seconds_per_question": gpu_online,
            "gpu_bound": GPU_BOUND,
            "gpu_bound_stated_for": "one RTX 4090",
            "gpu_inside": gpu_online <= GPU_BOUND,
            "inside_light_class": laptop_online <= LAPTOP_BOUND and gpu_online <= GPU_BOUND,
            "label": "derived",
        },
    }


def _fs(row: Mapping[str, Any]) -> int:
    return int(row["metrics"]["full_support_at_budget"][str(config.BUDGET)])


def set_table(set_name: str, phase02: Mapping[str, Any]) -> dict[str, Any]:
    spec = config.SETS[set_name]
    checks = Checks()
    old = OldData(spec.directory, spec.file_sha256, checks)
    questions = scoring.questions_of(old, checks, spec)
    counts = scoring.token_counts(old, checks, spec)
    here = config.PHASE03_RANKINGS_DIR / set_name
    there = config.RANKINGS_DIR / set_name
    comp = json.loads((here / components.MANIFEST).read_text("utf-8"))
    fused = json.loads((here / fuse.MANIFEST).read_text("utf-8"))
    old_rows = phase02["systems"]
    best_light = max(LIGHT, key=lambda s: _fs(old_rows[s]))
    best = max(old_rows, key=lambda s: _fs(old_rows[s]))
    if old_rows["g-l"]["rankings_sha256"] != config.GL_SHA256[set_name]:
        raise ArtifactError(f"{set_name}: Phase 02 results name another G-L ranking file")
    rows: dict[str, Any] = {}
    records: dict[str, list[int]] = {}
    for system in (*CANDIDATES, "g-l", best_light, best):
        if system in rows:
            continue
        path = (here if system in CANDIDATES else there) / f"{system}.jsonl.gz"
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded = (
            fused["outputs_sha256"][path.name]
            if system in CANDIDATES
            else old_rows[system]["rankings_sha256"]
        )
        if sha256 != recorded:
            raise ArtifactError(f"{path}: sha256 {sha256}, recorded {recorded}")
        summary, records[system] = scoring.score(questions, scoring.read_rankings(path), counts)
        if system not in CANDIDATES and summary != old_rows[system]["metrics"]:
            raise ArtifactError(f"{set_name} {system}: rescored metrics differ from Phase 02")
        rows[system] = {"rankings_sha256": sha256, "metrics": summary, "metrics_label": "measured"}
    gpu = json.loads((there / "g-l.manifest.json").read_text("utf-8"))["gpu"]
    for system in CANDIDATES:
        rows[system]["cost"] = candidate_cost(
            system, comp, fused, old_rows["g-l"]["cost"], gpu, set_name
        )
    for system in rows:
        if system in CANDIDATES:
            continue
        cost = dict(old_rows[system]["cost"])
        if system == "g-l":
            cost["hardware"] = gpu
        note = handover_cost(set_name, system)
        if note is not None:
            cost["handover"] = {"figure": note, "source": HANDOVER}
        rows[system]["cost"] = cost
    pairs = [
        ("f3", "g-l"),
        ("rrf3", "g-l"),
        ("f3", "rrf3"),
        ("f3", best_light),
        ("rrf3", best_light),
        ("f3", best),
        ("rrf3", best),
    ]
    paired = []
    for system, against in pairs:
        test = metrics.paired(records[against], records[system])
        paired.append(
            {
                "system": system,
                "against": against,
                **test,
                "state": state(test),
                "label": "measured",
            }
        )
    return {
        "set": set_name,
        "n": len(questions),
        "best_light_class_so_far": best_light,
        "best_so_far": best,
        "systems": rows,
        f"paired_full_support_at_{config.BUDGET}": paired,
        "boilerplate_units": fused["boilerplate_units"],
        "demotions_per_rule": fused["demotions_per_rule"],
        "exploratory_gold_diagnostics": fused["exploratory_gold_diagnostics"],
        "peak_rss_mb": {"components": comp["peak_rss_mb"], "fuse": fused["peak_rss_mb"]},
        "manifests_sha256": {
            name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in (components.MANIFEST, fuse.MANIFEST)
        },
        "digests_checked": checks.records,
    }


def outcome(tables: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """States per set for each bar, F3's verdict and RRF3's context verdict."""

    def states(system: str, against: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for set_name, table in tables.items():
            test = None
            if table is not None:
                target = table.get(against, against)
                test = next(
                    t
                    for t in table[f"paired_full_support_at_{config.BUDGET}"]
                    if t["system"] == system and t["against"] == target
                )
            found[set_name] = state(test)
        return found

    f3_g_l = states("f3", "g-l")
    f3_rrf3 = states("f3", "rrf3")
    f3_light = states("f3", "best_light_class_so_far")
    rrf3_g_l = states("rrf3", "g-l")
    f3_verdict = verdict(list(f3_g_l.values()), list(f3_rrf3.values()), list(f3_light.values()))
    return {
        "states": {
            "f3 vs g-l": f3_g_l,
            "f3 vs rrf3": f3_rrf3,
            "f3 vs best light-class so far": f3_light,
            "f3 vs best so far": states("f3", "best_so_far"),
            "rrf3 vs g-l": rrf3_g_l,
            "rrf3 vs best light-class so far": states("rrf3", "best_light_class_so_far"),
            "rrf3 vs best so far": states("rrf3", "best_so_far"),
        },
        "f3_verdict": f3_verdict,
        "rrf3_context_verdict": "advances" if advances(list(rrf3_g_l.values())) else
        "does not advance",
        "exam_entrant": "F3" if f3_verdict == "entrant" else "none from this phase",
        "label": "written by code under the frozen selection rule",
    }  # fmt: skip


def _num(value: Any, digits: str) -> str:
    return "-" if value is None else format(value, digits)


def markdown(table: Mapping[str, Any]) -> str:
    budget = str(config.BUDGET)
    out = table["outcome"]
    lines = [
        "# Phase 03 - Results",
        "",
        "Generated by `uv run edge-rag fusion-results` from `data/phase03/results.json`; "
        "do not edit by hand.",
        "Metrics and paired tests are measured; states, verdicts and the class check are "
        "written by code; per-hardware totals are derived.",
        "",
        "## Outcome",
        "",
        f"- F3 verdict: **{out['f3_verdict']}**.",
        f"- Exam entrant: {out['exam_entrant']}.",
        f"- RRF3 against G-L under the same advance rule (context): {out['rrf3_context_verdict']}.",
        "",
        "| Comparison | " + " | ".join(out["states"]["f3 vs g-l"]) + " |",
        "|---|" + "---|" * len(out["states"]["f3 vs g-l"]),
    ]
    for name, states in out["states"].items():
        lines.append(f"| {name} | " + " | ".join(states.values()) + " |")
    lines.append("")
    for set_name, reason in table["not_run"].items():
        lines.append(f"- {set_name}: {NOT_RUN} ({reason}).")
    for entry in table["sets"]:
        lines += [
            "",
            f"## {entry['set']} (n = {entry['n']})",
            "",
            f"Best light-class system so far: {entry['best_light_class_so_far']}; "
            f"best system so far: {entry['best_so_far']} (by FS@{budget}, Phase 02).",
            "",
            "| System | FS@1,024 | FS@2,048 | FS@4,096 | FS@2 | FS@5 | FS@20 | Share@5 "
            "| R@2 | R@5 | R@10 | R@20 | R@100 | nDCG@10 |",
            "|---|" + "---:|" * 13,
        ]
        for system, row in entry["systems"].items():
            m = row["metrics"]
            fs, at_k, r = m["full_support_at_budget"], m["full_support_at_k"], m["recall_at_k"]
            recalls = " | ".join(f"{r[k]:.4f}" for k in ("2", "5", "10", "20", "100"))
            lines.append(
                f"| {system} | {fs['1024']} | {fs['2048']} | {fs['4096']} | {at_k['2']} "
                f"| {at_k['5']} | {at_k['20']} | {m['gold_share_at_5']:.4f} | {recalls} "
                f"| {m['ndcg_at_10']:.4f} |"
            )
        lines += [
            "",
            f"Paired exact McNemar on FS@{budget}, alpha 0.05 per comparison, no correction:",
            "",
            "| System | Against | Wins | Losses | Ties | Exact p | State |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for test in entry[f"paired_full_support_at_{budget}"]:
            lines.append(
                f"| {test['system']} | {test['against']} | {test['wins']} | {test['losses']} "
                f"| {test['ties']} | {test['p']:.4g} | {test['state']} |"
            )
        lines += ["", "Cost per component (seconds per question online, per corpus offline):", ""]
        lines += ["| System | Component | Kind | Hardware | Seconds | USD | Label |"]
        lines += ["|---|---|---|---|---:|---:|---|"]
        for system in CANDIDATES:
            for c in entry["systems"][system]["cost"]["components"]:
                lines.append(
                    f"| {system} | {c['component']} | {c['kind']} | {c['hardware']} "
                    f"| {_num(c['seconds'], '.6g')} | {_num(c['usd'], '.6f')} | {c['label']} |"
                )
        lines += ["", "Totals per hardware (derived):", ""]
        for system in CANDIDATES:
            cost = entry["systems"][system]["cost"]
            for name, total in cost["totals_per_hardware"].items():
                lines.append(
                    f"- {system}, {name}: {total['seconds']:.6g} s, {total['usd']:.6f} USD."
                )
            check = cost["class_check"]
            where = (
                "inside the light class on this set"
                if check["inside_light_class"]
                else "outside the light class on this set"
            )
            laptop = check["laptop_online_seconds_per_question"]
            lines.append(
                f"- {system} class check: laptop online {laptop:.4g}"
                f" s/q (bound {check['laptop_bound']}), {check['gpu']} online "
                f"{check['gpu_online_seconds_per_question']:.4g} s/q (bound {check['gpu_bound']}, "
                f"stated for {check['gpu_bound_stated_for']}): {where} (derived)."
            )
            lines.append(
                f"- {system} cross-hardware online total, context only: "
                f"{cost['cross_hardware_online_seconds_per_question']:.4g} s/q (derived)."
            )
        lines += ["", "Cost of the reference rows (inherited labels):", ""]
        for system, row in entry["systems"].items():
            if system in CANDIDATES:
                continue
            cost = row["cost"]
            text = cost.get("label", "")
            if "online_seconds_per_question" in cost:
                text = (
                    f"offline {cost['offline_seconds']:.0f} s, {cost['offline_usd']:.3f} USD; "
                    f"online {cost['online_seconds_per_question']:.4f} s/q, "
                    f"{cost['online_usd_per_question']:.6f} USD/q on {cost.get('hardware', 'pod')} "
                    f"({cost['label']})"
                )
            if "handover" in cost:
                text += f"; handover: {cost['handover']['figure']}"
            lines.append(f"- {system}: {text}.")
        diag = entry["exploratory_gold_diagnostics"]
        lines += [
            "",
            f"Diversity pass: {entry['boilerplate_units']} boilerplate units in the corpus; "
            "demotions over all questions: "
            + ", ".join(f"{rule} {n}" for rule, n in entry["demotions_per_rule"].items())
            + " (measured).",
            "",
            "Gold-informed diagnostics (exploratory, decide nothing): questions where a gold unit "
            f"demoted by the rule was in RRF3's FS@{budget} context but not in F3's, and in "
            "RRF3's top 100 but not in F3's:",
            "",
            f"| Rule | Lost from the FS@{budget} context | Lost from the top 100 |",
            "|---|---:|---:|",
        ]
        for rule, found in diag["questions_with_a_gold_unit_demoted"].items():
            lines.append(f"| {rule} | {found['in_context_at_budget']} | {found['in_top_100']} |")
        lines += [
            "",
            f"Gold units flagged boilerplate: {diag['gold_units_flagged_boilerplate']} of "
            f"{diag['gold_units']} (exploratory).",
            f"Peak RSS: components {entry['peak_rss_mb']['components']} MB, fuse "
            f"{entry['peak_rss_mb']['fuse']} MB (measured).",
        ]
    return "\n".join(lines) + "\n"


def run(set_names: Sequence[str]) -> dict[str, Any]:
    phase02 = {e["set"]: e for e in json.loads(PHASE02_RESULTS.read_text("utf-8"))["sets"]}
    tables: dict[str, dict[str, Any] | None] = {}
    not_run: dict[str, str] = {}
    for set_name in set_names:
        if not (config.PHASE03_RANKINGS_DIR / set_name / fuse.MANIFEST).exists():
            tables[set_name] = None
            not_run[set_name] = "no Phase 03 fuse manifest"
            continue
        tables[set_name] = set_table(set_name, phase02[set_name])
    table = {
        "phase02_results_sha256": hashlib.sha256(PHASE02_RESULTS.read_bytes()).hexdigest(),
        "outcome": outcome(tables),
        "not_run": not_run,
        "sets": [t for t in tables.values() if t is not None],
    }
    write_json(RESULTS_JSON, table)
    write_bytes(RESULTS_MD, markdown(table).encode("utf-8"))
    return table
