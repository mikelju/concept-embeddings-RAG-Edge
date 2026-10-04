"""Phase 03 results table (spec C4-C6): RRF3 and F3 against G-L and the best systems so far.

Every figure is read from run files: Phase 03 rankings and manifests, Phase 02 `results.json`
and ranking files (read only, each checked against the sha256 recorded there). States, verdicts
and the class check are written here by code under the spec's frozen selection rule.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from edge_rag import components, config, fuse, metrics, scoring
from edge_rag.artifacts import ArtifactError, Checks, OldData, write_bytes
from edge_rag.pod.common import git_provenance
from edge_rag.retrieval.rrf import RULES

PHASE02_RESULTS = config.DATA_DIR / "phase02" / "results.json"
RESULTS_JSON = config.PHASE03_DIR / "results.json"
RESULTS_MD = config.REPO_ROOT / "docs" / "plans" / "fase-03-fusion" / "results.md"
LIGHT = ("p10-a", "p10-b", "p10-c", "p14", "g-l")
CANDIDATES = ("rrf3", "f3")
ALPHA = 0.05
LAPTOP_BOUND, GPU_BOUND = 2.0, 0.1
NOT_RUN = "not run"
STATE_NAMES = (
    "f3 vs g-l",
    "f3 vs rrf3",
    "f3 vs best light-class so far",
    "f3 vs best so far",
    "rrf3 vs g-l",
    "rrf3 vs best light-class so far",
    "rrf3 vs best so far",
)
LAPTOP_USD = (
    "seconds measured (Phase 03 manifests); USD 0 by assumption (owned laptop, energy not counted)"
)
# Costs the handover records for the old references, each with its label and source (spec C4).
COST_TABLE = "successor_project_handover.md, cost table"
DENSE_EMBEDDING = {
    "multihop-rag": [
        (
            "BGE-small encoding 41.20 s on the old pod",
            "measured (recorded time)",
            f"{COST_TABLE} (16.results.md section 3)",
        ),
        (
            "0.0404 USD attributable to GLiNER + BGE, the BGE encoding inside it",
            "derived (time x rate)",
            f"{COST_TABLE} (16.results.md section 3)",
        ),
        (
            "the session invoiced 0.165 USD",
            "measured (invoice)",
            f"{COST_TABLE} (16.results.md section 3)",
        ),
        (
            "the session's balance delta 0.1029 USD",
            "measured (balance delta)",
            "successor_project_handover.md, RunPod know-how",
        ),
    ],
}
GLINER = {
    "hotpotqa-dev": [
        ("GLiNER over all 5,233,329 FullWiki paragraphs 6.18 h, 4.57 USD attributable",
         "derived (time x rate)", f"{COST_TABLE} (research_summary.md point 2)"),
        ("the whole old Phase 9 session invoiced 12.71 USD, GLiNER and other work",
         "measured (invoice)", f"{COST_TABLE} (research_summary.md point 2)"),
    ],
    "musique": [
        ("GLiNER 10.3 min, 0.13 USD attributable", "derived (time x rate)",
         f"{COST_TABLE} (research_summary.md point 7)"),
        ("the session invoiced 0.356 USD", "measured (invoice)",
         f"{COST_TABLE} (research_summary.md point 7)"),
    ],
    "multihop-rag": [
        ("GLiNER 154.77 s extraction, BGE 41.20 s encoding, 0.0404 USD attributable",
         "derived (time x rate)", f"{COST_TABLE} (16.results.md section 3)"),
        ("the session invoiced 0.165 USD", "measured (invoice)",
         f"{COST_TABLE} (16.results.md section 3)"),
        ("the session's balance delta 0.1029 USD", "measured (balance delta)",
         "successor_project_handover.md, RunPod know-how"),
    ],
}  # fmt: skip
JUDGE = [
    ("J-strong 1.1576 USD attributable over the three sets; the judge share only, without the "
     "pool it reranks", "derived (time x rate)",
     "successor_project_handover.md, judge proposal row (17.results.md section 4)"),
]  # fmt: skip
IN_SAMPLE = {"hotpotqa-dev": ("p10-c", "p14")}
SET_NOTES = {
    "hotpotqa-dev": [
        "HotpotQA dev is in-sample for the old P10-C and P14 fitted weights, so their rows here "
        "are an upper bound; BGE-small (Dense, in every fused list) was fine-tuned on HotpotQA "
        "train and answerai-colbert-small-v1 (G-L) is in-domain here (research protocol, "
        "known traps).",
    ],
}


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
    against_g_l: Sequence[str],
    against_rrf3: Sequence[str],
    against_best_light: Sequence[str],
    control: str = "RRF3",
    in_class: str = "light-class",
) -> str:
    """The selection rule: states per set (`not run` counts as neither a win nor a loss) against
    the gate, the control and the best in-class system so far; Phase 04 names its own."""
    if not advances(against_g_l):
        return "does not advance"
    failed = []
    if "win" not in against_rrf3:
        failed.append(f"no win against {control}")
    if "loss" in against_rrf3:
        failed.append(f"a loss to {control}")
    if "loss" in against_best_light:
        failed.append(f"a loss to the best {in_class} system so far")
    return "entrant" if not failed else f"advances, no entrant ({'; '.join(failed)})"


def handover_cost(set_name: str, system: str) -> list[dict[str, str]]:
    notes: list[tuple[str, str, str]] = []
    if system in ("p10-a", "p10-b") and set_name in DENSE_EMBEDDING:
        notes += DENSE_EMBEDDING[set_name]
    if system in ("p10-c", "p14"):
        notes += GLINER[set_name]
    if system in ("j-p10b", "j-union"):
        notes += JUDGE
    return [{"figure": f, "label": label, "source": source} for f, label, source in notes]


def shown(set_name: str, system: str) -> str:
    return f"{system} (in-sample)" if system in IN_SAMPLE.get(set_name, ()) else system


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
    dense_note = (
        "; ".join(f"{figure} ({label})" for figure, label, _ in DENSE_EMBEDDING[set_name])
        if set_name in DENSE_EMBEDDING
        else "not recorded"
    )
    measured = LAPTOP_USD
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
         "label": f"{LAPTOP_USD}; seconds on the first "
                  f"{comp['question_encoding']['questions']} questions"},
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
    usd = {
        kind: sum(r["usd"] for r in rows if r["kind"] == kind and r["usd"] is not None)
        for kind in ("offline", "online")
    }
    laptop_online = totals[f"online {laptop}"]["seconds"]
    gpu_online = totals[f"online {gpu}"]["seconds"]
    return {
        "components": rows,
        "totals_per_hardware": totals,
        "totals_label": "derived (sums of the components on one hardware; the Dense corpus "
        "embeddings are not in the offline totals)",
        "usd_total": {"offline": usd["offline"], "online_per_question": usd["online"]},
        "usd_total_label": "derived, lower bound: summed across hardware; the Dense corpus "
        "embeddings are not included and laptop USD is 0 by assumption",
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
    notes = list(SET_NOTES.get(set_name, []))
    for system in CANDIDATES:
        rows[system]["cost"] = candidate_cost(
            system, comp, fused, old_rows["g-l"]["cost"], gpu, set_name
        )
    for system in rows:
        if system in CANDIDATES:
            continue
        cost = dict(old_rows[system]["cost"])
        system_manifest = there / f"{system}.manifest.json"
        if system_manifest.exists():
            cost["hardware"] = json.loads(system_manifest.read_text("utf-8"))["gpu"]
        found = handover_cost(set_name, system)
        if found:
            cost["handover"] = found
        rows[system]["cost"] = cost
        if system in IN_SAMPLE.get(set_name, ()):
            rows[system]["in_sample"] = True
            notes.append(f"{system} is in-sample on this set: its comparisons favour it.")
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
        "demotions_per_rule_scope": fused["demotions_per_rule_scope"],
        "demotions_per_rule_in_rrf3_top_100": fused["demotions_per_rule_in_rrf3_top_100"],
        "notes": notes,
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


def _systems(entry: Mapping[str, Any]) -> list[str]:
    """The table's system order, fixed: candidates, G-L, best light-class, best (no repeats)."""
    order = (*CANDIDATES, "g-l", entry["best_light_class_so_far"], entry["best_so_far"])
    return list(dict.fromkeys(order))


def markdown(table: Mapping[str, Any]) -> str:
    """Every row in an explicit order, so the page renders the same from the in-memory table
    and from `results.json` as written (sorted keys)."""
    budget = str(config.BUDGET)
    out = table["outcome"]
    set_names = sorted(out["states"][STATE_NAMES[0]])
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
        "| Comparison | " + " | ".join(set_names) + " |",
        "|---|" + "---|" * len(set_names),
    ]
    for name in STATE_NAMES:
        states = out["states"][name]
        lines.append(f"| {name} | " + " | ".join(states[s] for s in set_names) + " |")
    lines.append("")
    for set_name in sorted(table["not_run"]):
        lines.append(f"- {set_name}: {NOT_RUN} ({table['not_run'][set_name]}).")
    for entry in sorted(table["sets"], key=lambda e: str(e["set"])):
        set_name = entry["set"]
        systems = _systems(entry)
        lines += [
            "",
            f"## {set_name} (n = {entry['n']})",
            "",
            f"Best light-class system so far: {shown(set_name, entry['best_light_class_so_far'])}; "
            f"best system so far: {shown(set_name, entry['best_so_far'])} "
            f"(by FS@{budget}, Phase 02).",
        ]
        lines += [f"Note: {note}" for note in entry["notes"]]
        lines += [
            "",
            "| System | FS@1,024 | FS@2,048 | FS@4,096 | FS@2 | FS@5 | FS@20 | Share@5 "
            "| R@2 | R@5 | R@10 | R@20 | R@100 | nDCG@10 |",
            "|---|" + "---:|" * 13,
        ]
        for system in systems:
            m = entry["systems"][system]["metrics"]
            fs, at_k, r = m["full_support_at_budget"], m["full_support_at_k"], m["recall_at_k"]
            recalls = " | ".join(f"{r[k]:.4f}" for k in ("2", "5", "10", "20", "100"))
            lines.append(
                f"| {shown(set_name, system)} | {fs['1024']} | {fs['2048']} | {fs['4096']} "
                f"| {at_k['2']} | {at_k['5']} | {at_k['20']} | {m['gold_share_at_5']:.4f} "
                f"| {recalls} | {m['ndcg_at_10']:.4f} |"
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
                f"| {test['system']} | {shown(set_name, test['against'])} | {test['wins']} "
                f"| {test['losses']} | {test['ties']} | {test['p']:.4g} | {test['state']} |"
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
        first = entry["systems"][CANDIDATES[0]]["cost"]
        lines += ["", f"Totals per hardware, {first['totals_label']}:", ""]
        for system in CANDIDATES:
            cost = entry["systems"][system]["cost"]
            for name in sorted(cost["totals_per_hardware"]):
                total = cost["totals_per_hardware"][name]
                lines.append(
                    f"- {system}, {name}: {total['seconds']:.6g} s, {total['usd']:.6f} USD."
                )
            usd = cost["usd_total"]
            lines.append(
                f"- {system}, USD in total: offline {usd['offline']:.6f} USD, online "
                f"{usd['online_per_question']:.6f} USD per question ({cost['usd_total_label']})."
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
        for system in systems:
            if system in CANDIDATES:
                continue
            cost = entry["systems"][system]["cost"]
            text = cost.get("label", "")
            if "online_seconds_per_question" in cost:
                text = (
                    f"offline {cost['offline_seconds']:.0f} s, {cost['offline_usd']:.3f} USD; "
                    f"online {cost['online_seconds_per_question']:.4f} s/q, "
                    f"{cost['online_usd_per_question']:.6f} USD/q on "
                    f"{cost.get('hardware', 'hardware not recorded')} ({cost['label']})"
                )
            lines.append(f"- {shown(set_name, system)}: {text}.")
            for note in cost.get("handover", []):
                lines.append(f"  - {note['figure']} ({note['label']}; {note['source']}).")
        diag = entry["exploratory_gold_diagnostics"]
        lines += [
            "",
            f"Diversity pass: {entry['boilerplate_units']} boilerplate units in the corpus; "
            f"demotions over {entry['demotions_per_rule_scope']}, all questions: "
            + ", ".join(f"{rule} {entry['demotions_per_rule'][rule]}" for rule in RULES)
            + "; of these, inside RRF3's top 100: "
            + ", ".join(
                f"{rule} {entry['demotions_per_rule_in_rrf3_top_100'][rule]}" for rule in RULES
            )
            + " (measured).",
            "",
            "Gold-informed diagnostics (exploratory, decide nothing): questions where a gold unit "
            f"demoted by the rule was in RRF3's FS@{budget} context but not in F3's, and in "
            "RRF3's top 100 but not in F3's:",
            "",
            f"| Rule | Lost from the FS@{budget} context | Lost from the top 100 |",
            "|---|---:|---:|",
        ]
        for rule in RULES:
            found = diag["questions_with_a_gold_unit_demoted"][rule]
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
    provenance = git_provenance()
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
        **provenance,
        "phase02_results_sha256": hashlib.sha256(PHASE02_RESULTS.read_bytes()).hexdigest(),
        "outcome": outcome(tables),
        "not_run": not_run,
        "sets": [t for t in tables.values() if t is not None],
    }
    write_pair(table, markdown, RESULTS_JSON, RESULTS_MD)
    return table


Page = Callable[[Mapping[str, Any]], str]


def write_pair(table: Mapping[str, Any], page_of: Page, json_path: Path, md_path: Path) -> None:
    """Write `results.json` and its page, after checking the page regenerates byte-equal from
    the JSON as written (Phase 03 C4, Phase 04 C7), so a failed check writes nothing."""
    body = json.dumps(table, indent=2, sort_keys=True)
    page = page_of(table)
    if page_of(json.loads(body)) != page:
        raise ArtifactError(f"{md_path.name} does not regenerate from {json_path.name}")
    write_bytes(json_path, body.encode("utf-8"))
    write_bytes(md_path, page.encode("utf-8"))


def regenerate(
    page_of: Page | None = None, json_path: Path | None = None, md_path: Path | None = None
) -> str:
    """Render the page from results.json as written; equal to the file on disk (Phase 03's
    pair unless another is named)."""
    json_path, md_path = json_path or RESULTS_JSON, md_path or RESULTS_MD
    page = (page_of or markdown)(json.loads(json_path.read_text("utf-8")))
    if page.encode("utf-8") != md_path.read_bytes():
        raise ArtifactError(f"{md_path.name} differs from the page rendered from results.json")
    return hashlib.sha256(page.encode("utf-8")).hexdigest()
