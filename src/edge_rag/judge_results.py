"""Phase 04 results table (spec C7, C8; plan D7): `j-rrf4` against G-R, `j-rrf3` and the bars.

Reuses `fusion_results` for the states, the advance rule, the verdict, the inherited cost
labels and the byte-equal write of `results.json` and `results.md`. Every figure is read from
run files, each ranking checked against the sha256 its manifest or Phase 02 recorded.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from edge_rag import components, config, fuse, judge, metrics, pool, scoring
from edge_rag import fusion_results as fr
from edge_rag.artifacts import ArtifactError, Checks, OldData
from edge_rag.pod.common import git_provenance

RESULTS_JSON = config.PHASE04_DIR / "results.json"
RESULTS_MD = config.REPO_ROOT / "docs" / "plans" / "fase-04-judge" / "results.md"
CANDIDATE, CONTROL = "j-rrf4", "j-rrf3"
IN_BOUND = ("g-r", "j-p10b")  # rerank-class systems that judge 100 units (spec)
UNIT_BOUND, GPU_BOUND, LAPTOP_BOUND = 100, 1.0, 2.0
COMPARISONS = (
    (CANDIDATE, "g-r"),
    (CANDIDATE, CONTROL),
    (CANDIDATE, "best_in_bound"),
    (CANDIDATE, "j-union"),
    (CANDIDATE, "best_so_far"),
    (CANDIDATE, "f3"),
    (CONTROL, "g-r"),
    (CONTROL, "best_so_far"),
)
STATE_NAMES = tuple(f"{s} vs {a.replace('_', ' ')}" for s, a in COMPARISONS)
JUDGED = {judged: pooled for pooled, judged in judge.JUDGED.items()}
POD = "seconds measured (Phase 04 pod manifest); USD derived (time x rate)"


def outcome(tables: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """States per set and comparison, the verdict, `j-rrf3`'s context verdict, the entrant."""
    states: dict[str, dict[str, str]] = {}
    for name, (system, against) in zip(STATE_NAMES, COMPARISONS, strict=True):
        states[name] = {}
        for set_name, table in tables.items():
            test = None
            if table is not None:
                target = table.get(against, against)
                test = next(
                    t
                    for t in table[f"paired_full_support_at_{config.BUDGET}"]
                    if t["system"] == system and t["against"] == target
                )
            states[name][set_name] = fr.state(test)

    def of(i: int) -> list[str]:
        return list(states[STATE_NAMES[i]].values())

    found = fr.verdict(of(0), of(1), of(2), CONTROL, "in-bound rerank-class")
    return {
        "states": states,
        "verdict": found,
        "j-rrf3_context_verdict": "advances" if fr.advances(of(6)) else "does not advance",
        "exam_entrant": CANDIDATE if found == "entrant" else "none from this phase",
        "label": "written by code under the frozen selection rule",
    }


def _row(name: str, kind: str, hardware: str, seconds: Any, usd: Any, label: str) -> dict:
    return {"component": name, "kind": kind, "hardware": hardware, "seconds": seconds,
            "usd": usd, "label": label}  # fmt: skip


def system_cost(
    system: str, set_name: str, comp: Mapping, fused: Mapping, pooled: Mapping,
    g_l: Mapping, gpu: str, pod: Mapping | None, units: int = 0,
) -> dict[str, Any]:  # fmt: skip
    """Each component on its own hardware, per-hardware totals and the rerank-class check;
    `units` is the judged units per question, read from the judged rankings."""
    laptop, measured = comp["hardware"], fr.LAPTOP_USD
    hop = system in ("rrf4", CANDIDATE)
    inherited = "; ".join(
        f"{n['figure']} ({n['label']}; {n['source']})" for n in fr.handover_cost(set_name, "p10-a")
    )
    rows = [
        _row("BM25 build", "offline", laptop, comp["offline_seconds"]["bm25_build"], 0.0, measured),
        _row("Dense corpus embeddings", "offline", "old pod", None, None,
             f"inherited, not measured here: {inherited or 'not recorded'}"),
        _row("G-L encode and index", "offline", gpu, g_l["offline_seconds"], g_l["offline_usd"],
             "seconds measured (Phase 02 G-L manifest); USD derived (time x rate)"),
        _row("BGE-small question encoding", "online", laptop,
             comp["question_encoding"]["seconds_per_question"], 0.0, measured),
        _row("Dense retrieval", "online", laptop,
             comp["retrieval_seconds_per_question"]["dense"], 0.0, measured),
        _row("BM25 retrieval", "online", laptop,
             comp["retrieval_seconds_per_question"]["bm25"], 0.0, measured),
        _row("G-L search", "online", gpu, g_l["online_seconds_per_question"],
             g_l["online_usd_per_question"],
             "seconds measured (Phase 02 G-L manifest); USD derived (time x rate)"),
    ]  # fmt: skip
    if hop:
        gliner = "; ".join(
            f"{n['figure']} ({n['label']}; {n['source']})"
            for n in fr.handover_cost(set_name, "p10-c")
        )
        rows += [
            _row("GLiNER entity index", "offline", "old pod", None, None, f"inherited: {gliner}"),
            _row("Hop entity index and weights load", "offline", laptop,
                 pooled["offline_seconds"]["hop_entity_index_and_weights"], 0.0,
                 "seconds measured (Phase 04 pool manifest); USD 0 by assumption"),
            _row("Entity hop", "online", laptop, pooled["online_seconds_per_question"]["hop"], 0.0,
                 "seconds measured (Phase 04 pool manifest); USD 0 by assumption"),
            _row("RRF4", "online", laptop, pooled["online_seconds_per_question"]["rrf4"], 0.0,
                 "seconds measured (Phase 04 pool manifest); USD 0 by assumption"),
        ]  # fmt: skip
    else:
        rows.append(_row("RRF", "online", laptop, fused["online_seconds_per_question"]["rrf"],
                         0.0, measured))  # fmt: skip
    if system in JUDGED and pod is not None:
        rate = pod["cost_per_hr_usd"]
        if rate is None:
            raise ArtifactError(f"{set_name}: the pod manifest has no cost_per_hr_usd")
        per_100 = pod["timing_sample"]["seconds_per_100_pairs"]
        load = pod["seconds"]["load_model"]
        rows += [
            _row("J-strong model load (setup, per session)", "offline", pod["gpu"], load,
                 load * rate / 3600, POD),
            _row("J-strong over 100 units", "online", pod["gpu"], per_100, per_100 * rate / 3600,
                 "seconds derived (timing sample seconds over its pairs x 100, every pair scored "
                 "fresh, Phase 04 pod manifest; plan D9); USD derived (time x rate)"),
        ]  # fmt: skip
    totals: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["seconds"] is not None:
            entry = totals.setdefault(f"{r['kind']} {r['hardware']}", {"seconds": 0.0, "usd": 0.0})
            entry["seconds"] += r["seconds"]
            entry["usd"] += r["usd"]
    online = {
        hw: sum(r["seconds"] for r in rows if r["kind"] == "online" and r["hardware"] == hw)
        for hw in {r["hardware"] for r in rows if r["kind"] == "online"}
    }
    gpus = {hw: s for hw, s in online.items() if hw != laptop}
    return {
        "components": rows,
        "totals_per_hardware": totals,
        "totals_label": "derived (sums of the components on one hardware; inherited rows "
        "without seconds are not in the totals)",
        "class_check": {
            "judged_units_per_question": units,
            "unit_bound": UNIT_BOUND,
            "laptop_online_seconds_per_question": online.get(laptop, 0.0),
            "laptop_bound": LAPTOP_BOUND,
            "gpu_online_seconds_per_question": gpus,
            "gpu_bound": GPU_BOUND,
            "gpu_bound_stated_for": "one RTX 4090",
            "inside_rerank_class": units <= UNIT_BOUND
            and online.get(laptop, 0.0) <= LAPTOP_BOUND
            and all(s <= GPU_BOUND for s in gpus.values()),
            "label": "derived; GPU seconds are checked per hardware, never summed across",
        },
    }


def set_table(set_name: str, phase02: Mapping[str, Any]) -> dict[str, Any]:
    spec = config.SETS[set_name]
    checks = Checks()
    old = OldData(spec.directory, spec.file_sha256, checks)
    questions = scoring.questions_of(old, checks, spec)
    counts = scoring.token_counts(old, checks, spec)
    p02, p03 = config.RANKINGS_DIR / set_name, config.PHASE03_RANKINGS_DIR / set_name
    p04 = config.PHASE04_RANKINGS_DIR / set_name
    comp = json.loads((p03 / components.MANIFEST).read_text("utf-8"))
    fused = json.loads((p03 / fuse.MANIFEST).read_text("utf-8"))
    pooled = json.loads((p04 / pool.MANIFEST).read_text("utf-8"))
    judged = json.loads((p04 / judge.JUDGE_MANIFEST).read_text("utf-8"))
    pod_path = config.PHASE04_SCORES_DIR / f"{set_name}.manifest.json"
    pod = json.loads(pod_path.read_text("utf-8"))
    old_rows = phase02["systems"]
    best_in = max(IN_BOUND, key=lambda s: fr.fs(old_rows[s]))
    best = max(old_rows, key=lambda s: fr.fs(old_rows[s]))
    sources = {
        CANDIDATE: (p04, judged["outputs_sha256"]),
        CONTROL: (p04, judged["outputs_sha256"]),
        "rrf4": (p04, pooled["outputs_sha256"]),
        "f3": (p03, fused["outputs_sha256"]),
    }
    rows: dict[str, Any] = {}
    records: dict[str, list[int]] = {}
    ranked: dict[str, dict[str, list[str]]] = {}
    for system in (CANDIDATE, CONTROL, "rrf4", "g-r", best_in, "j-union", best, "f3"):
        if system in rows:
            continue
        name = f"{system}.jsonl.gz"
        if system in sources:
            ranked[system] = judge.pinned_rankings(
                sources[system][0] / name, sources[system][1][name]
            )
        else:
            ranked[system] = judge.pinned_rankings(p02 / name, old_rows[system]["rankings_sha256"])
        summary, records[system] = scoring.score(questions, ranked[system], counts)
        if system in old_rows and summary != old_rows[system]["metrics"]:
            raise ArtifactError(f"{set_name} {system}: rescored metrics differ from Phase 02")
        ceiling = sum(
            metrics.full_support_at_k(ranked[system][q.qid], q.gold_unit_ids, config.DEPTH)
            for q in questions
        )
        rows[system] = {
            "metrics": summary,
            "metrics_label": "measured",
            "pool_ceiling_fs_at_100_units": ceiling,
        }
    gpu = json.loads((p02 / "g-l.manifest.json").read_text("utf-8"))["gpu"]
    for system in (CANDIDATE, CONTROL, "rrf4"):
        units = max((len(r) for r in ranked[system].values()), default=0) if system in JUDGED else 0
        rows[system]["cost"] = system_cost(
            system, set_name, comp, fused, pooled, old_rows["g-l"]["cost"], gpu, pod, units
        )
    for system in rows:
        if system in old_rows:
            rows[system]["cost"] = {**old_rows[system]["cost"]}
            if fr.handover_cost(set_name, system):
                rows[system]["cost"]["handover"] = fr.handover_cost(set_name, system)
    if "f3" in rows and "cost" not in rows["f3"]:
        rows["f3"]["cost"] = {"label": "Phase 03 results.md (light class, laptop and G-L)"}
    # Exploratory: questions whose gold enters RRF4's top 100 only through the hop.
    lists = [
        judge.pinned_rankings(p02 / "p10-a.jsonl.gz", pooled["inputs_sha256"]["p10-a.jsonl.gz"]),
        judge.pinned_rankings(p03 / "bm25.jsonl.gz", pooled["inputs_sha256"]["bm25.jsonl.gz"]),
        judge.pinned_rankings(p02 / "g-l.jsonl.gz", pooled["inputs_sha256"]["g-l.jsonl.gz"]),
    ]
    hop = judge.pinned_rankings(p04 / "hop.jsonl.gz", pooled["outputs_sha256"]["hop.jsonl.gz"])
    through_hop = sum(
        any(
            pool.hop_only([g], [lst[q.qid][: config.DEPTH] for lst in lists], hop[q.qid])
            for g in q.gold_unit_ids
            if g in ranked["rrf4"][q.qid][: config.DEPTH]
        )
        for q in questions
    )
    paired = []
    for system, against in COMPARISONS:
        target = {"best_in_bound": best_in, "best_so_far": best}.get(against, against)
        test = metrics.paired(records[target], records[system])
        paired.append({"system": system, "against": target, **test, "state": fr.state(test),
                       "label": "measured"})  # fmt: skip
    return {
        "set": set_name,
        "n": len(questions),
        "best_in_bound": best_in,
        "best_so_far": best,
        "systems": rows,
        f"paired_full_support_at_{config.BUDGET}": paired,
        "rrf4_units_only_in_hop": pooled["rrf4_units_only_in_hop"],
        "questions_with_gold_only_through_hop": {"count": through_hop, "label": "exploratory"},
        "judge_counts": judged["counts"],
        "pod_check": pod["check"],
        "notes": list(fr.SET_NOTES.get(set_name, [])),
        "manifests_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (p04 / pool.MANIFEST, p04 / judge.JUDGE_MANIFEST, pod_path)
        },
        "digests_checked": checks.records,
    }


def _systems(entry: Mapping[str, Any]) -> list[str]:
    order = (CANDIDATE, CONTROL, "rrf4", "g-r", entry["best_in_bound"], "j-union",
             entry["best_so_far"], "f3")  # fmt: skip
    return list(dict.fromkeys(order))


def markdown(table: Mapping[str, Any]) -> str:
    budget = str(config.BUDGET)
    out = table["outcome"]
    set_names = sorted(out["states"][STATE_NAMES[0]])
    lines = [
        "# Phase 04 - Results",
        "",
        "Generated by `uv run edge-rag judge-results` from `data/phase04/results.json`; "
        "do not edit by hand.",
        "Metrics, pool ceilings and paired tests are measured; states, verdicts and the class "
        "check are written by code; per-hardware totals are derived.",
        "",
        "## Outcome",
        "",
        f"- j-rrf4 verdict: **{out['verdict']}**.",
        f"- Exam entrant: {out['exam_entrant']}.",
        f"- j-rrf3 against G-R under the same advance rule (context): "
        f"{out['j-rrf3_context_verdict']}.",
        "",
        "| Comparison | " + " | ".join(set_names) + " |",
        "|---|" + "---|" * len(set_names),
    ]
    for name in STATE_NAMES:
        lines.append(f"| {name} | " + " | ".join(out["states"][name][s] for s in set_names) + " |")
    lines.append("")
    for set_name in sorted(table["not_run"]):
        lines.append(f"- {set_name}: {fr.NOT_RUN} ({table['not_run'][set_name]}).")
    for entry in sorted(table["sets"], key=lambda e: str(e["set"])):
        lines += [
            "",
            f"## {entry['set']} (n = {entry['n']})",
            "",
            f"Best in-bound rerank-class system so far: {entry['best_in_bound']}; best system so "
            f"far: {entry['best_so_far']} (by FS@{budget}, Phase 02).",
        ]
        lines += [f"Note: {note}" for note in entry["notes"]]
        lines += [
            "",
            "| System | FS@1,024 | FS@2,048 | FS@4,096 | FS@2 | FS@5 | FS@20 | FS@100 (pool "
            "ceiling) | Share@5 | R@2 | R@5 | R@10 | R@20 | R@100 | nDCG@10 |",
            "|---|" + "---:|" * 14,
        ]
        for system in _systems(entry):
            row = entry["systems"][system]
            m = row["metrics"]
            fs, at_k, r = m["full_support_at_budget"], m["full_support_at_k"], m["recall_at_k"]
            recalls = " | ".join(f"{r[k]:.4f}" for k in ("2", "5", "10", "20", "100"))
            lines.append(
                f"| {system} | {fs['1024']} | {fs['2048']} | {fs['4096']} | {at_k['2']} "
                f"| {at_k['5']} | {at_k['20']} | {row['pool_ceiling_fs_at_100_units']} "
                f"| {m['gold_share_at_5']:.4f} | {recalls} | {m['ndcg_at_10']:.4f} |"
            )
        lines += [
            "",
            f"Paired exact McNemar on FS@{budget}, alpha 0.05 per comparison, no correction:",
            "",
            "| System | Against | Wins | Losses | Ties | Exact p | State |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for t in entry[f"paired_full_support_at_{budget}"]:
            lines.append(
                f"| {t['system']} | {t['against']} | {t['wins']} | {t['losses']} | {t['ties']} "
                f"| {t['p']:.4g} | {t['state']} |"
            )
        lines += ["", "Cost per component (seconds per question online, per corpus offline):", ""]
        lines += ["| System | Component | Kind | Hardware | Seconds | USD | Label |"]
        lines += ["|---|---|---|---|---:|---:|---|"]
        for system in (CANDIDATE, CONTROL, "rrf4"):
            for c in entry["systems"][system]["cost"]["components"]:
                lines.append(
                    f"| {system} | {c['component']} | {c['kind']} | {c['hardware']} "
                    f"| {fr.num(c['seconds'], '.6g')} | {fr.num(c['usd'], '.6f')} "
                    f"| {c['label']} |"
                )
        lines.append("")
        for system in (CANDIDATE, CONTROL, "rrf4"):
            cost = entry["systems"][system]["cost"]
            for name in sorted(cost["totals_per_hardware"]):
                total = cost["totals_per_hardware"][name]
                lines.append(
                    f"- {system}, {name}: {total['seconds']:.6g} s, {total['usd']:.6f} USD "
                    "(derived)."
                )
            check = cost["class_check"]
            gpus = ", ".join(
                f"{hw} {s:.4g} s/q"
                for hw, s in sorted(check["gpu_online_seconds_per_question"].items())
            )
            where = "inside" if check["inside_rerank_class"] else "outside"
            lines.append(
                f"- {system} class check: {check['judged_units_per_question']} judged units "
                f"(bound {check['unit_bound']}); laptop online "
                f"{check['laptop_online_seconds_per_question']:.4g} s/q (bound "
                f"{check['laptop_bound']}); GPU online {gpus} (bound {check['gpu_bound']}, stated "
                f"for {check['gpu_bound_stated_for']}): {where} the rerank class on this set "
                "(derived)."
            )
            for hw in sorted(
                h for h in check["gpu_online_seconds_per_question"] if "4090" not in h
            ):
                timed = ", ".join(
                    c["component"]
                    for c in cost["components"]
                    if c["kind"] == "online" and c["hardware"] == hw
                )
                lines.append(
                    f"  - {system}: {timed} time was measured on {hw}, not the RTX 4090 the "
                    "GPU bound names, so the margin may be overstated."
                )
        lines += ["", "Cost of the reference rows (inherited labels):", ""]
        for system in _systems(entry):
            if system in (CANDIDATE, CONTROL, "rrf4"):
                continue
            cost = entry["systems"][system]["cost"]
            text = cost.get("label", "")
            if "online_seconds_per_question" in cost:
                text = (
                    f"offline {cost['offline_seconds']:.0f} s, {cost['offline_usd']:.3f} USD; "
                    f"online {cost['online_seconds_per_question']:.4f} s/q, "
                    f"{cost['online_usd_per_question']:.6f} USD/q ({cost['label']})"
                )
            lines.append(f"- {system}: {text}.")
            for note in cost.get("handover", []):
                lines.append(f"  - {note['figure']} ({note['label']}; {note['source']}).")
        only = entry["rrf4_units_only_in_hop"]
        jc = entry["judge_counts"]
        lines += [
            "",
            f"RRF4 top-100 units contributed only by the hop: {only['total']} "
            f"({only['per_question']} per question, measured).",
            "Questions whose gold enters RRF4's top 100 only through the hop: "
            f"{entry['questions_with_gold_only_through_hop']['count']} (exploratory).",
            f"Judge: {jc['pairs_with_one_score']} pairs with one score, {jc['from_cache']} from "
            f"the old cache, {jc['from_pod']} from the pod; pod C4 check: "
            f"{entry['pod_check']['pairs_compared']} pairs, max abs diff "
            f"{entry['pod_check']['max_abs_diff']:.3g} (measured).",
        ]
    return "\n".join(lines) + "\n"


def run(set_names: Sequence[str]) -> dict[str, Any]:
    provenance = git_provenance()
    phase02 = {e["set"]: e for e in json.loads(fr.PHASE02_RESULTS.read_text("utf-8"))["sets"]}
    tables: dict[str, dict[str, Any] | None] = {}
    not_run: dict[str, str] = {}
    for set_name in set_names:
        if not (config.PHASE04_RANKINGS_DIR / set_name / judge.JUDGE_MANIFEST).exists():
            tables[set_name], not_run[set_name] = None, "no Phase 04 judge manifest"
            continue
        tables[set_name] = set_table(set_name, phase02[set_name])
    table = {
        **provenance,
        "phase02_results_sha256": hashlib.sha256(fr.PHASE02_RESULTS.read_bytes()).hexdigest(),
        "outcome": outcome(tables),
        "not_run": not_run,
        "sets": [t for t in tables.values() if t is not None],
    }
    fr.write_pair(table, markdown, RESULTS_JSON, RESULTS_MD)
    return table


def regenerate() -> str:
    return fr.regenerate(markdown, RESULTS_JSON, RESULTS_MD)
