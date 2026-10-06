"""Results tables of Phases 03, 04, 05 and 06 (Phase 05 spec C3-C6; plan D3, D4): one module.

The shared code (states, the advance rule, the verdict, inherited cost labels, the byte-equal
write of `results.json` and `results.md`, the run) sits at module level and on `Phase`; each
phase is one small description, a subclass of `Phase` with its output paths, rows, comparisons,
bars, set table and page. Every figure is read from run files, each ranking checked against the
sha256 its manifest or Phase 02 recorded. States, verdicts and class checks are written here by
code under each phase's frozen selection rule.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from edge_rag import components, config, converge, fuse, judge, metrics, pool, scoring
from edge_rag.artifacts import ArtifactError, Checks, OldData, write_bytes
from edge_rag.pod.common import git_provenance
from edge_rag.retrieval.rrf import RULES

PHASE02_RESULTS = config.DATA_DIR / "phase02" / "results.json"
ALPHA = 0.05
NOT_RUN = "not run"
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
    the gate, the control and the best in-class system so far; each phase names its own."""
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


def fs(row: Mapping[str, Any]) -> int:
    return int(row["metrics"]["full_support_at_budget"][str(config.BUDGET)])


def num(value: Any, digits: str) -> str:
    return "-" if value is None else format(value, digits)


def cost_row(name: str, kind: str, hardware: str, seconds: Any, usd: Any, label: str) -> dict:
    return {"component": name, "kind": kind, "hardware": hardware, "seconds": seconds,
            "usd": usd, "label": label}  # fmt: skip


def states_of(
    tables: Mapping[str, Mapping[str, Any] | None], system: str, against: str
) -> dict[str, str]:
    """One comparison's state per set; `against` may name a table key holding the system."""
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


def regenerate(page_of: Page, json_path: Path, md_path: Path) -> str:
    """Render the page from results.json as written; equal to the file on disk."""
    page = page_of(json.loads(json_path.read_text("utf-8")))
    if page.encode("utf-8") != md_path.read_bytes():
        raise ArtifactError(f"{md_path.name} differs from the page rendered from results.json")
    return hashlib.sha256(page.encode("utf-8")).hexdigest()


class Phase:
    """What every phase description shares; a subclass names its paths, set table and page."""

    RESULTS_JSON: ClassVar[Path]
    RESULTS_MD: ClassVar[Path]
    MISSING: ClassVar[str]  # the `not run` reason of a set without the phase's last manifest
    ArtifactError = ArtifactError
    NOT_RUN = NOT_RUN
    LAPTOP_USD = LAPTOP_USD
    state = staticmethod(state)
    advances = staticmethod(advances)
    verdict = staticmethod(verdict)
    handover_cost = staticmethod(handover_cost)
    fs = staticmethod(fs)
    num = staticmethod(num)

    @classmethod
    def ran(cls, set_name: str) -> bool:
        raise NotImplementedError

    @classmethod
    def set_table(cls, set_name: str, phase02: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def outcome(cls, tables: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def markdown(cls, table: Mapping[str, Any]) -> str:
        raise NotImplementedError

    @classmethod
    def run(cls, set_names: Sequence[str], out: Path | None = None) -> dict[str, Any]:
        """Recompute the table from run files; write the stored pair, or `out` (plan D4)."""
        targets = (
            (cls.RESULTS_JSON, cls.RESULTS_MD)
            if out is None
            else (out / "results.json", out / "results.md")
        )
        for target in targets:
            if target.exists():
                raise ArtifactError(
                    f"{target} exists and is written once; recompute with --out to an empty folder"
                )
        provenance = git_provenance()
        phase02 = {e["set"]: e for e in json.loads(PHASE02_RESULTS.read_text("utf-8"))["sets"]}
        tables: dict[str, dict[str, Any] | None] = {}
        not_run: dict[str, str] = {}
        for set_name in set_names:
            if not cls.ran(set_name):
                tables[set_name], not_run[set_name] = None, cls.MISSING
                continue
            tables[set_name] = cls.set_table(set_name, phase02[set_name])
        table = {
            **provenance,
            "phase02_results_sha256": hashlib.sha256(PHASE02_RESULTS.read_bytes()).hexdigest(),
            "outcome": cls.outcome(tables),
            "not_run": not_run,
            "sets": [t for t in tables.values() if t is not None],
        }
        write_pair(table, cls.markdown, *targets)
        return table

    @classmethod
    def regenerate(cls) -> str:
        return regenerate(cls.markdown, cls.RESULTS_JSON, cls.RESULTS_MD)


class Phase03(Phase):
    """Phase 03 (its spec C4-C6): RRF3 and F3 against G-L and the best systems so far."""

    RESULTS_JSON = config.PHASE03_DIR / "results.json"
    RESULTS_MD = config.REPO_ROOT / "docs" / "plans" / "fase-03-fusion" / "results.md"
    MISSING = "no Phase 03 fuse manifest"
    LIGHT = ("p10-a", "p10-b", "p10-c", "p14", "g-l")
    CANDIDATES = ("rrf3", "f3")
    LAPTOP_BOUND, GPU_BOUND = 2.0, 0.1
    STATE_NAMES = (
        "f3 vs g-l",
        "f3 vs rrf3",
        "f3 vs best light-class so far",
        "f3 vs best so far",
        "rrf3 vs g-l",
        "rrf3 vs best light-class so far",
        "rrf3 vs best so far",
    )

    @classmethod
    def ran(cls, set_name: str) -> bool:
        return (config.PHASE03_RANKINGS_DIR / set_name / fuse.MANIFEST).exists()

    @classmethod
    def candidate_cost(
        cls,
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
             "seconds": g_l_cost["offline_seconds"], "usd": g_l_cost["offline_usd"],
             "label": pod},
            {"component": "BGE-small question encoding", "kind": "online", "hardware": laptop,
             "seconds": comp["question_encoding"]["seconds_per_question"], "usd": 0.0,
             "label": f"{LAPTOP_USD}; seconds on the first "
                      f"{comp['question_encoding']['questions']} questions"},
            {"component": "Dense retrieval", "kind": "online", "hardware": laptop,
             "seconds": comp["retrieval_seconds_per_question"]["dense"], "usd": 0.0,
             "label": measured},
            {"component": "BM25 retrieval", "kind": "online", "hardware": laptop,
             "seconds": comp["retrieval_seconds_per_question"]["bm25"], "usd": 0.0,
             "label": measured},
            {"component": "G-L search", "kind": "online", "hardware": gpu,
             "seconds": g_l_cost["online_seconds_per_question"],
             "usd": g_l_cost["online_usd_per_question"], "label": pod},
            {"component": "RRF", "kind": "online", "hardware": laptop,
             "seconds": fused["online_seconds_per_question"]["rrf"], "usd": 0.0,
             "label": measured},
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
            key = f"{row['kind']} {row['hardware']}"
            entry = totals.setdefault(key, {"seconds": 0.0, "usd": 0.0})
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
                "laptop_bound": cls.LAPTOP_BOUND,
                "laptop_inside": laptop_online <= cls.LAPTOP_BOUND,
                "gpu": gpu,
                "gpu_online_seconds_per_question": gpu_online,
                "gpu_bound": cls.GPU_BOUND,
                "gpu_bound_stated_for": "one RTX 4090",
                "gpu_inside": gpu_online <= cls.GPU_BOUND,
                "inside_light_class": laptop_online <= cls.LAPTOP_BOUND
                and gpu_online <= cls.GPU_BOUND,
                "label": "derived",
            },
        }

    @classmethod
    def set_table(cls, set_name: str, phase02: Mapping[str, Any]) -> dict[str, Any]:
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
        best_light = max(cls.LIGHT, key=lambda s: fs(old_rows[s]))
        best = max(old_rows, key=lambda s: fs(old_rows[s]))
        if old_rows["g-l"]["rankings_sha256"] != config.GL_SHA256[set_name]:
            raise ArtifactError(f"{set_name}: Phase 02 results name another G-L ranking file")
        rows: dict[str, Any] = {}
        records: dict[str, list[int]] = {}
        for system in (*cls.CANDIDATES, "g-l", best_light, best):
            if system in rows:
                continue
            path = (here if system in cls.CANDIDATES else there) / f"{system}.jsonl.gz"
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            recorded = (
                fused["outputs_sha256"][path.name]
                if system in cls.CANDIDATES
                else old_rows[system]["rankings_sha256"]
            )
            if sha256 != recorded:
                raise ArtifactError(f"{path}: sha256 {sha256}, recorded {recorded}")
            summary, records[system] = scoring.score(questions, scoring.read_rankings(path), counts)
            if system not in cls.CANDIDATES and summary != old_rows[system]["metrics"]:
                raise ArtifactError(f"{set_name} {system}: rescored metrics differ from Phase 02")
            rows[system] = {
                "rankings_sha256": sha256,
                "metrics": summary,
                "metrics_label": "measured",
            }
        gpu = json.loads((there / "g-l.manifest.json").read_text("utf-8"))["gpu"]
        notes = list(SET_NOTES.get(set_name, []))
        for system in cls.CANDIDATES:
            rows[system]["cost"] = cls.candidate_cost(
                system, comp, fused, old_rows["g-l"]["cost"], gpu, set_name
            )
        for system in rows:
            if system in cls.CANDIDATES:
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

    @classmethod
    def outcome(cls, tables: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
        """States per set for each bar, F3's verdict and RRF3's context verdict."""
        f3_g_l = states_of(tables, "f3", "g-l")
        f3_rrf3 = states_of(tables, "f3", "rrf3")
        f3_light = states_of(tables, "f3", "best_light_class_so_far")
        rrf3_g_l = states_of(tables, "rrf3", "g-l")
        f3_verdict = verdict(list(f3_g_l.values()), list(f3_rrf3.values()), list(f3_light.values()))
        return {
            "states": {
                "f3 vs g-l": f3_g_l,
                "f3 vs rrf3": f3_rrf3,
                "f3 vs best light-class so far": f3_light,
                "f3 vs best so far": states_of(tables, "f3", "best_so_far"),
                "rrf3 vs g-l": rrf3_g_l,
                "rrf3 vs best light-class so far": states_of(
                    tables, "rrf3", "best_light_class_so_far"
                ),
                "rrf3 vs best so far": states_of(tables, "rrf3", "best_so_far"),
            },
            "f3_verdict": f3_verdict,
            "rrf3_context_verdict": "advances" if advances(list(rrf3_g_l.values())) else
            "does not advance",
            "exam_entrant": "F3" if f3_verdict == "entrant" else "none from this phase",
            "label": "written by code under the frozen selection rule",
        }  # fmt: skip

    @classmethod
    def _systems(cls, entry: Mapping[str, Any]) -> list[str]:
        """The table's system order, fixed: candidates, G-L, best light-class, best (no
        repeats)."""
        order = (*cls.CANDIDATES, "g-l", entry["best_light_class_so_far"], entry["best_so_far"])
        return list(dict.fromkeys(order))

    @classmethod
    def markdown(cls, table: Mapping[str, Any]) -> str:
        """Every row in an explicit order, so the page renders the same from the in-memory table
        and from `results.json` as written (sorted keys)."""
        budget = str(config.BUDGET)
        out = table["outcome"]
        set_names = sorted(out["states"][cls.STATE_NAMES[0]])
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
            "- RRF3 against G-L under the same advance rule (context): "
            f"{out['rrf3_context_verdict']}.",
            "",
            "| Comparison | " + " | ".join(set_names) + " |",
            "|---|" + "---|" * len(set_names),
        ]
        for name in cls.STATE_NAMES:
            states = out["states"][name]
            lines.append(f"| {name} | " + " | ".join(states[s] for s in set_names) + " |")
        lines.append("")
        for set_name in sorted(table["not_run"]):
            lines.append(f"- {set_name}: {NOT_RUN} ({table['not_run'][set_name]}).")
        for entry in sorted(table["sets"], key=lambda e: str(e["set"])):
            set_name = entry["set"]
            systems = cls._systems(entry)
            lines += [
                "",
                f"## {set_name} (n = {entry['n']})",
                "",
                "Best light-class system so far: "
                f"{shown(set_name, entry['best_light_class_so_far'])}; "
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
                at_b, at_k, r = (
                    m["full_support_at_budget"],
                    m["full_support_at_k"],
                    m["recall_at_k"],
                )
                recalls = " | ".join(f"{r[k]:.4f}" for k in ("2", "5", "10", "20", "100"))
                lines.append(
                    f"| {shown(set_name, system)} | {at_b['1024']} | {at_b['2048']} "
                    f"| {at_b['4096']} | {at_k['2']} | {at_k['5']} | {at_k['20']} "
                    f"| {m['gold_share_at_5']:.4f} | {recalls} | {m['ndcg_at_10']:.4f} |"
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
            lines += [
                "",
                "Cost per component (seconds per question online, per corpus offline):",
                "",
            ]
            lines += ["| System | Component | Kind | Hardware | Seconds | USD | Label |"]
            lines += ["|---|---|---|---|---:|---:|---|"]
            for system in cls.CANDIDATES:
                for c in entry["systems"][system]["cost"]["components"]:
                    lines.append(
                        f"| {system} | {c['component']} | {c['kind']} | {c['hardware']} "
                        f"| {num(c['seconds'], '.6g')} | {num(c['usd'], '.6f')} | {c['label']} |"
                    )
            first = entry["systems"][cls.CANDIDATES[0]]["cost"]
            lines += ["", f"Totals per hardware, {first['totals_label']}:", ""]
            for system in cls.CANDIDATES:
                cost = entry["systems"][system]["cost"]
                for name in sorted(cost["totals_per_hardware"]):
                    total = cost["totals_per_hardware"][name]
                    lines.append(
                        f"- {system}, {name}: {total['seconds']:.6g} s, {total['usd']:.6f} USD."
                    )
                usd = cost["usd_total"]
                lines.append(
                    f"- {system}, USD in total: offline {usd['offline']:.6f} USD, online "
                    f"{usd['online_per_question']:.6f} USD per question "
                    f"({cost['usd_total_label']})."
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
                    f"{check['gpu_online_seconds_per_question']:.4g} s/q (bound "
                    f"{check['gpu_bound']}, stated for {check['gpu_bound_stated_for']}): "
                    f"{where} (derived)."
                )
                lines.append(
                    f"- {system} cross-hardware online total, context only: "
                    f"{cost['cross_hardware_online_seconds_per_question']:.4g} s/q (derived)."
                )
            lines += ["", "Cost of the reference rows (inherited labels):", ""]
            for system in systems:
                if system in cls.CANDIDATES:
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
                "Gold-informed diagnostics (exploratory, decide nothing): questions where a gold "
                f"unit demoted by the rule was in RRF3's FS@{budget} context but not in F3's, and "
                "in RRF3's top 100 but not in F3's:",
                "",
                f"| Rule | Lost from the FS@{budget} context | Lost from the top 100 |",
                "|---|---:|---:|",
            ]
            for rule in RULES:
                found = diag["questions_with_a_gold_unit_demoted"][rule]
                lines.append(
                    f"| {rule} | {found['in_context_at_budget']} | {found['in_top_100']} |"
                )
            lines += [
                "",
                f"Gold units flagged boilerplate: {diag['gold_units_flagged_boilerplate']} of "
                f"{diag['gold_units']} (exploratory).",
                f"Peak RSS: components {entry['peak_rss_mb']['components']} MB, fuse "
                f"{entry['peak_rss_mb']['fuse']} MB (measured).",
            ]
        return "\n".join(lines) + "\n"


class Phase04(Phase):
    """Phase 04 (its spec C7, C8; plan D7): `j-rrf4` against G-R, `j-rrf3` and the bars."""

    RESULTS_JSON = config.PHASE04_DIR / "results.json"
    RESULTS_MD = config.REPO_ROOT / "docs" / "plans" / "fase-04-judge" / "results.md"
    MISSING = "no Phase 04 judge manifest"
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
    JUDGED: ClassVar[dict[str, str]] = {judged: pooled for pooled, judged in judge.JUDGED.items()}
    POD = "seconds measured (Phase 04 pod manifest); USD derived (time x rate)"

    @classmethod
    def ran(cls, set_name: str) -> bool:
        return (config.PHASE04_RANKINGS_DIR / set_name / judge.JUDGE_MANIFEST).exists()

    @classmethod
    def outcome(cls, tables: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
        """States per set and comparison, the verdict, `j-rrf3`'s context verdict, the entrant."""
        states = {
            name: states_of(tables, system, against)
            for name, (system, against) in zip(cls.STATE_NAMES, cls.COMPARISONS, strict=True)
        }

        def of(i: int) -> list[str]:
            return list(states[cls.STATE_NAMES[i]].values())

        found = verdict(of(0), of(1), of(2), cls.CONTROL, "in-bound rerank-class")
        return {
            "states": states,
            "verdict": found,
            "j-rrf3_context_verdict": "advances" if advances(of(6)) else "does not advance",
            "exam_entrant": cls.CANDIDATE if found == "entrant" else "none from this phase",
            "label": "written by code under the frozen selection rule",
        }

    @staticmethod
    def judged_units(rankings: dict[str, list[str]]) -> int:
        """Units the judge scores per question: the longest judged list."""
        return max((len(r) for r in rankings.values()), default=0)

    @classmethod
    def system_cost(
        cls, system: str, set_name: str, comp: Mapping, fused: Mapping, pooled: Mapping,
        g_l: Mapping, gpu: str, pod: Mapping | None, units: int = 0,
    ) -> dict[str, Any]:  # fmt: skip
        """Each component on its own hardware, per-hardware totals and the rerank-class check;
        `units` is the judged units per question, read from the judged rankings, and sets the
        judge's online row (Phase 05 spec C4)."""
        laptop, measured = comp["hardware"], LAPTOP_USD
        hop = system in ("rrf4", cls.CANDIDATE)
        inherited = "; ".join(
            f"{n['figure']} ({n['label']}; {n['source']})" for n in handover_cost(set_name, "p10-a")
        )
        g_l_label = "seconds measured (Phase 02 G-L manifest); USD derived (time x rate)"
        rows = [
            cost_row("BM25 build", "offline", laptop, comp["offline_seconds"]["bm25_build"], 0.0,
                     measured),
            cost_row("Dense corpus embeddings", "offline", "old pod", None, None,
                     f"inherited, not measured here: {inherited or 'not recorded'}"),
            cost_row("G-L encode and index", "offline", gpu, g_l["offline_seconds"],
                     g_l["offline_usd"], g_l_label),
            cost_row("BGE-small question encoding", "online", laptop,
                     comp["question_encoding"]["seconds_per_question"], 0.0, measured),
            cost_row("Dense retrieval", "online", laptop,
                     comp["retrieval_seconds_per_question"]["dense"], 0.0, measured),
            cost_row("BM25 retrieval", "online", laptop,
                     comp["retrieval_seconds_per_question"]["bm25"], 0.0, measured),
            cost_row("G-L search", "online", gpu, g_l["online_seconds_per_question"],
                     g_l["online_usd_per_question"], g_l_label),
        ]  # fmt: skip
        if hop:
            gliner = "; ".join(
                f"{n['figure']} ({n['label']}; {n['source']})"
                for n in handover_cost(set_name, "p10-c")
            )
            pool_label = "seconds measured (Phase 04 pool manifest); USD 0 by assumption"
            rows += [
                cost_row("GLiNER entity index", "offline", "old pod", None, None,
                         f"inherited: {gliner}"),
                cost_row("Hop entity index and weights load", "offline", laptop,
                         pooled["offline_seconds"]["hop_entity_index_and_weights"], 0.0,
                         pool_label),
                cost_row("Entity hop", "online", laptop,
                         pooled["online_seconds_per_question"]["hop"], 0.0, pool_label),
                cost_row("RRF4", "online", laptop, pooled["online_seconds_per_question"]["rrf4"],
                         0.0, pool_label),
            ]  # fmt: skip
        else:
            rrf = fused["online_seconds_per_question"]["rrf"]
            rows.append(cost_row("RRF", "online", laptop, rrf, 0.0, measured))
        if system in cls.JUDGED and pod is not None:
            rate = pod["cost_per_hr_usd"]
            if rate is None:
                raise ArtifactError(f"{set_name}: the pod manifest has no cost_per_hr_usd")
            # units / 100 first: with 100 units the factor is exactly 1.0, so the seconds are
            # the per-100 figure bit for bit (the stored Phase 04 page, spec C3).
            seconds = pod["timing_sample"]["seconds_per_100_pairs"] * (units / 100)
            load = pod["seconds"]["load_model"]
            rows += [
                cost_row("J-strong model load (setup, per session)", "offline", pod["gpu"], load,
                         load * rate / 3600, cls.POD),
                cost_row(f"J-strong over {units} units", "online", pod["gpu"], seconds,
                         seconds * rate / 3600,
                         f"seconds derived (timing sample seconds over its pairs x {units}, every "
                         "pair scored fresh, Phase 04 pod manifest; plan D9); USD derived (time x "
                         "rate)"),
            ]  # fmt: skip
        totals: dict[str, dict[str, float]] = {}
        for r in rows:
            if r["seconds"] is not None:
                key = f"{r['kind']} {r['hardware']}"
                entry = totals.setdefault(key, {"seconds": 0.0, "usd": 0.0})
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
                "unit_bound": cls.UNIT_BOUND,
                "laptop_online_seconds_per_question": online.get(laptop, 0.0),
                "laptop_bound": cls.LAPTOP_BOUND,
                "gpu_online_seconds_per_question": gpus,
                "gpu_bound": cls.GPU_BOUND,
                "gpu_bound_stated_for": "one RTX 4090",
                "inside_rerank_class": units <= cls.UNIT_BOUND
                and online.get(laptop, 0.0) <= cls.LAPTOP_BOUND
                and all(s <= cls.GPU_BOUND for s in gpus.values()),
                "label": "derived; GPU seconds are checked per hardware, never summed across",
            },
        }

    @classmethod
    def set_table(cls, set_name: str, phase02: Mapping[str, Any]) -> dict[str, Any]:
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
        best_in = max(cls.IN_BOUND, key=lambda s: fs(old_rows[s]))
        best = max(old_rows, key=lambda s: fs(old_rows[s]))
        sources = {
            cls.CANDIDATE: (p04, judged["outputs_sha256"]),
            cls.CONTROL: (p04, judged["outputs_sha256"]),
            "rrf4": (p04, pooled["outputs_sha256"]),
            "f3": (p03, fused["outputs_sha256"]),
        }
        rows: dict[str, Any] = {}
        records: dict[str, list[int]] = {}
        ranked: dict[str, dict[str, list[str]]] = {}
        for system in (cls.CANDIDATE, cls.CONTROL, "rrf4", "g-r", best_in, "j-union", best, "f3"):
            if system in rows:
                continue
            name = f"{system}.jsonl.gz"
            if system in sources:
                ranked[system] = judge.pinned_rankings(
                    sources[system][0] / name, sources[system][1][name]
                )
            else:
                ranked[system] = judge.pinned_rankings(
                    p02 / name, old_rows[system]["rankings_sha256"]
                )
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
        for system in (cls.CANDIDATE, cls.CONTROL, "rrf4"):
            units = cls.judged_units(ranked[system]) if system in cls.JUDGED else 0
            rows[system]["cost"] = cls.system_cost(
                system, set_name, comp, fused, pooled, old_rows["g-l"]["cost"], gpu, pod, units
            )
        for system in rows:
            if system in old_rows:
                rows[system]["cost"] = {**old_rows[system]["cost"]}
                if handover_cost(set_name, system):
                    rows[system]["cost"]["handover"] = handover_cost(set_name, system)
        if "f3" in rows and "cost" not in rows["f3"]:
            rows["f3"]["cost"] = {"label": "Phase 03 results.md (light class, laptop and G-L)"}
        # Exploratory: questions whose gold enters RRF4's top 100 only through the hop.
        lists = [
            judge.pinned_rankings(
                p02 / "p10-a.jsonl.gz", pooled["inputs_sha256"]["p10-a.jsonl.gz"]
            ),
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
        for system, against in cls.COMPARISONS:
            target = {"best_in_bound": best_in, "best_so_far": best}.get(against, against)
            test = metrics.paired(records[target], records[system])
            paired.append({"system": system, "against": target, **test, "state": state(test),
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
            "notes": list(SET_NOTES.get(set_name, [])),
            "manifests_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (p04 / pool.MANIFEST, p04 / judge.JUDGE_MANIFEST, pod_path)
            },
            "digests_checked": checks.records,
        }

    @classmethod
    def _systems(cls, entry: Mapping[str, Any]) -> list[str]:
        order = (cls.CANDIDATE, cls.CONTROL, "rrf4", "g-r", entry["best_in_bound"], "j-union",
                 entry["best_so_far"], "f3")  # fmt: skip
        return list(dict.fromkeys(order))

    @classmethod
    def markdown(cls, table: Mapping[str, Any]) -> str:
        budget = str(config.BUDGET)
        out = table["outcome"]
        set_names = sorted(out["states"][cls.STATE_NAMES[0]])
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
        for name in cls.STATE_NAMES:
            lines.append(
                f"| {name} | " + " | ".join(out["states"][name][s] for s in set_names) + " |"
            )
        lines.append("")
        for set_name in sorted(table["not_run"]):
            lines.append(f"- {set_name}: {NOT_RUN} ({table['not_run'][set_name]}).")
        own = (cls.CANDIDATE, cls.CONTROL, "rrf4")
        for entry in sorted(table["sets"], key=lambda e: str(e["set"])):
            lines += [
                "",
                f"## {entry['set']} (n = {entry['n']})",
                "",
                f"Best in-bound rerank-class system so far: {entry['best_in_bound']}; best system "
                f"so far: {entry['best_so_far']} (by FS@{budget}, Phase 02).",
            ]
            lines += [f"Note: {note}" for note in entry["notes"]]
            lines += [
                "",
                "| System | FS@1,024 | FS@2,048 | FS@4,096 | FS@2 | FS@5 | FS@20 | FS@100 (pool "
                "ceiling) | Share@5 | R@2 | R@5 | R@10 | R@20 | R@100 | nDCG@10 |",
                "|---|" + "---:|" * 14,
            ]
            for system in cls._systems(entry):
                row = entry["systems"][system]
                m = row["metrics"]
                at_b, at_k, r = (
                    m["full_support_at_budget"],
                    m["full_support_at_k"],
                    m["recall_at_k"],
                )
                recalls = " | ".join(f"{r[k]:.4f}" for k in ("2", "5", "10", "20", "100"))
                lines.append(
                    f"| {system} | {at_b['1024']} | {at_b['2048']} | {at_b['4096']} | {at_k['2']} "
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
                    f"| {t['system']} | {t['against']} | {t['wins']} | {t['losses']} "
                    f"| {t['ties']} | {t['p']:.4g} | {t['state']} |"
                )
            lines += [
                "",
                "Cost per component (seconds per question online, per corpus offline):",
                "",
            ]
            lines += ["| System | Component | Kind | Hardware | Seconds | USD | Label |"]
            lines += ["|---|---|---|---|---:|---:|---|"]
            for system in own:
                for c in entry["systems"][system]["cost"]["components"]:
                    lines.append(
                        f"| {system} | {c['component']} | {c['kind']} | {c['hardware']} "
                        f"| {num(c['seconds'], '.6g')} | {num(c['usd'], '.6f')} "
                        f"| {c['label']} |"
                    )
            lines.append("")
            for system in own:
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
                    f"{check['laptop_bound']}); GPU online {gpus} (bound {check['gpu_bound']}, "
                    f"stated for {check['gpu_bound_stated_for']}): {where} the rerank class on "
                    "this set (derived)."
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
            for system in cls._systems(entry):
                if system in own:
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
                f"Judge: {jc['pairs_with_one_score']} pairs with one score, {jc['from_cache']} "
                f"from the old cache, {jc['from_pod']} from the pod; pod C4 check: "
                f"{entry['pod_check']['pairs_compared']} pairs, max abs diff "
                f"{entry['pod_check']['max_abs_diff']:.3g} (measured).",
            ]
        return "\n".join(lines) + "\n"


EQUALITY_CHECKS = ("bm25_equals_bm25", "dense_equals_p10-a", "seed_hop_equals_phase04_hop")


def converge_complete(manifest: Mapping[str, Any]) -> bool:
    """Spec C1: each of the three equality checks covers every question of the set."""
    checks = manifest.get("equality", {})
    return set(checks) == set(EQUALITY_CHECKS) and all(
        c["equal"] == c["of"] == manifest["questions"] for c in checks.values()
    )


class Phase05(Phase):
    """Phase 05 (its spec C5, C6; plan D11): `mch` against G-L, `rrf-prf` and the bars.

    The best light-class system so far and the best system so far are read from the stored
    Phase 02-04 results only, never from a Phase 05 figure; a tie on FS@2,048 keeps the earlier
    phase's system, and inside one phase the system first in name order.
    """

    RESULTS_JSON = config.PHASE05_DIR / "results.json"
    RESULTS_MD = config.REPO_ROOT / "docs" / "plans" / "fase-05-hop" / "results.md"
    MISSING = "no Phase 05 converge manifest"
    CANDIDATE, CONTROL, SINGLE = "mch", "rrf-prf", "rrf-1s"
    FUSED = (CANDIDATE, CONTROL, SINGLE)
    OWN = (*FUSED, "hop-ms", "dense-prf")
    LIGHT = ("p10-a", "p10-b", "p10-c", "p14", "g-l", "rrf3", "f3", "rrf4")
    LAPTOP_BOUND, GPU_BOUND, GLINER_BOUND = 2.0, 0.1, 2.0
    COMPARISONS: ClassVar[tuple[tuple[str, str], ...]] = (
        (CANDIDATE, "g-l"),
        (CANDIDATE, CONTROL),
        (CANDIDATE, SINGLE),
        (CANDIDATE, "best_light_class_so_far"),
        (CANDIDATE, "best_so_far"),
        (CONTROL, "g-l"),
        (CONTROL, "best_so_far"),
    )
    STATE_NAMES = tuple(
        f"{s} vs " + a.replace("_", " ").replace("light class", "light-class")
        for s, a in COMPARISONS
    )
    EARLIER: ClassVar[tuple[tuple[str, Path, Path], ...]] = (
        ("Phase 02", PHASE02_RESULTS, config.RANKINGS_DIR),
        ("Phase 03", Phase03.RESULTS_JSON, config.PHASE03_RANKINGS_DIR),
        ("Phase 04", Phase04.RESULTS_JSON, config.PHASE04_RANKINGS_DIR),
    )
    # Phase 03 and 04 digests from the spec; Phase 02's measured at the Phase 05 close.
    EARLIER_SHA256: ClassVar[dict[str, str]] = {
        "Phase 02": "287718c3a4c1f0fc429b1ee63cb04a7b5ba56e4b429cb51a1f9a8a91a0e1a483",
        "Phase 03": "ccf4dfc98ebc893d81220dbbc3db3c324a3c8be1a2a471afb5f906e4a4b31efd",
        "Phase 04": "93fa91404e2ec9a180225ce6f22740adbee45b69ba7163c6e5ea93d260ae4dc9",
    }
    PAGES: ClassVar[dict[str, str]] = {
        "Phase 03": "docs/plans/fase-03-fusion/results.md",
        "Phase 04": "docs/plans/fase-04-judge/results.md",
    }
    # GLiNER extraction, inherited: (seconds, USD attributable or None when shared), from the
    # handover's cost table (spec, class check and cost accounting).
    GLINER_OFFLINE: ClassVar[dict[str, tuple[float, float | None]]] = {
        "hotpotqa-dev": (22248.0, 4.57),  # 6.18 h
        "musique": (618.0, 0.13),  # 10.3 min
        "multihop-rag": (154.77, None),  # inside 0.0404 USD with the BGE encoding
    }

    @classmethod
    def ran(cls, set_name: str) -> bool:
        return (config.PHASE05_RANKINGS_DIR / set_name / converge.MANIFEST).exists()

    @staticmethod
    def best_of(earlier: Mapping[str, Mapping[str, Any]], light: Sequence[str]) -> tuple[str, str]:
        """The best light-class system and the best system so far by FS@2,048, from the earlier
        phases' rows in their order (the first of equals is kept)."""
        best_light = max((s for s in earlier if s in light), key=lambda s: fs(earlier[s]))
        best = max(earlier, key=lambda s: fs(earlier[s]))
        return best_light, best

    @classmethod
    def earlier(cls, set_name: str) -> tuple[dict[str, tuple[str, Any]], dict[str, str]]:
        """Every system of the stored Phase 02-04 results on one set, first phase first."""
        found: dict[str, tuple[str, Any]] = {}
        sha256: dict[str, str] = {}
        for phase, path, _ in cls.EARLIER:
            body = path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            if digest != cls.EARLIER_SHA256[phase]:
                raise ArtifactError(f"{path}: sha256 {digest} is not the pinned {phase} results")
            sha256[str(path.relative_to(config.REPO_ROOT).as_posix())] = digest
            for entry in json.loads(body)["sets"]:
                if entry["set"] == set_name:
                    for system in sorted(entry["systems"]):
                        found.setdefault(system, (phase, entry["systems"][system]))
        return found, sha256

    @classmethod
    def earlier_rankings(cls, set_name: str, phase: str, system: str, row: Mapping) -> Any:
        """An earlier system's ranking file, read only after its recorded sha256 matches."""
        directory = {p: d for p, _, d in cls.EARLIER}[phase] / set_name
        name = f"{system}.jsonl.gz"
        if phase == "Phase 04":
            recorded: dict[str, str] = {}
            for manifest in (judge.JUDGE_MANIFEST, pool.MANIFEST):
                recorded.update(
                    json.loads((directory / manifest).read_text("utf-8"))["outputs_sha256"]
                )
            sha = recorded[name]
        else:
            sha = row["rankings_sha256"]
        return judge.pinned_rankings(directory / name, sha)

    @classmethod
    def system_cost(
        cls, system: str, set_name: str, comp: Mapping, conv: Mapping, pooled: Mapping,
        units: int,
    ) -> dict[str, Any]:  # fmt: skip
        """Each component on its own hardware, per-hardware totals and the light-class check."""
        laptop = conv["hardware"]
        if comp["hardware"] != laptop:
            raise ArtifactError(f"{set_name}: Phase 03 and Phase 05 name another laptop")
        on = conv["online_seconds_per_question"]
        measured = (
            "seconds measured (Phase 05 converge manifest); USD 0 by assumption (owned laptop, "
            "energy not counted)"
        )
        dense = "; ".join(
            f"{n['figure']} ({n['label']}; {n['source']})" for n in handover_cost(set_name, "p10-a")
        )
        hop = system in (cls.CANDIDATE, cls.SINGLE)
        rows = [
            cost_row("BM25 build", "offline", laptop, comp["offline_seconds"]["bm25_build"], 0.0,
                     LAPTOP_USD),
            cost_row("Dense corpus embeddings", "offline", "old pod",
                     41.20 if set_name == "multihop-rag" else None, None,
                     f"inherited, not measured here: {dense or 'not recorded'}"),
        ]  # fmt: skip
        if hop:
            seconds, gliner_usd = cls.GLINER_OFFLINE[set_name]
            gliner = "; ".join(
                f"{n['figure']} ({n['label']}; {n['source']})"
                for n in handover_cost(set_name, "p10-c")
            )
            rows += [
                cost_row("GLiNER entity extraction", "offline", "old pod", seconds, gliner_usd,
                         f"inherited: {gliner}"),
                cost_row("Entity index and weights load", "offline", laptop,
                         conv["offline_seconds"]["hop_entity_index_and_weights"], 0.0, measured),
            ]  # fmt: skip
        rows += [
            cost_row("BGE-small question encoding", "online", laptop,
                     comp["question_encoding"]["seconds_per_question"], 0.0,
                     f"{LAPTOP_USD}; seconds on the first "
                     f"{comp['question_encoding']['questions']} questions, same model and laptop"),
            cost_row("Dense retrieval", "online", laptop, on["dense"], 0.0, measured),
            cost_row("BM25 retrieval", "online", laptop, on["bm25"], 0.0, measured),
            cost_row("Refined query and dense-prf search", "online", laptop,
                     on["refine_and_dense_prf"], 0.0, measured),
        ]  # fmt: skip
        if system == cls.CANDIDATE:
            rows += [
                cost_row("Per-seed entity hops, summed over the seeds", "online", laptop,
                         on["seed_hops"], 0.0, measured),
                cost_row("RRF hop-ms", "online", laptop, on["rrf_hop_ms"], 0.0, measured),
                cost_row("RRF mch", "online", laptop, on["rrf_mch"], 0.0, measured),
            ]  # fmt: skip
        elif system == cls.CONTROL:
            rows.append(cost_row("RRF rrf-prf", "online", laptop, on["rrf_rrf_prf"], 0.0, measured))
        else:
            rows += [
                cost_row("Single-seed entity hop", "online", pooled["hardware"],
                         pooled["online_seconds_per_question"]["hop"], 0.0,
                         "seconds measured (Phase 04 pool manifest); USD 0 by assumption "
                         "(owned laptop, energy not counted)"),
                cost_row("RRF rrf-1s", "online", laptop, on["rrf_rrf_1s"], 0.0, measured),
            ]  # fmt: skip
        totals: dict[str, dict[str, float | None]] = {}
        for r in rows:
            if r["seconds"] is not None:
                group = [
                    x
                    for x in rows
                    if x["seconds"] is not None
                    and (x["kind"], x["hardware"]) == (r["kind"], r["hardware"])
                ]
                shared = any(x["usd"] is None for x in group)
                totals[f"{r['kind']} {r['hardware']}"] = {
                    "seconds": sum(x["seconds"] for x in group),
                    "usd": None if shared else sum(x["usd"] for x in group),
                }  # fmt: skip
        usd = {
            kind: sum(r["usd"] for r in rows if r["kind"] == kind and r["usd"] is not None)
            for kind in ("offline", "online")
        }
        online = {
            hw: sum(r["seconds"] for r in rows if r["kind"] == "online" and r["hardware"] == hw)
            for hw in sorted({r["hardware"] for r in rows if r["kind"] == "online"})
        }
        gpus = {hw: s for hw, s in online.items() if hw != laptop}
        laptop_online = online.get(laptop, 0.0)
        check: dict[str, Any] = {
            "laptop_online_seconds_per_question": laptop_online,
            "laptop_bound": cls.LAPTOP_BOUND,
            "gpu_online_seconds_per_question": gpus,
            "gpu_bound": cls.GPU_BOUND,
            "gpu_bound_stated_for": "one RTX 4090",
        }
        inside = laptop_online <= cls.LAPTOP_BOUND and all(
            s <= cls.GPU_BOUND for s in gpus.values()
        )
        if hop:
            per_million = cls.GLINER_OFFLINE[set_name][0] / 3600 / (units / 1_000_000)
            check.update(
                {
                    "gliner_gpu_hours_per_million_units": per_million,
                    "corpus_units": units,
                    "gliner_bound": cls.GLINER_BOUND,
                }
            )
            inside = inside and per_million <= cls.GLINER_BOUND
        check["inside_light_class"] = inside
        check["label"] = "derived; GPU and laptop seconds are checked apart, never summed"
        return {
            "components": rows,
            "totals_per_hardware": totals,
            "totals_label": "derived (sums of the components on one hardware; a row without "
            "seconds is not in the totals; a total with a row without USD has no USD)",
            "usd_total": {"offline": usd["offline"], "online_per_question": usd["online"]},
            "usd_total_label": "derived, lower bound: the Dense corpus embeddings and any USD "
            "shared with another job are not included; laptop USD is 0 by assumption",
            "class_check": check,
        }

    @classmethod
    def set_table(cls, set_name: str, phase02: Mapping[str, Any]) -> dict[str, Any]:
        spec = config.SETS[set_name]
        checks = Checks()
        old = OldData(spec.directory, spec.file_sha256, checks)
        questions = scoring.questions_of(old, checks, spec)
        counts = scoring.token_counts(old, checks, spec)
        here = config.PHASE05_RANKINGS_DIR / set_name
        p03, p04 = config.PHASE03_RANKINGS_DIR / set_name, config.PHASE04_RANKINGS_DIR / set_name
        conv = json.loads((here / converge.MANIFEST).read_text("utf-8"))
        if not converge_complete(conv):
            raise ArtifactError(f"{set_name}: converge equality counts are not complete (C1)")
        comp = json.loads((p03 / components.MANIFEST).read_text("utf-8"))
        pooled = json.loads((p04 / pool.MANIFEST).read_text("utf-8"))
        found, earlier_sha256 = cls.earlier(set_name)
        if found["g-l"][1]["rankings_sha256"] != config.GL_SHA256[set_name]:
            raise ArtifactError(f"{set_name}: Phase 02 results name another G-L ranking file")
        best_light, best = cls.best_of({s: r for s, (_, r) in found.items()}, cls.LIGHT)
        rows: dict[str, Any] = {}
        records: dict[str, list[int]] = {}
        ranked: dict[str, dict[str, list[str]]] = {}
        for system in (*cls.OWN, "g-l", best_light, best):
            if system in rows:
                continue
            if system in cls.OWN:
                name = f"{system}.jsonl.gz"
                ranked[system] = judge.pinned_rankings(here / name, conv["outputs_sha256"][name])
            else:
                phase, row = found[system]
                ranked[system] = cls.earlier_rankings(set_name, phase, system, row)
            summary, records[system] = scoring.score(questions, ranked[system], counts)
            rows[system] = {"metrics": summary, "metrics_label": "measured"}
            if system not in cls.OWN:
                phase, row = found[system]
                if summary != row["metrics"]:
                    raise ArtifactError(f"{set_name} {system}: rescored metrics differ ({phase})")
                cost = {**row["cost"]}
                system_manifest = config.RANKINGS_DIR / set_name / f"{system}.manifest.json"
                if phase == "Phase 02" and system_manifest.exists():
                    cost["hardware"] = json.loads(system_manifest.read_text("utf-8"))["gpu"]
                if handover_cost(set_name, system):
                    cost["handover"] = handover_cost(set_name, system)
                rows[system].update({"from": f"{phase} results.json", "cost": cost})
        for system in cls.FUSED:
            rows[system]["cost"] = cls.system_cost(
                system, set_name, comp, conv, pooled, len(counts)
            )
        notes = list(SET_NOTES.get(set_name, []))
        for system in rows:
            if system in IN_SAMPLE.get(set_name, ()):
                rows[system]["in_sample"] = True
                notes.append(f"{system} is in-sample on this set: its comparisons favour it.")
        # Exploratory: questions whose gold enters mch's top 100 only through hop-ms.
        p10a = judge.pinned_rankings(
            config.RANKINGS_DIR / set_name / "p10-a.jsonl.gz",
            conv["inputs_sha256"]["p10-a.jsonl.gz"],
        )
        bm25 = judge.pinned_rankings(p03 / "bm25.jsonl.gz", conv["inputs_sha256"]["bm25.jsonl.gz"])
        through_hop = sum(
            any(
                pool.hop_only(
                    [g],
                    [p10a[q.qid][: config.DEPTH], bm25[q.qid][: config.DEPTH],
                     ranked["dense-prf"][q.qid]],
                    ranked["hop-ms"][q.qid],
                )
                for g in q.gold_unit_ids
                if g in ranked[cls.CANDIDATE][q.qid][: config.DEPTH]
            )
            for q in questions
        )  # fmt: skip
        paired = []
        for system, against in cls.COMPARISONS:
            target = {"best_light_class_so_far": best_light, "best_so_far": best}.get(
                against, against
            )
            test = metrics.paired(records[target], records[system])
            paired.append({"system": system, "against": target, **test, "state": state(test),
                           "label": "measured"})  # fmt: skip
        return {
            "set": set_name,
            "n": len(questions),
            "best_light_class_so_far": best_light,
            "best_so_far": best,
            "best_read_from": "Phase 02-04 results.json, by FS@2,048 (spec selection rule)",
            "systems": rows,
            f"paired_full_support_at_{config.BUDGET}": paired,
            "mch_units_only_in_hop_ms": conv["mch_units_only_in_hop_ms"],
            "questions_with_gold_only_through_hop_ms": {"count": through_hop,
                                                        "label": "exploratory"},
            "converge": {
                "equality": conv["equality"],
                "seed_count": conv["seed_count"],
                "hop_ms_list_lengths": conv["hop_ms_list_lengths"],
                "peak_rss_mb": conv["peak_rss_mb"],
                "git_commit": conv["git_commit"],
                "label": "measured (Phase 05 converge manifest)",
            },
            "notes": notes,
            "earlier_results_sha256": earlier_sha256,
            "manifests_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (here / converge.MANIFEST, p03 / components.MANIFEST, p04 / pool.MANIFEST)
            },
            "digests_checked": checks.records,
        }  # fmt: skip

    @classmethod
    def outcome(cls, tables: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
        """States per set and comparison, the verdict, `rrf-prf`'s context verdict, the entrant."""
        states = {
            name: states_of(tables, system, against)
            for name, (system, against) in zip(cls.STATE_NAMES, cls.COMPARISONS, strict=True)
        }

        def of(i: int) -> list[str]:
            return list(states[cls.STATE_NAMES[i]].values())

        found = verdict(of(0), of(1), of(3), cls.CONTROL, "light-class")
        return {
            "states": states,
            "verdict": found,
            "rrf-prf_context_verdict": "advances" if advances(of(5)) else "does not advance",
            "exam_entrant": cls.CANDIDATE if found == "entrant" else "none from this phase",
            "label": "written by code under the frozen selection rule",
        }

    @classmethod
    def _systems(cls, entry: Mapping[str, Any]) -> list[str]:
        order = (*cls.OWN, "g-l", entry["best_light_class_so_far"], entry["best_so_far"])
        return list(dict.fromkeys(order))

    @classmethod
    def markdown(cls, table: Mapping[str, Any]) -> str:
        budget = str(config.BUDGET)
        out = table["outcome"]
        set_names = sorted(out["states"][cls.STATE_NAMES[0]])
        lines = [
            "# Phase 05 - Results",
            "",
            "Generated by `uv run edge-rag hop-results` from `data/phase05/results.json`; "
            "do not edit by hand.",
            "Metrics and paired tests are measured; states, verdicts and the class check are "
            "written by code; per-hardware totals are derived.",
            "",
            "Controls, said first: the advance gate G-L is weak, weaker than the best light-class "
            "systems already on disk; the working control `rrf-prf` (the candidate without its "
            "hop) is weaker than MDR, the strongest light-class literature recipe for multi-hop "
            "retrieval, which is not reproduced here (spec, Objective).",
            "",
            "## Outcome",
            "",
            f"- mch verdict: **{out['verdict']}**.",
            f"- Exam entrant: {out['exam_entrant']}.",
            "- rrf-prf against G-L under the same advance rule (context): "
            f"{out['rrf-prf_context_verdict']}.",
            "",
            "| Comparison | " + " | ".join(set_names) + " |",
            "|---|" + "---|" * len(set_names),
        ]
        for name in cls.STATE_NAMES:
            lines.append(
                f"| {name} | " + " | ".join(out["states"][name][s] for s in set_names) + " |"
            )
        lines.append("")
        for set_name in sorted(table["not_run"]):
            lines.append(f"- {set_name}: {NOT_RUN} ({table['not_run'][set_name]}).")
        for entry in sorted(table["sets"], key=lambda e: str(e["set"])):
            set_name = entry["set"]
            systems = cls._systems(entry)
            lines += [
                "",
                f"## {set_name} (n = {entry['n']})",
                "",
                "Best light-class system so far: "
                f"{shown(set_name, entry['best_light_class_so_far'])}; best system so far: "
                f"{shown(set_name, entry['best_so_far'])} (by FS@{budget}, read from the Phase "
                "02-04 results).",
            ]
            lines += [f"Note: {note}" for note in entry["notes"]]
            lines += [
                "",
                "| System | FS@1,024 | FS@2,048 | FS@4,096 | FS@2 | FS@5 | FS@20 | Share@5 "
                "| R@2 | R@5 | R@10 | R@20 | R@100 | nDCG@10 | Label |",
                "|---|" + "---:|" * 13 + "---|",
            ]
            for system in systems:
                row = entry["systems"][system]
                m = row["metrics"]
                at_b, at_k, r = (
                    m["full_support_at_budget"],
                    m["full_support_at_k"],
                    m["recall_at_k"],
                )
                recalls = " | ".join(f"{r[k]:.4f}" for k in ("2", "5", "10", "20", "100"))
                source = f"; scored as in {row['from']}" if "from" in row else ""
                lines.append(
                    f"| {shown(set_name, system)} | {at_b['1024']} | {at_b['2048']} "
                    f"| {at_b['4096']} | {at_k['2']} | {at_k['5']} | {at_k['20']} "
                    f"| {m['gold_share_at_5']:.4f} | {recalls} | {m['ndcg_at_10']:.4f} "
                    f"| {row['metrics_label']}{source} |"
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
                    f"| {t['system']} | {shown(set_name, t['against'])} | {t['wins']} "
                    f"| {t['losses']} | {t['ties']} | {t['p']:.4g} | {t['state']} |"
                )
            lines += [
                "",
                "Cost per component (seconds per question online, per corpus offline):",
                "",
                "| System | Component | Kind | Hardware | Seconds | USD | Label |",
                "|---|---|---|---|---:|---:|---|",
            ]
            for system in cls.FUSED:
                for c in entry["systems"][system]["cost"]["components"]:
                    lines.append(
                        f"| {system} | {c['component']} | {c['kind']} | {c['hardware']} "
                        f"| {num(c['seconds'], '.6g')} | {num(c['usd'], '.6f')} | {c['label']} |"
                    )
            first = entry["systems"][cls.CANDIDATE]["cost"]
            lines += ["", f"Totals per hardware, {first['totals_label']}:", ""]
            for system in cls.FUSED:
                cost = entry["systems"][system]["cost"]
                for name in sorted(cost["totals_per_hardware"]):
                    total = cost["totals_per_hardware"][name]
                    lines.append(
                        f"- {system}, {name}: {total['seconds']:.6g} s, "
                        f"{num(total['usd'], '.6f')} USD."
                    )
                usd = cost["usd_total"]
                lines.append(
                    f"- {system}, USD in total: offline {usd['offline']:.6f} USD, online "
                    f"{usd['online_per_question']:.6f} USD per question "
                    f"({cost['usd_total_label']})."
                )
                check = cost["class_check"]
                gpus = check["gpu_online_seconds_per_question"]
                gpu_text = (
                    ", ".join(f"{hw} {s:.4g} s/q" for hw, s in sorted(gpus.items()))
                    if gpus
                    else "no GPU component"
                )
                gliner = ""
                if "gliner_gpu_hours_per_million_units" in check:
                    gliner = (
                        f"; GLiNER offline {check['gliner_gpu_hours_per_million_units']:.3g} GPU-h "
                        f"per million units over {check['corpus_units']:,} units (bound "
                        f"{check['gliner_bound']})"
                    )
                where = "inside" if check["inside_light_class"] else "outside"
                lines.append(
                    f"- {system} class check: laptop online "
                    f"{check['laptop_online_seconds_per_question']:.4g} s/q (bound "
                    f"{check['laptop_bound']}); GPU online {gpu_text} (bound {check['gpu_bound']}, "
                    f"stated for {check['gpu_bound_stated_for']}){gliner}: {where} the light "
                    "class on this set (derived)."
                )
            lines += ["", "Cost of the reference rows (inherited labels):", ""]
            for system in systems:
                if system in cls.OWN:
                    continue
                row = entry["systems"][system]
                cost = row["cost"]
                text = cost.get("label", "")
                if "online_seconds_per_question" in cost:
                    text = (
                        f"offline {cost['offline_seconds']:.0f} s, {cost['offline_usd']:.3f} USD; "
                        f"online {cost['online_seconds_per_question']:.4f} s/q, "
                        f"{cost['online_usd_per_question']:.6f} USD/q on "
                        f"{cost.get('hardware', 'hardware not recorded')} ({cost['label']})"
                    )
                elif "components" in cost:
                    phase = row["from"].split(" results")[0]
                    text = f"per component in {cls.PAGES[phase]} ({phase})"
                lines.append(f"- {shown(set_name, system)}: {text}.")
                for note in cost.get("handover", []):
                    lines.append(f"  - {note['figure']} ({note['label']}; {note['source']}).")
            only = entry["mch_units_only_in_hop_ms"]
            conv = entry["converge"]
            eq = ", ".join(
                f"{name} {e['equal']:,} of {e['of']:,}"
                for name, e in sorted(conv["equality"].items())
            )
            seeds = conv["seed_count"]
            lengths = conv["hop_ms_list_lengths"]
            lines += [
                "",
                f"mch top-100 units contributed only by hop-ms: {only['total']} "
                f"({only['per_question']} per question, measured).",
                "Questions whose gold enters mch's top 100 only through hop-ms: "
                f"{entry['questions_with_gold_only_through_hop_ms']['count']} (exploratory).",
                f"Converge run: equality {eq}; seeds per question mean {seeds['mean']}, range "
                f"{seeds['min']}-{seeds['max']}; hop-ms lists empty {lengths['empty']}, short "
                f"{lengths['short']}; peak RSS {conv['peak_rss_mb']} MB (measured).",
            ]
        return "\n".join(lines) + "\n"


FULL_SET = "full set"
SUBSET = "the 1,000 preregistered qids (D11)"


def strongest(fs_by_ghost: Mapping[str, int | None]) -> str | None:
    """The measured ghost with the highest FS@2,048, the first of equals in class order."""
    measured = [g for g, value in fs_by_ghost.items() if value is not None]
    return max(measured, key=lambda g: fs_by_ghost[g] or 0) if measured else None


def literature_bar(
    class_name: str,
    ghost_records: Mapping[str, Sequence[int] | None],
    own: str,
    own_record: Sequence[int],
    on: str,
) -> dict[str, Any]:
    """Phase 06 spec, literature bar: the strongest measured ghost of one class (records are
    per-question FS@2,048 hits on `on`, None when the ghost was not run) and the state of the
    project's best own system of that class or a cheaper one against it."""
    fs_by = {g: None if r is None else sum(r) for g, r in ghost_records.items()}
    best = strongest(fs_by)
    test = None
    if best is not None:
        test = metrics.paired(ghost_records[best] or [], own_record)
    return {
        "class": class_name,
        "ghosts_fs": fs_by,
        "strongest_ghost": best,
        "own": own,
        "own_fs": sum(own_record),
        "on": on,
        **(test or {}),
        "state": state(test),
        "not_measured": [g for g, value in fs_by.items() if value is None],
        "label": "FS and paired test measured; strongest ghost and state written by code",
    }


def literature_claims(bars: Mapping[str, Mapping[str, str]]) -> list[str]:
    """Where "beats the literature" may be written: only a bar whose state is `win`."""
    return [f"class {c} on {s}" for s in sorted(bars) for c, found in bars[s].items()
            if found == "win"]  # fmt: skip


def cost_text(cost: Mapping[str, Any]) -> str:
    """One line for a stored cost of any phase: per-question figures, per-hardware totals or
    the inherited label."""
    if "online_seconds_per_question" in cost:
        text = (
            f"offline {cost['offline_seconds']:.0f} s, {cost['offline_usd']:.3f} USD; online "
            f"{cost['online_seconds_per_question']:.4f} s/q, "
            f"{cost['online_usd_per_question']:.6f} USD/q on "
            f"{cost.get('hardware', 'hardware not recorded')} ({cost['label']})"
        )
        if "with_g_l" in cost:
            g_l = cost["with_g_l"]
            text += (
                f"; with G-L's first stage {g_l['offline_usd']:.3f} USD offline and "
                f"{g_l['online_usd_per_question']:.6f} USD/q online ({g_l['label']})"
            )
        return text
    if "totals_per_hardware" in cost:
        parts = [
            f"{name} {t['seconds']:.4g} s, {num(t['usd'], '.6f')} USD"
            for name, t in sorted(cost["totals_per_hardware"].items())
        ]
        return "; ".join(parts) + (
            " (per hardware: offline per corpus, online per question; derived)"
        )
    return str(cost.get("label", "not recorded"))


def metric_line(name: str, m: Mapping[str, Any], label: str) -> str:
    at_b, at_k, r = m["full_support_at_budget"], m["full_support_at_k"], m["recall_at_k"]
    recalls = " | ".join(f"{r[k]:.4f}" for k in ("2", "5", "10", "20", "100"))
    return (
        f"| {name} | {at_b['1024']} | {at_b['2048']} | {at_b['4096']} | {at_k['2']} "
        f"| {at_k['5']} | {at_k['20']} | {m['gold_share_at_5']:.4f} | {recalls} "
        f"| {m['ndcg_at_10']:.4f} | {label} |"
    )


METRIC_HEADER = [
    "| System | FS@1,024 | FS@2,048 | FS@4,096 | FS@2 | FS@5 | FS@20 | Share@5 "
    "| R@2 | R@5 | R@10 | R@20 | R@100 | nDCG@10 | Label |",
    "|---|" + "---:|" * 13 + "---|",
]


class Phase06(Phase05):
    """Phase 06 (its spec C5, C6; plan D11, D12): the rivals G-R2, G-A1 on the HotpotQA 1,000
    and G-A2 against the best system so far, the project's best own system and the other ghost
    of their class, and the literature bar per set and class.

    The best systems are read from the stored Phase 02-05 results by FS@2,048 on the full set,
    the first of equals kept (phase order, then name order). On HotpotQA dev every comparison
    with a rival, and every bar whose ghosts include a rival scored only on the 1,000 qids, is
    measured on those qids for both sides; the full-set figures stay as they are.
    """

    RESULTS_JSON = config.PHASE06_DIR / "results.json"
    RESULTS_MD = config.REPO_ROOT / "docs" / "plans" / "fase-06-rivals" / "results.md"
    MISSING = "no Phase 06 rankings manifest"
    RIVALS = ("g-r2", "g-a1", "g-a2")
    GHOSTS = ("g-l", "g-r", "g-r2", "g-a1", "g-a2")
    CLASSES: ClassVar[dict[str, tuple[str, ...]]] = {
        "L": ("g-l",), "R": ("g-r", "g-r2"), "A": ("g-a1", "g-a2"),
    }  # fmt: skip
    CHEAPER: ClassVar[dict[str, tuple[str, ...]]] = {
        "L": ("L",), "R": ("L", "R"), "A": ("L", "R", "A"),
    }  # fmt: skip
    # The project's own systems by cost class: Phase 05's light-class list and its own systems
    # (class L); the J-strong reranked systems of the old line and Phase 04 (class R).
    OWN_CLASS: ClassVar[dict[str, str]] = {
        **{s: "L" for s in (*Phase05.LIGHT, *Phase05.OWN) if s != "g-l"},
        **{s: "R" for s in ("j-p10b", "j-union", "j-rrf3", "j-rrf4")},
    }
    OTHER_GHOST: ClassVar[dict[str, str]] = {"g-r2": "g-r", "g-a1": "g-a2", "g-a2": "g-a1"}
    NOT_RUN_REASONS: ClassVar[dict[tuple[str, str], str]] = {
        ("multihop-rag", "g-a2"): "cost gate: the probe projected 7.10 USD against G-A2's "
        "4.5 USD hard cut, so the session stopped before the full index (plan increment 4, "
        "deviation 06.3, D12)",
    }
    LIMITS = (
        "G-R2 on HotpotQA dev was scored on the 1,000 preregistered qids only, not on the full "
        "7,405 the spec named (deviation 06.2, D11: G-R2's measured pace projected past its "
        "hard cut).",
        "G-A1 on the full HotpotQA dev is outside the spec's scope; it ran on the 1,000 qids.",
        "G-A2 on MuSiQue and HotpotQA dev is outside the spec's scope.",
        "G-A1's figures come from a rerun after a GPU out-of-memory failure (deviation 06.4): "
        "the spec's stop rule (abort an item without retry under other settings when it runs "
        "out of memory once) was not followed as written; the rerun changed only the "
        "retrieval-server memory margin, to reach E3's planned vLLM share 0.46, and the "
        "author ratified the deviation on 2026-10-06 (plan, deviation 06.4).",
    )
    # Per set, systems whose training data covers the set (research protocol, known traps;
    # plan, C1 record): a claim on that set is printed with them beside the verdict.
    IN_DOMAIN: ClassVar[dict[str, dict[str, str]]] = {
        "hotpotqa-dev": {
            "g-l": "G-L (answerai-colbert-small-v1) is in-domain",
            "rrf4": "BGE-small, inside the fused own systems (rrf4, j-rrf4 and the other Dense "
            "fusions), was fine-tuned on HotpotQA train",
            "g-a1": "G-A1's Search-R1 checkpoint was trained on HotpotQA train",
        },
    }
    COMPARISONS = tuple(
        (r, a)
        for r in ("g-r2", "g-a1", "g-a2")
        for a in ("best_so_far", "best_own", {"g-r2": "g-r", "g-a1": "g-a2", "g-a2": "g-a1"}[r])
    )
    STATE_NAMES = tuple(f"{r} vs {a.replace('_', ' ')}" for r, a in COMPARISONS)
    EARLIER = (*Phase05.EARLIER, ("Phase 05", Phase05.RESULTS_JSON, config.PHASE05_RANKINGS_DIR))
    # Phase 05's results.json digest, measured when the Phase 06 results were written.
    EARLIER_SHA256: ClassVar[dict[str, str]] = {
        **Phase05.EARLIER_SHA256,
        "Phase 05": "e116991eb976f3a1647f6f29a37a844aee70e1305eb921c6b8d8e1a5fd858865",
    }
    PAGES: ClassVar[dict[str, str]] = {
        **Phase05.PAGES,
        "Phase 05": "docs/plans/fase-05-hop/results.md",
    }

    @classmethod
    def ran(cls, set_name: str) -> bool:
        return (config.PHASE06_RANKINGS_DIR / set_name / "manifest.json").exists()

    @classmethod
    def earlier_rankings(cls, set_name: str, phase: str, system: str, row: Mapping) -> Any:
        if phase != "Phase 05":
            return super().earlier_rankings(set_name, phase, system, row)
        directory = config.PHASE05_RANKINGS_DIR / set_name
        name = f"{system}.jsonl.gz"
        sha = json.loads((directory / converge.MANIFEST).read_text("utf-8"))["outputs_sha256"]
        return judge.pinned_rankings(directory / name, sha[name])

    @classmethod
    def bests(cls, earlier: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Best so far, best own and best own per class (that class or a cheaper one), by
        FS@2,048 from the earlier rows in their order (the first of equals is kept)."""
        own = [s for s in earlier if s not in cls.GHOSTS]
        unknown = [s for s in own if s not in cls.OWN_CLASS]
        if unknown:
            raise ArtifactError(f"own systems without a cost class: {unknown}")
        return {
            "best_so_far": max(earlier, key=lambda s: fs(earlier[s])),
            "best_own": max(own, key=lambda s: fs(earlier[s])),
            "best_own_by_class": {
                c: max((s for s in own if cls.OWN_CLASS[s] in cls.CHEAPER[c]),
                       key=lambda s: fs(earlier[s]))
                for c in cls.CLASSES
            },
        }  # fmt: skip

    @staticmethod
    def rival_cost(manifest: Mapping[str, Any], g_l: Mapping[str, Any] | None) -> dict:
        rate = manifest["pod"]["cost_per_hr_usd"]
        online = manifest["online_seconds_per_question"]
        cost: dict[str, Any] = {
            "hardware": manifest["pod"]["gpu"],
            "cost_per_hr_usd": rate,
            "offline_seconds": manifest["offline_seconds"],
            "offline_usd": manifest["offline_seconds"] * rate / 3600,
            "offline_note": manifest.get(
                "offline_note", "the G-L HotpotQA index rebuilt for this run (spec)"
            ),
            "online_seconds_per_question": online,
            "online_usd_per_question": online * rate / 3600,
            "label": "seconds measured (Phase 06 manifest); USD derived (time x rate)",
        }
        if g_l is not None:
            cost["with_g_l"] = {
                "offline_usd": cost["offline_usd"] + g_l["offline_usd"],
                "online_usd_per_question": cost["online_usd_per_question"]
                + g_l["online_usd_per_question"],
                "label": "derived: G-L's Phase 02 row added; seconds on two GPUs not summed",
            }
        return cost

    @classmethod
    def rivals_of(
        cls, set_name: str, here: Path
    ) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]], list[str]]:
        """Each rival run on the set: manifest, rankings pinned by both manifests, and which
        rivals were scored only on the preregistered qids (by the manifest's `subset` or
        `qids_sha256`)."""
        listed = json.loads((here / "manifest.json").read_text("utf-8"))
        manifests: dict[str, Any] = {}
        ranked: dict[str, dict[str, list[str]]] = {}
        subset_only: list[str] = []
        for rival in cls.RIVALS:
            path = here / f"{rival}.manifest.json"
            if not path.exists():
                continue
            manifest = json.loads(path.read_text("utf-8"))
            name = f"{rival}.jsonl.gz"
            if listed.get(name) != manifest["outputs_sha256"][name]:
                raise ArtifactError(f"{set_name} {rival}: manifests name other rankings")
            subset = (manifest.get("subset") or {}).get("sha256") or manifest.get("qids_sha256")
            if subset is not None:
                if subset != config.PHASE06_SUBSAMPLE_SHA256:
                    raise ArtifactError(f"{set_name} {rival}: not the preregistered qids")
                subset_only.append(rival)
            manifests[rival] = manifest
            ranked[rival] = judge.pinned_rankings(here / name, manifest["outputs_sha256"][name])
        return manifests, ranked, subset_only

    @classmethod
    def set_table(cls, set_name: str, phase02: Mapping[str, Any]) -> dict[str, Any]:
        spec = config.SETS[set_name]
        checks = Checks()
        old = OldData(spec.directory, spec.file_sha256, checks)
        questions = scoring.questions_of(old, checks, spec)
        counts = scoring.token_counts(old, checks, spec)
        here = config.PHASE06_RANKINGS_DIR / set_name
        found, earlier_sha256 = cls.earlier(set_name)
        if found["g-l"][1]["rankings_sha256"] != config.GL_SHA256[set_name]:
            raise ArtifactError(f"{set_name}: Phase 02 results name another G-L ranking file")
        bests = cls.bests({s: r for s, (_, r) in found.items()})
        manifests, ranked, subset_only = cls.rivals_of(set_name, here)
        subset_file = config.PHASE06_SUBSAMPLE_DIR / config.PHASE06_SUBSAMPLE_FILE
        qids: list[str] | None = None
        if subset_only:
            body = subset_file.read_bytes()
            if hashlib.sha256(body).hexdigest() != config.PHASE06_SUBSAMPLE_SHA256:
                raise ArtifactError(f"{subset_file}: not the pinned subsample")
            qids = body.decode("utf-8").split()
        earlier_systems = (
            "g-l", "g-r", "g-a1", bests["best_so_far"], bests["best_own"],
            *bests["best_own_by_class"].values(),
        )  # fmt: skip
        rows: dict[str, Any] = {}
        records: dict[str, dict[str, list[int]]] = {FULL_SET: {}, SUBSET: {}}
        for system in dict.fromkeys(earlier_systems):
            if system not in found:
                continue
            phase, row = found[system]
            ranked[system] = cls.earlier_rankings(set_name, phase, system, row)
            summary, records[FULL_SET][system] = scoring.score(questions, ranked[system], counts)
            if summary != row["metrics"]:
                raise ArtifactError(f"{set_name} {system}: rescored metrics differ ({phase})")
            cost = {**row["cost"]}
            system_manifest = config.RANKINGS_DIR / set_name / f"{system}.manifest.json"
            if phase == "Phase 02" and system_manifest.exists():
                cost["hardware"] = json.loads(system_manifest.read_text("utf-8"))["gpu"]
            if handover_cost(set_name, system):
                cost["handover"] = handover_cost(set_name, system)
            rows[system] = {"metrics": summary, "metrics_label": "measured",
                            "from": f"{phase} results.json", "cost": cost}  # fmt: skip
        for rival, manifest in manifests.items():
            g_l = rows["g-l"]["cost"] if rival == "g-r2" else None
            rows[rival] = {"from": "Phase 06 rankings", "cost": cls.rival_cost(manifest, g_l),
                           "questions": manifest["questions"]}  # fmt: skip
            if rival not in subset_only:
                rows[rival]["metrics"], records[FULL_SET][rival] = scoring.score(
                    questions, ranked[rival], counts
                )
                rows[rival]["metrics_label"] = "measured"
        for system, row in rows.items():
            if system in manifests:
                row["role"] = "rival"
            elif system in cls.GHOSTS:
                row["role"] = "ghost"
            else:
                row["role"] = f"own, class {cls.OWN_CLASS[system]}"
        subset_metrics: dict[str, Any] = {}
        if qids is not None:
            wanted = set(qids)
            sub_questions = [q for q in questions if q.qid in wanted]
            if len(sub_questions) != len(wanted):
                raise ArtifactError(f"{set_name}: subsample qids missing from the questions")
            for system in rows:
                lists = {q: ranked[system][q] for q in wanted}
                subset_metrics[system], records[SUBSET][system] = scoring.score(
                    sub_questions, lists, counts
                )

        def on_for(systems: Sequence[str]) -> str:
            return SUBSET if any(s in subset_only for s in systems) else FULL_SET

        paired = []
        for name, (rival, against) in zip(cls.STATE_NAMES, cls.COMPARISONS, strict=True):
            target = bests.get(against, against)
            entry: dict[str, Any] = {"name": name, "system": rival, "against": target}
            if target == rival:
                paired.append({**entry, "state": NOT_RUN, "reason": cls.self_reason(name)})
                continue
            missing = rival if rival not in manifests else target if target not in rows else None
            if missing is not None:
                reason = cls.NOT_RUN_REASONS.get(
                    (set_name, missing), f"{missing} is outside the spec's scope on this set"
                )
                paired.append({**entry, "state": NOT_RUN, "reason": reason})
                continue
            on = on_for((rival, target))
            a, b = records[on][target], records[on][rival]
            test = metrics.paired(a, b)
            paired.append({**entry, "on": on, "system_fs": sum(b), "against_fs": sum(a),
                           "n": len(a), **test, "state": state(test),
                           "label": "measured; state written by code"})  # fmt: skip
        bars = []
        for class_name, ghosts in cls.CLASSES.items():
            on = on_for([g for g in ghosts if g in rows])
            own = bests["best_own_by_class"][class_name]
            bar = literature_bar(
                class_name, {g: records[on].get(g) for g in ghosts}, own, records[on][own], on
            )
            bar["not_run_in_scope"] = [
                g for g in bar["not_measured"] if (set_name, g) in cls.NOT_RUN_REASONS
            ]
            bars.append(bar)
        notes = list(SET_NOTES.get(set_name, []))
        for system in rows:
            if system in IN_SAMPLE.get(set_name, ()):
                rows[system]["in_sample"] = True
                notes.append(f"{system} is in-sample on this set: its comparisons favour it.")
        if "g-a1" in manifests:
            notes.append(
                "G-A1's Search-R1 checkpoint was trained on NQ and HotpotQA train: HotpotQA is "
                "in-domain for it (plan, C1 record); its evidence lists are short (retrieved "
                "passages only, not filled to the budget), so FS@k and recall at large k are "
                "bounded by them; the code follows Phase 02's Search-R1 driver, and the spec's "
                '"filled to the budget" wording is stale, with no effect on the figures (plan, '
                "notes)."
            )
        if "g-r2" in manifests:
            notes.append(
                "G-R2 reranks G-L's top 100: it has no index of its own, and its online cost "
                "comes on top of G-L's first stage."
            )
        return {
            "set": set_name,
            "n": len(questions),
            **bests,
            "best_read_from": "Phase 02-05 results.json, by FS@2,048 on the full set",
            "subset": None if qids is None else {
                "file": subset_file.relative_to(config.REPO_ROOT).as_posix(),
                "sha256": config.PHASE06_SUBSAMPLE_SHA256,
                "questions": len(qids),
                "systems_scored_only_here": subset_only,
                "label": "measured; every rival comparison on this set uses these qids for both "
                "sides",
            },
            "systems": rows,
            "subset_metrics": subset_metrics,
            f"paired_full_support_at_{config.BUDGET}": paired,
            "literature_bar": bars,
            "in_domain": [
                note for system, note in cls.IN_DOMAIN.get(set_name, {}).items() if system in rows
            ],
            "notes": notes,
            "earlier_results_sha256": earlier_sha256,
            "manifests_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(here.glob("*manifest.json"))
            },
            "digests_checked": checks.records,
        }  # fmt: skip

    @classmethod
    def outcome(cls, tables: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
        """States per comparison and set, bar states per set and class, and the claims."""
        states: dict[str, dict[str, str]] = {name: {} for name in cls.STATE_NAMES}
        bars: dict[str, dict[str, str]] = {}
        for set_name, table in tables.items():
            for name in cls.STATE_NAMES:
                states[name][set_name] = NOT_RUN
            bars[set_name] = {c: NOT_RUN for c in cls.CLASSES}
            if table is None:
                continue
            for t in table[f"paired_full_support_at_{config.BUDGET}"]:
                states[t["name"]][set_name] = t["state"]
            for bar in table["literature_bar"]:
                bars[set_name][bar["class"]] = bar["state"]
        return {
            "states": states,
            "literature_bar": bars,
            "beats_the_literature": literature_claims(bars),
            "label": "written by code under the frozen rule (spec, comparisons and the bar)",
        }

    @staticmethod
    def self_reason(name: str) -> str:
        rival, against = name.split(" vs ")
        role = {"best so far": "best system so far", "best own": "best own system"}[against]
        return f"{rival} is itself the {role} on this set; no self-comparison"

    @classmethod
    def claim_caveats(cls, entry: Mapping[str, Any]) -> list[str]:
        """Beside the verdict: per won bar, the ghosts it was not measured against, and per set
        with a won bar, the systems in-domain on it."""
        lines = []
        won = [bar for bar in entry["literature_bar"] if bar["state"] == "win"]
        for bar in won:
            if bar["not_measured"]:
                lines.append(
                    f"The class {bar['class']} claim on {entry['set']} stands on measured "
                    f"ghosts only ({bar['strongest_ghost']}): "
                    f"{', '.join(bar['not_measured'])} {NOT_RUN}."
                )
        if won and entry.get("in_domain"):
            lines.append(
                f"The claims on {entry['set']} carry in-domain caveats: "
                + "; ".join(entry["in_domain"])
                + "."
            )
        return lines

    @classmethod
    def verdict_lines(cls, entry: Mapping[str, Any]) -> list[str]:
        """Per rival on one set: against the best own system first, then the other two."""
        set_name, systems = entry["set"], entry["systems"]
        tests = entry[f"paired_full_support_at_{config.BUDGET}"]
        lines = []
        for rival in cls.RIVALS:
            mine = [t for t in tests if t["system"] == rival]
            if mine[0]["state"] == NOT_RUN:
                if (set_name, rival) in cls.NOT_RUN_REASONS:
                    lines.append(f"- **{rival} on {set_name}: {NOT_RUN}** ({mine[0]['reason']}).")
                continue
            own = next(t for t in mine if t["name"] == f"{rival} vs best own")
            role = {"best so far": "the best system so far", "best own": "the best own system"}
            others = "; ".join(
                f"against {role.get(t['name'].split(' vs ')[1], 'the other ghost of its class')} "
                f"{t['against']}"
                + (f" ({t['against_fs']}): {t['state']}, {t['wins']} wins, {t['losses']} "
                   f"losses, p {t['p']:.3g}" if t["state"] != NOT_RUN
                   else f": {NOT_RUN} ({t['reason']})")
                for t in mine
                if t is not own
            )  # fmt: skip
            lines += [
                f"- **{rival} on {set_name}** (on {own['on']}, n {own['n']:,}): FS@2,048 "
                f"{own['system_fs']} against {own['against_fs']} for {own['against']}, the "
                f"project's best own system: {own['wins']} wins, {own['losses']} losses, exact p "
                f"{own['p']:.3g}, **{own['state']}**.",
                f"  Also {others}.",
                f"  Cost of {rival}: {cost_text(systems[rival]['cost'])}.",
                f"  Cost of {own['against']}: {cost_text(systems[own['against']]['cost'])}.",
            ]
        return lines

    @classmethod
    def markdown(cls, table: Mapping[str, Any]) -> str:
        out = table["outcome"]
        set_names = sorted(out["literature_bar"])
        entries = sorted(table["sets"], key=lambda e: str(e["set"]))
        claims = out["beats_the_literature"]
        lines = [
            "# Phase 06 - Results",
            "",
            "Generated by `uv run edge-rag rivals-results` from `data/phase06/results.json`; "
            "do not edit by hand.",
            "Metrics and paired tests are measured; USD is derived (time x rate); states, the "
            "literature bar and the claims are written by code.",
            "",
            "Controls, said first: two of this phase's comparisons are weaker than the spec "
            "intended.",
            "G-A2 (HippoRAG 2), the class A rival on MultiHop-RAG, was not run, so the class A "
            "bar there stands on G-A1 alone.",
            "On HotpotQA dev, G-R2 and G-A1 were scored on the 1,000 preregistered qids only, so "
            "every rival comparison there and the R and A bars cover those 1,000 questions for "
            "both sides, not the full 7,405.",
            "G-L, the class L bar, is weaker than the light systems already on disk, and MDR, the "
            "strongest light-class literature recipe, is not reproduced (Phase 05).",
            "",
            "## Verdict",
            "",
            "Beats the literature (only where a bar's state is `win`): "
            + (", ".join(claims) if claims else "nowhere")
            + ".",
        ]
        for entry in entries:
            lines += cls.claim_caveats(entry)
        lines += ["", "Each rival against the project's best own system, per set:", ""]
        for rival in cls.RIVALS:
            per_set = out["states"][f"{rival} vs best own"]
            lines.append(f"- {rival}: " + ", ".join(f"{s} {per_set[s]}" for s in set_names) + ".")
        lines.append("")
        for entry in entries:
            lines += cls.verdict_lines(entry)
        lines += ["", "## Not run", ""]
        for set_name in sorted(table["not_run"]):
            lines.append(f"- {set_name}: {NOT_RUN} ({table['not_run'][set_name]}).")
        for (set_name, rival), reason in sorted(cls.NOT_RUN_REASONS.items()):
            lines.append(f"- {rival} on {set_name}: {NOT_RUN} ({reason}).")
        lines += [f"- {limit}" for limit in cls.LIMITS]
        lines += [
            "",
            "## Comparisons",
            "",
            "Exact McNemar on per-question FS@2,048, two-sided, alpha 0.05 per comparison, no "
            "correction.",
            "",
            "| Comparison | " + " | ".join(set_names) + " |",
            "|---|" + "---|" * len(set_names),
        ]
        for name in cls.STATE_NAMES:
            lines.append(
                f"| {name} | " + " | ".join(out["states"][name][s] for s in set_names) + " |"
            )
        lines += [
            "",
            "## Literature bar",
            "",
            "Per set and class, the strongest measured ghost by FS@2,048 and the state of the "
            "project's best own system of that class or a cheaper one against it.",
            "",
            "| Set | Class | Ghosts (FS@2,048) | Strongest | Own system (FS@2,048) | Wins | Losses "
            "| Ties | Exact p | On | State |",
            "|---|---|---|---|---|---:|---:|---:|---:|---|---|",
        ]
        for entry in entries:
            for bar in entry["literature_bar"]:
                ghosts = ", ".join(
                    f"{g} {NOT_RUN if bar['ghosts_fs'][g] is None else bar['ghosts_fs'][g]}"
                    for g in cls.CLASSES[bar["class"]]
                )
                tested = bar["state"] != NOT_RUN
                lines.append(
                    f"| {entry['set']} | {bar['class']} | {ghosts} | {bar['strongest_ghost']} "
                    f"| {shown(entry['set'], bar['own'])} ({bar['own_fs']}) "
                    f"| {bar['wins'] if tested else '-'} | {bar['losses'] if tested else '-'} "
                    f"| {bar['ties'] if tested else '-'} "
                    f"| {format(bar['p'], '.4g') if tested else '-'} | {bar['on']} "
                    f"| {bar['state']} |"
                )
        lines.append("")
        for entry in entries:
            for bar in entry["literature_bar"]:
                for g in bar["not_run_in_scope"]:
                    lines.append(
                        f"- {entry['set']}, class {bar['class']}: measured ghosts only; {g} was "
                        f"{NOT_RUN} ({cls.NOT_RUN_REASONS[(entry['set'], g)]})."
                    )
        for entry in entries:
            lines += cls.set_section(entry)
        return "\n".join(lines) + "\n"

    @classmethod
    def order(cls, systems: Mapping[str, Any]) -> list[str]:
        """Rivals, then ghosts, then own systems, each in a fixed order (JSON sorts keys)."""
        fixed = (*cls.RIVALS, *cls.GHOSTS)
        return sorted(systems, key=lambda s: (fixed.index(s) if s in fixed else len(fixed), s))

    @classmethod
    def set_section(cls, entry: Mapping[str, Any]) -> list[str]:
        set_name, systems = entry["set"], entry["systems"]
        by_class = ", ".join(f"{c} {entry['best_own_by_class'][c]}" for c in cls.CLASSES)
        lines = [
            "",
            f"## {set_name} (n = {entry['n']})",
            "",
            f"Best system so far: {shown(set_name, entry['best_so_far'])}; best own system: "
            f"{shown(set_name, entry['best_own'])}; best own by class or cheaper: {by_class} "
            "(by FS@2,048 on the full set, read from the Phase 02-05 results).",
        ]
        lines += [f"Note: {note}" for note in entry["notes"]]
        order = cls.order(systems)
        full = [s for s in order if "metrics" in systems[s]]
        lines += ["", f"Full set ({entry['n']:,} questions):", "", *METRIC_HEADER]
        for system in full:
            row = systems[system]
            source = f"; scored as in {row['from']}" if row["from"] != "Phase 06 rankings" else ""
            lines.append(
                metric_line(
                    f"{shown(set_name, system)} ({row['role']})",
                    row["metrics"],
                    f"{row['metrics_label']}{source}",
                )
            )
        if entry["subset"] is not None:
            sub = entry["subset"]
            lines += [
                "",
                f"On {SUBSET} (`{sub['file']}`, sha256 {sub['sha256']}; scored only here: "
                f"{', '.join(sub['systems_scored_only_here'])}):",
                "",
                *METRIC_HEADER,
            ]
            for system in order:
                m = entry["subset_metrics"][system]
                name = f"{shown(set_name, system)} ({systems[system]['role']})"
                lines.append(metric_line(name, m, "measured"))
        lines += [
            "",
            "Paired exact McNemar on FS@2,048:",
            "",
            "| Comparison | Against | On | System FS | Against FS | Wins | Losses | Ties "
            "| Exact p | State |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for t in entry[f"paired_full_support_at_{config.BUDGET}"]:
            if t["state"] == NOT_RUN:
                lines.append(
                    f"| {t['name']} | {t['against']} | - | - | - | - | - | - | - "
                    f"| {NOT_RUN}: {t['reason']} |"
                )
                continue
            lines.append(
                f"| {t['name']} | {shown(set_name, t['against'])} | {t['on']} "
                f"| {t['system_fs']} | {t['against_fs']} | {t['wins']} | {t['losses']} "
                f"| {t['ties']} | {t['p']:.4g} | {t['state']} |"
            )
        lines += ["", "Cost (offline per corpus, online per question):", ""]
        for system in order:
            row = systems[system]
            cost = row["cost"]
            text = cost_text(cost)
            if "components" in cost and row["from"] != "Phase 06 rankings":
                phase = row["from"].split(" results")[0]
                text += f"; per component in {cls.PAGES[phase]}"
            lines.append(f"- {shown(set_name, system)} ({row['role']}): {text}.")
            if "offline_note" in cost:
                lines.append(f"  - Offline: {cost['offline_note']}.")
            for note in cost.get("handover", []):
                lines.append(f"  - {note['figure']} ({note['label']}; {note['source']}).")
        return lines
