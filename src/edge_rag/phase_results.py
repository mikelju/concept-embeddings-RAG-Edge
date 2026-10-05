"""Results tables of Phases 03, 04 and 05 (Phase 05 spec C3-C6; plan D3, D4): one module.

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
        if out is None:
            write_pair(table, cls.markdown, cls.RESULTS_JSON, cls.RESULTS_MD)
        else:
            write_pair(table, cls.markdown, out / "results.json", out / "results.md")
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
    COMPARISONS = (
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
    EARLIER = (
        ("Phase 02", PHASE02_RESULTS, config.RANKINGS_DIR),
        ("Phase 03", Phase03.RESULTS_JSON, config.PHASE03_RANKINGS_DIR),
        ("Phase 04", Phase04.RESULTS_JSON, config.PHASE04_RANKINGS_DIR),
    )
    PAGES = {
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
            sha256[str(path.relative_to(config.REPO_ROOT).as_posix())] = hashlib.sha256(
                body
            ).hexdigest()
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
