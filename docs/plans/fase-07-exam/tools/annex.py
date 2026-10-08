"""Phase 07 deviation 07.5: the exam annex, written without touching src/, scripts/ or tests/.

Reads the scored rankings in data/phase07/exam-rankings/, the test gold, the upstream
manifests and results.json; writes data/phase07/annex.json and
docs/plans/fase-07-exam/annex.md. It never writes results.json or results.md.
Run: uv run python docs/plans/fase-07-exam/tools/annex.py
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from edge_rag import config, metrics, qasper, scoring
from edge_rag import phase_results as pr

P7 = config.DATA_DIR / "phase07"
RANKINGS = P7 / "exam-rankings"
LAPTOP = P7 / "test" / "laptop" / "pooled"
POD = P7 / "pod"
OUT_JSON = P7 / "annex.json"
OUT_MD = config.REPO_ROOT / "docs" / "plans" / "fase-07-exam" / "annex.md"
OWN = ("rrf4", "j-rrf4", "j-rrf3", "p10-b", "p14")
GHOSTS = tuple(pr.EXAM_GHOSTS)
LAPTOP_HW = "laptop CPU (ARM64, Windows)"
RTX = "NVIDIA GeForce RTX 4090"
# The 4090 manifests carry no rate; plan.md task 9 records the offered rate at launch.
RTX_RATE = 0.74
RTX_RATE_SOURCE = "plan.md task 9 (offered rate; the 4090 manifests carry none)"
NOT_RECORDED = "not recorded"
INPUTS: dict[str, str] = {}


def load(path: Path) -> Any:
    """Read a JSON file and record its sha256 as an input of the annex."""
    INPUTS[path.relative_to(P7).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text("utf-8"))


def part(name: str, kind: str, hardware: str, seconds: float, rate: float | None,
         source: str) -> dict[str, Any]:  # fmt: skip
    """One measured component; USD 0 on the laptop by assumption, seconds x rate on a pod."""
    usd = 0.0 if rate is None else round(seconds * rate / 3600, 6)
    return {"name": name, "kind": kind, "hardware": hardware, "seconds": round(seconds, 3),
            "rate_usd_per_hour": rate, "usd": usd, "seconds_label": "measured",
            "usd_label": pr.LAPTOP_USD if rate is None else "derived (seconds x rate)",
            "source": source}  # fmt: skip


def gap(name: str, why: str) -> dict[str, Any]:
    return {"name": name, "kind": NOT_RECORDED, "why": why}


def components() -> dict[str, dict[str, Any]]:
    """Every cost component the manifests and logs record, measured, keyed by short name."""
    log = (P7 / "test-laptop.log").read_text("utf-8")
    INPUTS["test-laptop.log"] = hashlib.sha256(log.encode("utf-8")).hexdigest()
    summary = json.loads(re.findall(r"^\{.*\"encode_units\".*\}$", log, re.M)[-1])["seconds"]
    gliner = load(P7 / "test" / "gliner" / "extraction.json")
    names = ("dense", "bm25", "p10-b", "hop", "p14")
    marks = {s: load(LAPTOP / f"{s}.complete.json") for s in names}
    c = {
        "encode-units": part("laptop: encode 22,588 test units (dense vectors)", "offline",
                             LAPTOP_HW, summary["encode_units"], None, "test-laptop.log summary"),
        "index": part("laptop: dense, BM25 and entity-hop indexes", "offline", LAPTOP_HW,
                      marks["dense"]["index_seconds"], None, "test/laptop/pooled/*.complete.json"),
        "gliner": part("laptop: GLiNER extraction over the test units, model load included",
                       "offline", LAPTOP_HW, gliner["seconds"] + gliner["model_load_seconds"],
                       None, "test/gliner/extraction.json"),
        "encode-queries": part("laptop: encode all 2,902 queries (pooled and within, counted in "
                               "full)", "online", LAPTOP_HW, summary["encode_queries"], None,
                               "test-laptop.log summary"),
    }  # fmt: skip
    for s, m in marks.items():
        c[f"online-{s}"] = part(f"laptop: {s} retrieval step, 1,451 questions", "online",
                                LAPTOP_HW, m["online_seconds"], None,
                                f"test/laptop/pooled/{s}.complete.json")  # fmt: skip
    g_l = load(POD / "g-l" / "g-l.manifest.json")
    c["g-l"] = part("pod: G-L ColBERT PLAID index and 1,451 queries (one figure, not split)",
                    "offline and online, not split", RTX, g_l["seconds"], RTX_RATE,
                    f"pod/g-l/g-l.manifest.json; rate {RTX_RATE_SOURCE}")  # fmt: skip
    for system, path in (("g-r", "j-strong/g-r"), ("j-rrf4", "j-strong/j-rrf4"),
                         ("g-r2", "g-r2/g-r2"), ("j-rrf3", "j-rrf3/j-rrf3")):  # fmt: skip
        m = load(POD / f"{path}.manifest.json")
        src = f"pod/{path}.manifest.json; rate {RTX_RATE_SOURCE}"
        c[f"load-{system}"] = part(f"pod: {m['model']} load", "offline", m["gpu"],
                                   m["load_seconds"], RTX_RATE, src)  # fmt: skip
        c[f"rerank-{system}"] = part(f"pod: {system} rerank of {m['pairs']:,} pairs", "online",
                                     m["gpu"], m["seconds"], RTX_RATE, src)  # fmt: skip
        c[f"first-stage-{system}"] = {"inputs": sorted(m["first_stage"])}
    a1 = load(POD / "g-a1" / "rankings" / "qasper" / "g-a1.manifest.json")
    src = "pod/g-a1/rankings/qasper/g-a1.manifest.json"
    c["load-g-a1"] = part(f"pod: {a1['model']} load", "offline", a1["gpu"],
                          a1["seconds"]["load_model"], a1["cost_per_hr_usd"], src)  # fmt: skip
    c["run-g-a1"] = part(f"pod: G-A1 agent run, 1,451 questions, {a1['searches_total']:,} "
                         "searches", "online", a1["gpu"], a1["seconds"]["run"],
                         a1["cost_per_hr_usd"], src)  # fmt: skip
    c["run-g-a1"]["manifest_seconds_per_question"] = round(a1["seconds_per_question"], 6)
    return c


LAPTOP_PARTS = {
    "dense": ["encode-units", "index", "encode-queries", "online-dense"],
    "bm25": ["index", "online-bm25"],
    "p10-b": ["encode-units", "index", "encode-queries", "online-dense", "online-bm25",
              "online-p10-b"],
    "hop": ["encode-units", "gliner", "index", "encode-queries", "online-dense", "online-hop"],
    "p14": ["encode-units", "gliner", "index", "encode-queries", "online-dense", "online-bm25",
            "online-p14"],
    "G-L": ["g-l"],
}  # fmt: skip
FIRST_STAGE = {"dense": ["online-dense"], "bm25": ["online-bm25"], "hop": ["online-hop"],
               "g-l": ["g-l"]}  # fmt: skip


def system_parts(system: str, c: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    """The recorded components of a line-up system and the parts with no record."""
    first = ["encode-units", "gliner", "index", "encode-queries"]
    if system in LAPTOP_PARTS:
        return LAPTOP_PARTS[system], []
    if system == "rrf4":
        names = first + [p for s in ("dense", "bm25", "g-l", "hop") for p in FIRST_STAGE[s]]
        return names, [gap("rrf4 fusion on the laptop", "built by tools/assemble.py "
                           "(deviation 07.4) with no timing")]  # fmt: skip
    if system == "G-A1":
        return ["load-g-a1", "run-g-a1"], [gap(
            "G-L PLAID index the G-A1 retriever served on the A100 pod",
            "the g-a1 manifest times model load and run only")]  # fmt: skip
    key = {"G-R": "g-r", "G-R2": "g-r2", "j-rrf4": "j-rrf4", "j-rrf3": "j-rrf3"}[system]
    inputs = c[f"first-stage-{key}"]["inputs"]
    names = first + [p for s in inputs for p in FIRST_STAGE[s]]
    names += [f"load-{key}", f"rerank-{key}"]
    return names, [gap("first-stage fusion on the pod", f"not timed apart in the {key} "
                       "manifest (first-stage inputs as the manifest records them: "
                       + ", ".join(inputs) + ")")]  # fmt: skip


def cost_table(c: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for system in pr.EXAM_ORDER:
        names, gaps = system_parts(system, c)
        sums = {k: round(sum(c[n]["seconds"] for n in names if c[n]["kind"] == k), 3)
                for k in ("offline", "online", "offline and online, not split")}  # fmt: skip
        hardware = sorted({c[n]["hardware"] for n in names})
        rows[system] = {
            "components": names,
            "offline_seconds": sums["offline"],
            "online_seconds": sums["online"],
            "unsplit_seconds": sums["offline and online, not split"],
            "online_seconds_per_question": round(sums["online"] / pr.EXAM_QUESTIONS, 6),
            "hardware": hardware,
            "usd": round(sum(c[n]["usd"] for n in names), 6),
            "label": "derived (sum of measured components)",
            "not_recorded": gaps,
        }
    return rows


def main() -> None:
    gold, tokens = qasper.read_gold(P7 / "test")
    for name in ("gold.json", "units.jsonl"):
        path = P7 / "test" / name
        INPUTS[f"test/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    results = load(P7 / "results.json")
    for path in sorted(RANKINGS.rglob("*.jsonl.gz")):
        INPUTS[path.relative_to(P7).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    pooled = pr.exam_records(RANKINGS / "pooled", tokens, gold, pr.EXAM_QUESTIONS)
    within = pr.exam_records(RANKINGS / "within", tokens, gold, pr.EXAM_QUESTIONS)
    kinds = pr.exam_kinds(gold)
    qids = sorted(kinds)

    check, bad = {}, []
    for system, row in sorted(results["rows"].items()):
        found = {"fs": pooled.get(system), "fs_within": within.get(system)}
        mine = {
            "fs": None if found["fs"] is None else sum(found["fs"].values()),
            "fs_single": None if found["fs"] is None else sum(
                v for q, v in found["fs"].items() if kinds[q] == "single"),
            "fs_multi": None if found["fs"] is None else sum(
                v for q, v in found["fs"].items() if kinds[q] == "multi"),
            "fs_within": None if found["fs_within"] is None else sum(
                found["fs_within"].values()),
        }  # fmt: skip
        check[system] = mine
        bad += [f"{system} {k}: annex {v} vs results.json {row[k]}"
                for k, v in mine.items() if v != row[k]]  # fmt: skip
    if bad or len(qids) != results["n"]:
        raise SystemExit("[ERROR] FS@2,048 differs from results.json: " + "; ".join(bad))

    tests = []
    for own in OWN:
        for ghost in GHOSTS:
            a = [pooled[ghost][q] for q in qids]
            b = [pooled[own][q] for q in qids]
            t = metrics.paired(a, b)
            tests.append({"system": own, "against": ghost, "fs": sum(b), "fs_against": sum(a),
                          **t, "state": pr.state(t)})  # fmt: skip

    hop = scoring.read_rankings(RANKINGS / "pooled" / "hop.jsonl.gz")
    empty = {s: sum(1 for r in scoring.read_rankings(p).values() if not r)
             for s, p in ((p.name.removesuffix(".jsonl.gz"), p)
                          for p in sorted((RANKINGS / "pooled").glob("*.jsonl.gz")))}  # fmt: skip
    c = components()
    table = {
        "generated_by": "docs/plans/fase-07-exam/tools/annex.py",
        "deviation": "07.5",
        "n_in_scope": len(qids),
        "fs_check": {"state": "equal to results.json", "systems": check},
        "context_mcnemar": tests,
        "empty_rankings": {
            "hop": {"empty": empty["hop"], "rankings": len(hop)},
            "all_pooled": empty,
            "method": "scoring.read_rankings on exam-rankings/pooled/<system>.jsonl.gz; a ranking "
            "is empty when its unit list has no element",
        },
        "components": {k: v for k, v in c.items() if not k.startswith("first-stage-")},
        "costs": cost_table(c),
        "inputs_sha256": dict(sorted(INPUTS.items())),
    }
    pr.write_pair(table, markdown, OUT_JSON, OUT_MD)
    print(f"[OK] {OUT_JSON}")
    print(f"[OK] {OUT_MD}")


def seconds(value: float) -> str:
    return format(value, ",.3f")


def markdown(t: dict[str, Any]) -> str:
    lines = [
        "<!-- Generated by docs/plans/fase-07-exam/tools/annex.py under deviation 07.5; "
        "do not edit by hand. -->",
        "",
        "# Phase 07 - Exam annex: cost, context tests and evidence",
        "",
        "Generated by `docs/plans/fase-07-exam/tools/annex.py` under deviation 07.5 from "
        "`data/phase07/annex.json`; not editable by hand.",
        "It adds context to `results.md` and changes no figure, state or verdict there.",
        "Every input file and its sha256 is listed in `annex.json` under `inputs_sha256`.",
        "",
        "## FS@2,048 sanity check",
        "",
        f"The annex recomputes FS@2,048 on the {t['n_in_scope']:,} in-scope questions with the "
        "scorer's own functions (`exam_records`, `exam_kinds`).",
        f"Result: {t['fs_check']['state']} for every system (total, single, multi and "
        "within-paper).",
        "",
        "| System | FS@2,048 | single | multi | within |",
        "|---|---|---|---|---|",
    ]
    for s, r in t["fs_check"]["systems"].items():
        cells = [pr.num(r[k], "d") for k in ("fs", "fs_single", "fs_multi", "fs_within")]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Context exact McNemar (context, not a verdict)",
        "",
        "Every own system against every ghost on the same in-scope questions, per-question "
        "FS@2,048 as the scorer defines it, `metrics.paired` (exact, two-sided).",
        "Wins and losses are the own system's; no correction for the number of tests.",
        "These are context, not a verdict: the verdict rule and its states are in `results.md`.",
        "",
        "| Own | Ghost | FS own | FS ghost | wins | losses | ties | p | state |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for x in t["context_mcnemar"]:
        lines.append(f"| {x['system']} | {x['against']} | {x['fs']} | {x['fs_against']} | "
                     f"{x['wins']} | {x['losses']} | {x['ties']} | {x['p']:.4g} | "
                     f"{x['state']} |")  # fmt: skip
    e = t["empty_rankings"]
    lines += [
        "",
        "## Empty rankings",
        "",
        f"`hop` has {e['hop']['empty']} empty rankings of {e['hop']['rankings']:,}.",
        f"Method: {e['method']}.",
        "Other pooled systems: " + ", ".join(
            f"{s} {n}" for s, n in sorted(e["all_pooled"].items()) if s != "hop") + ".",
        "",
        "## Cost per line-up system",
        "",
        "Component seconds are measured (manifests, completion markers, the laptop log).",
        "Sums per system and every USD figure are derived: laptop USD is 0 by assumption "
        "(owned laptop, energy not counted); pod USD is seconds x rate.",
        "Query encoding on the laptop is counted in full for each system that needs it.",
        "Pod idle, setup and transfer time are not apportioned here; the whole-pod money figures "
        "are in `plan.md`, tasks 9 and 10.",
        "Anything with no record is written `not recorded`, never estimated.",
        "",
        "| System | offline s | online s | not split s | online s/q | hardware | USD | "
        "label | not recorded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    for s in pr.EXAM_ORDER:
        r = t["costs"][s]
        gaps = "; ".join(f"{g['name']}: {g['why']}" for g in r["not_recorded"]) or "-"
        lines.append(
            f"| {s} | {seconds(r['offline_seconds'])} | {seconds(r['online_seconds'])} | "
            f"{seconds(r['unsplit_seconds'])} | {r['online_seconds_per_question']:.4f} | "
            f"{', '.join(r['hardware'])} | {r['usd']:.4f} | {r['label']} | {gaps} |"
        )
    lines += [
        "",
        "### Components",
        "",
        "| Component | kind | hardware | seconds | rate USD/h | USD | source |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, p in sorted(t["components"].items()):
        usd = (
            "0 (laptop, by assumption)"
            if p["rate_usd_per_hour"] is None
            else (f"{p['usd']:.6f} (derived)")
        )
        lines.append(
            f"| `{k}` {p['name']} | {p['kind']} | {p['hardware']} | "
            f"{seconds(p['seconds'])} (measured) | "
            f"{pr.num(p['rate_usd_per_hour'], '.2f')} | {usd} | {p['source']} |"
        )
    lines += ["", "Components per system:", ""]
    lines += [f"- {s}: " + ", ".join(f"`{n}`" for n in r["components"]) for s, r in
              ((s, t["costs"][s]) for s in pr.EXAM_ORDER)]  # fmt: skip
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
