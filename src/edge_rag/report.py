"""Phase 08 report (spec C1-C4, D1-D3): `tournament.json`, the tables of `report.md`, the checker.

Nothing is recomputed: figures, verdict states, paired-test counts and cost cells are copied from
each phase's run file as its outcome code wrote them, with the SHA-256 of every input. The only
arithmetic is the money totals, labelled derived. Money rows are read from the master plan and
the phase plans: each row quotes its source, and every quote must appear verbatim in that file.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from edge_rag import config, results
from edge_rag import phase_results as pr
from edge_rag.artifacts import ArtifactError, write_bytes

OUT_DIR = config.REPO_ROOT / "docs" / "plans" / "fase-08-report"
EXPORT = OUT_DIR / "tournament.json"
REPORT = OUT_DIR / "report.md"
INPUTS = {
    "02": results.PHASE_DIR / "results.json",
    "03": pr.Phase03.RESULTS_JSON,
    "04": pr.Phase04.RESULTS_JSON,
    "05": pr.Phase05.RESULTS_JSON,
    "06": pr.Phase06.RESULTS_JSON,
    "07": pr.EXAM_JSON,
    "07-annex": pr.EXAM_JSON.parent / "annex.json",
}
PAGES = {
    "02": "docs/plans/fase-02-ghosts/results.md",
    "03": "docs/plans/fase-03-fusion/results.md",
    "04": "docs/plans/fase-04-judge/results.md",
    "05": "docs/plans/fase-05-hop/results.md",
    "06": "docs/plans/fase-06-rivals/results.md",
    "07": "docs/plans/fase-07-exam/results.md",
    "07-annex": "docs/plans/fase-07-exam/annex.md",
}
TERRAIN = ("02", "03", "04", "05", "06")
CANDIDATES = ("03", "04", "05", "06")
CONTEXT = "context (old project)"
SET_KEYS: tuple[str, ...] = (
    "best_old_by_fs_at_budget",
    "ghosts_not_measured",
    "literature_bar",
    "subset",
)
SET_KEYS += ("subset_metrics", "best_own_by_class", "best_so_far", "best_light_class_so_far")
BUDGET = str(config.BUDGET)
PLAN = "docs/plans/fase-0{}/plan.md"
NO_POD = ("docs/plans/fase-06-rivals/spec.md", "Phases 00, 01, 03 and 05 rented no pod")
# Money per phase (spec C6): the three figures where they exist, each quoted from its source.
MONEY: list[dict[str, Any]] = [
    {
        "phase": "00",
        "time_x_rate": None,
        "balance_delta": None,
        "invoice": None,
        "note": "no pod rented",
        "quotes": [NO_POD],
    },
    {
        "phase": "01",
        "time_x_rate": None,
        "balance_delta": None,
        "invoice": None,
        "note": "no pod rented",
        "quotes": [NO_POD],
    },
    {
        "phase": "02",
        "time_x_rate": "14.48",
        "balance_delta": "14.73",
        "invoice": "14.78",
        "note": "over the 11.5 USD phase cap (deviation 02.1)",
        "quotes": [
            (
                PLAN.format("2-ghosts"),
                "C7 not met: 14.48 USD time x rate, 14.73 USD balance delta, 14.78 USD invoice",
            )
        ],
    },
    {
        "phase": "03",
        "time_x_rate": None,
        "balance_delta": None,
        "invoice": "0",
        "note": "no pod; the author's billing reading shows no charge inside the phase (deviation"
        " 03.1)",
        "quotes": [
            NO_POD,
            (PLAN.format("3-fusion"), "Phase 03 spends 0 USD: no pod, no paid API"),
            (PLAN.format("3-fusion"), "shows no charge dated inside Phase 03"),
        ],
    },
    {
        "phase": "04",
        "time_x_rate": "0.32",
        "balance_delta": "0.33",
        "invoice": "0.33",
        "note": "",
        "quotes": [
            (
                PLAN.format("4-judge"),
                "Phase spend, three figures: time x rate 0.32 USD"
                " (derived, increment 9); post-pod balance drop 14.4439 - 14.1148 = 0.33 USD"
                " (measured, 08:24:28Z); invoice 0.33 USD (measured, author)",
            )
        ],
    },
    {
        "phase": "05",
        "time_x_rate": None,
        "balance_delta": "0",
        "invoice": None,
        "note": "no pod rented",
        "quotes": [NO_POD, (PLAN.format("5-hop"), "Phase spend 0 USD (measured)")],
    },
    {
        "phase": "06",
        "time_x_rate": "7.490",
        "balance_delta": "7.63",
        "invoice": None,
        "note": "invoice pending when the phase closed",
        "quotes": [
            (PLAN.format("6-rivals"), "1.098 + 0.314 + 5.069 = 7.490 USD (derived"),
            (PLAN.format("6-rivals"), "6.4846933702 = 7.6300570340 USD, about 7.63 USD"),
            (PLAN.format("6-rivals"), "is the figure that can settle the split; pending"),
        ],
    },
    {
        "phase": "07",
        "time_x_rate": "1.464",
        "balance_delta": "1.489",
        "invoice": "1.489",
        "note": "balance delta 1.4894290670 USD in the source",
        "quotes": [
            (
                PLAN.format("7-exam"),
                "time x rate 1.464 USD, balance delta 1.4894290670 USD,"
                " invoice 1.489 USD (GPU 1.464 + storage 0.025)",
            )
        ],
    },
]
FIRST_BALANCE = ("29.22", PLAN.format("2-ghosts"), "balance before the phase: 29.22 USD")
LAST_BALANCE = ("4.995", "docs/plans/0_plan_maestro.md", "closing `clientBalance` 4.995 USD")
MONEY_LABELS = {
    "time_x_rate": "derived (time x rate)",
    "balance_delta": "measured (balance delta)",
    "invoice": "measured (invoice)",
}
LABELS = ("measured", "derived", "exploratory", "interpretation", "projection")
BEGIN = re.compile(r"<!-- BEGIN GENERATED: ([a-z-]+) -->\n.*?<!-- END GENERATED -->", re.S)
POINTER = re.compile(r"\[src: ([^\]\s]+)\]")
NUMBER = re.compile(r"(?<![\w@./+-])\d(?:\d|,\d)*(?:\.\d+)?(?:e-?\d+)?(?![\w@/+-]|\.\d)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role(system: str) -> str:
    if system in results.OLD_SYSTEMS:
        return CONTEXT
    return "ghost" if system.lower().startswith("g-") else "own"


def terrain_phase(table: Mapping[str, Any]) -> dict[str, Any]:
    """One terrain phase as its outcome code wrote it: systems, paired tests, outcome, not run."""
    sets = {}
    for entry in table["sets"]:
        sets[entry["set"]] = {
            "n": entry["n"],
            "systems": {
                name: {"report_role": role(name), **row} for name, row in entry["systems"].items()
            },
            "paired_full_support_at_2048": entry["paired_full_support_at_2048"],
            **{key: entry[key] for key in SET_KEYS if key in entry},
        }
    return {
        "outcome": table.get("outcome"),
        "not_run": table.get("not_run", {}),
        "sets": sets,
    }


def money_check(root: Path) -> None:
    """Every money quote appears verbatim in its source and holds the figures of its row."""
    rows = [*MONEY, {"quotes": [FIRST_BALANCE[1:], LAST_BALANCE[1:]]}]
    for row in rows:
        text = " ".join(quote for _, quote in row["quotes"])
        for source, quote in row["quotes"]:
            if quote not in (root / source).read_text("utf-8"):
                raise ArtifactError(f"money quote not found in {source}: {quote}")
        for key in MONEY_LABELS:
            if row.get(key) not in (None, "0") and row[key] not in text:
                raise ArtifactError(f"Phase {row['phase']} {key} {row[key]} not in its quotes")


def money(root: Path) -> dict[str, Any]:
    money_check(root)
    totals: dict[str, Any] = {}
    for key in ("time_x_rate", "balance_delta"):
        total = sum((Decimal(r[key]) for r in MONEY if r[key] is not None), Decimal(0))
        totals[f"{key}_sum"] = str(total)
    first, last = Decimal(FIRST_BALANCE[0]), Decimal(LAST_BALANCE[0])
    return {
        "label": "per-phase figures as each source records them; totals derived (sums)",
        "phases": [
            {**row, "quotes": [{"source": s, "quote": q} for s, q in row["quotes"]]}
            for row in MONEY
        ],
        "totals": {
            **totals,
            "invoice_sum": None,
            "invoice_sum_note": "not available: the Phase 06 invoice was pending at its close",
            "balance_first_to_last": str(first - last),
            "balance_first_to_last_note": (
                f"derived: {FIRST_BALANCE[0]} USD before Phase 02 ({FIRST_BALANCE[1]}) minus"
                f" {LAST_BALANCE[0]} USD at the Phase 07 close ({LAST_BALANCE[1]})"
            ),
        },
    }


def build(inputs: Mapping[str, Path], root: Path = config.REPO_ROOT) -> dict[str, Any]:
    """The export: copied from the inputs, never retyped (spec C1, C2)."""
    loaded = {key: json.loads(path.read_text("utf-8")) for key, path in inputs.items()}
    phases: dict[str, Any] = {}
    for key in TERRAIN:
        phases[key] = {"page": PAGES[key], **terrain_phase(loaded[key])}
    exam, annex = loaded["07"], loaded["07-annex"]
    phases["07"] = {
        "page": PAGES["07"],
        "annex_page": PAGES["07-annex"],
        **{k: exam[k] for k in ("label", "n", "n_multi", "n_single", "expected", "dry_run")},
        "rows": exam["rows"],
        "verdict": exam["verdict"],
        "annex": {k: annex[k] for k in ("costs", "context_mcnemar", "empty_rankings")},
    }
    return {
        "generated_by": "uv run edge-rag report",
        "label": "figures, states and cost cells copied from the phase run files; money from"
        " the plans (see money.label); roles: context = the old project's systems (spec D3)",
        "inputs": {
            key: {"file": path.name, "phase": key[:2], "sha256": sha256(path)}
            for key, path in inputs.items()
        },
        "phases": phases,
        "money": money(root),
    }


def dump(export: Mapping[str, Any]) -> bytes:
    return (json.dumps(export, indent=2, sort_keys=True) + "\n").encode("utf-8")


# ---- tables -------------------------------------------------------------------------------


def cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.0f}" if abs(value) >= 1000 else f"{value:.3g}"
    if isinstance(value, dict):
        return "; ".join(f"{k}: {cell(v)}" for k, v in value.items()) or "-"
    return str(value).replace("|", "/")


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]


def fs_of(row: Mapping[str, Any]) -> Any:
    metrics = row.get("metrics")
    return None if metrics is None else metrics["full_support_at_budget"][BUDGET]


def t_verdicts(export: Mapping[str, Any]) -> list[str]:
    rows = []
    for key in CANDIDATES:
        outcome = export["phases"][key]["outcome"]
        for name, value in sorted(outcome.items()):
            if name not in ("states", "literature_bar", "label"):
                rows.append([key, name, value, outcome["label"]])
    return table(["Phase", "Field", "Value", "Label"], rows)


def t_states(export: Mapping[str, Any]) -> list[str]:
    rows = []
    for key in CANDIDATES:
        for comparison, by_set in sorted(export["phases"][key]["outcome"]["states"].items()):
            rows.append([key, comparison, *(by_set.get(s) for s in sorted(config.SETS))])
    return table(["Phase", "Comparison", *sorted(config.SETS)], rows)


def t_fs(export: Mapping[str, Any]) -> list[str]:
    rows = []
    for key in TERRAIN:
        sets = export["phases"][key]["sets"]
        names = sorted({n for entry in sets.values() for n in entry["systems"]})
        for name in names:
            values = []
            for set_name in sorted(config.SETS):
                entry = sets.get(set_name, {})
                row = entry.get("systems", {}).get(name)
                value = None if row is None else fs_of(row)
                subset = entry.get("subset_metrics", {}).get(name)
                if value is None and subset is not None:
                    value = f"{subset['full_support_at_budget'][BUDGET]} (of {subset['n']})"
                values.append(value)
            label = next(
                (
                    r["metrics_label"]
                    for e in sets.values()
                    for n, r in e["systems"].items()
                    if n == name and "metrics_label" in r
                ),
                "measured (subset metrics)",
            )
            rows.append([key, name, role(name), *values, label])
    return table(["Phase", "System", "Role", *sorted(config.SETS), "Label"], rows)


def t_paired(export: Mapping[str, Any]) -> list[str]:
    rows = []
    for key in TERRAIN:
        for set_name, entry in sorted(export["phases"][key]["sets"].items()):
            for test in entry["paired_full_support_at_2048"]:
                counts = [test.get(k) for k in ("wins", "losses", "ties", "p")]
                rows.append(
                    [
                        key,
                        set_name,
                        test["system"],
                        test["against"],
                        *counts,
                        test.get("state"),
                        test.get("on") or test.get("reason"),
                        test.get("label"),
                    ]
                )
    header = ["Phase", "Set", "System", "Against", "Wins", "Losses", "Ties", "Exact p", "State"]
    return table([*header, "On or reason", "Label"], rows)


def t_bar(export: Mapping[str, Any]) -> list[str]:
    rows = []
    for set_name, entry in sorted(export["phases"]["06"]["sets"].items()):
        for bar in entry["literature_bar"]:
            rows.append(
                [
                    set_name,
                    bar["class"],
                    bar["own"],
                    bar["own_fs"],
                    bar["strongest_ghost"],
                    bar["ghosts_fs"],
                    bar["wins"],
                    bar["losses"],
                    bar["ties"],
                    bar["p"],
                    bar["state"],
                    bar["on"],
                    bar["label"],
                ]
            )
    header = ["Set", "Class", "Own", "Own FS", "Strongest ghost", "Ghosts FS", "Wins", "Losses"]
    return table([*header, "Ties", "Exact p", "State", "On", "Label"], rows)


def t_exam(export: Mapping[str, Any]) -> list[str]:
    exam = export["phases"]["07"]
    rows = [
        [
            name,
            r["role"],
            r["fs"],
            r["fs_single"],
            r["fs_multi"],
            r["fs_within"],
            r["cost"]["label"],
        ]
        for name, r in sorted(exam["rows"].items())
    ]
    lines = table(["System", "Role", "FS@2,048", "Single", "Multi", "Within-paper", "Cost"], rows)
    verdicts = [
        [
            v["class"],
            v["own"],
            v["own_fs"],
            v["strongest_ghost"],
            v["ghosts_fs"],
            v["wins"],
            v["losses"],
            v["ties"],
            v["p"],
            v["state"],
            v.get("control", {}).get("state"),
            v["label"],
        ]
        for v in exam["verdict"]
    ]
    header = ["Class", "Own", "Own FS", "Strongest ghost", "Ghosts FS", "Wins", "Losses", "Ties"]
    return [
        f"In scope: {exam['n']} questions ({exam['n_single']} single-evidence,"
        f" {exam['n_multi']} multi-evidence); {exam['label']}.",
        "",
        *lines,
        "",
        *table([*header, "Exact p", "State", "Against j-rrf3", "Label"], verdicts),
    ]


def class_check(cost: Mapping[str, Any]) -> list[Any]:
    check = cost.get("class_check")
    if check is None:
        gpu = {cost.get("hardware"): cost.get("online_seconds_per_question")}
        return [None, gpu, None, cost.get("label")]
    gpu = check.get("gpu_online_seconds_per_question")
    if not isinstance(gpu, dict):
        gpu = {check.get("gpu"): gpu}
    inside = {k: v for k, v in check.items() if k.startswith("inside_")}
    return [check.get("laptop_online_seconds_per_question"), gpu, inside, check.get("label")]


def t_cost(export: Mapping[str, Any]) -> list[str]:
    rows = []
    for key in CANDIDATES:
        phase = export["phases"][key]
        subjects = sorted({c.split(" vs ")[0] for c in phase["outcome"]["states"]})
        for name in subjects:
            for set_name, entry in sorted(phase["sets"].items()):
                row = entry["systems"].get(name)
                if row is not None:
                    rows.append([key, name, set_name, *class_check(row.get("cost", {}))])
    header = ["Phase", "System", "Set", "Laptop online s/question", "GPU online s/question"]
    return table([*header, "Inside class", "Label"], rows)


def t_unrecorded(export: Mapping[str, Any]) -> list[str]:
    seen: dict[tuple[str, str, str], None] = {}
    for key in CANDIDATES:
        for entry in export["phases"][key]["sets"].values():
            for row in entry["systems"].values():
                for comp in row.get("cost", {}).get("components", []):
                    if comp.get("seconds") is None:
                        seen[(key, comp["component"], comp["label"])] = None
    for name, cost in sorted(export["phases"]["07"]["annex"]["costs"].items()):
        for item in cost["not_recorded"]:
            seen[("07", f"{name}: {item['name']}", f"{item['kind']} ({item['why']})")] = None
    return table(["Phase", "Cost cell", "Label as written"], [list(k) for k in seen])


def t_exam_cost(export: Mapping[str, Any]) -> list[str]:
    rows = [
        [
            name,
            c["offline_seconds"],
            c["online_seconds_per_question"],
            c["unsplit_seconds"],
            c["usd"],
            ", ".join(c["hardware"]),
            c["label"],
        ]
        for name, c in sorted(export["phases"]["07"]["annex"]["costs"].items())
    ]
    header = ["System", "Offline s", "Online s/question", "Unsplit s", "USD", "Hardware"]
    return table([*header, "Label"], rows)


def t_money(export: Mapping[str, Any]) -> list[str]:
    m = export["money"]
    rows = [
        [
            r["phase"],
            r["time_x_rate"],
            r["balance_delta"],
            r["invoice"],
            r["note"],
            "; ".join(sorted({q["source"] for q in r["quotes"]})),
        ]
        for r in m["phases"]
    ]
    t = m["totals"]
    rows.append(
        [
            "total",
            t["time_x_rate_sum"],
            t["balance_delta_sum"],
            t["invoice_sum"],
            f"derived sums; {t['invoice_sum_note']}; balance first to last"
            f" {t['balance_first_to_last']} ({t['balance_first_to_last_note']})",
            "-",
        ]
    )
    header = [
        f"Time x rate USD, {MONEY_LABELS['time_x_rate']}",
        f"Balance delta USD, {MONEY_LABELS['balance_delta']}",
        f"Invoice USD, {MONEY_LABELS['invoice']}",
    ]
    return table(["Phase", *header, "Note", "Source"], rows)


TABLES = {
    "verdicts": t_verdicts,
    "states": t_states,
    "fs": t_fs,
    "paired": t_paired,
    "cost": t_cost,
    "unrecorded": t_unrecorded,
    "bar": t_bar,
    "exam": t_exam,
    "exam-cost": t_exam_cost,
    "money": t_money,
}


def render(text: str, export: Mapping[str, Any]) -> str:
    """Rewrite every generated block of the report from the export; prose is left untouched."""

    def block(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in TABLES:
            raise ArtifactError(f"unknown generated block: {name}")
        body = "\n".join(TABLES[name](export))
        return f"<!-- BEGIN GENERATED: {name} -->\n{body}\n<!-- END GENERATED -->"

    return BEGIN.sub(block, text)


# ---- checker ------------------------------------------------------------------------------


def decimal(token: str) -> Decimal | None:
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def export_values(export: Any) -> list[Decimal]:
    """Every number in the export: numeric leaves and numbers written inside string leaves."""
    found: list[Decimal] = []
    stack = [export]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, bool) or item is None:
            continue
        elif isinstance(item, (int, float)):
            found.append(Decimal(repr(item)))
        elif isinstance(item, str):
            found.extend(d for t in NUMBER.findall(item) if (d := decimal(t)) is not None)
    return found


def matches(token: str, values: set[Decimal]) -> bool:
    """A prose figure matches an export value equal to it at the precision it is written with."""
    figure = decimal(token)
    if figure is None:
        return False
    exponent = figure.as_tuple().exponent
    quantum = Decimal(1).scaleb(exponent if isinstance(exponent, int) else 0)
    if "e" in token:
        digits = len(token.split("e")[0].replace(".", "").replace(",", ""))
        return any(v != 0 and f"{v:.{digits - 1}e}" == f"{figure:.{digits - 1}e}" for v in values)
    return any(v.quantize(quantum) == figure for v in values if abs(v) < 10**15)


def prose(text: str) -> list[tuple[int, str]]:
    """Hand-written lines only: generated blocks, HTML comments and inline code removed."""
    text = BEGIN.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    text = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    lines = []
    for number, line in enumerate(text.splitlines(), 1):
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        lines.append((number, line))
    return lines


def exempt(token: str) -> bool:
    """Single digits and zero-padded phase numbers are words, not figures."""
    return re.fullmatch(r"\d|0\d", token) is not None


def check(text: str, export: Mapping[str, Any], root: Path = config.REPO_ROOT) -> list[str]:
    """Untraced or unlabelled figures in the prose (spec C3, C4); empty means it passes."""
    values = set(export_values(export))
    problems = []
    for number, line in prose(text):
        pointers = POINTER.findall(line)
        pages = []
        for pointer in pointers:
            path = root / pointer
            if pointer.startswith("data/") or not path.is_file():
                problems.append(
                    f"line {number}: pointer to a missing or uncommitted page {pointer}"
                )
            else:
                pages.append(path.read_text("utf-8"))
        bare = POINTER.sub("", line)
        figures = [t for t in NUMBER.findall(bare) if not exempt(t)]
        for token in figures:
            if not matches(token, values) and not any(token in page for page in pages):
                problems.append(f"line {number}: untraced figure {token}")
        if figures and not any(label in bare for label in LABELS):
            problems.append(f"line {number}: figures without a label")
    return problems


def run(write: bool, say: Any) -> int:
    """`edge-rag report`: write the export and the tables; `--check`: verify both, write nothing."""
    text = REPORT.read_text("utf-8")
    if write:
        body = dump(build(INPUTS))
        write_bytes(EXPORT, body)
        text = render(text, json.loads(body))
        write_bytes(REPORT, text.encode("utf-8"))
        say(f"[OK] wrote {EXPORT.name}, sha256 {hashlib.sha256(body).hexdigest()}")
    export = json.loads(EXPORT.read_text("utf-8"))
    if render(text, export) != text:
        say(f"[ERROR] the tables of {REPORT.name} differ from {EXPORT.name}; rerun edge-rag report")
        return 1
    problems = check(text, export)
    for problem in problems:
        say(f"[ERROR] {problem}")
    if problems:
        return 1
    say(f"[OK] {REPORT.name}: tables equal to the export, every prose figure traced")
    return 0
