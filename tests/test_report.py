import hashlib
import json
from pathlib import Path

import pytest

from edge_rag import report

STATES = {"x vs g-l": {"hotpotqa-dev": "win", "multihop-rag": "tie", "musique": "loss"}}


def _system(fs: int) -> dict:
    return {
        "metrics": {"full_support_at_budget": {"2048": fs}, "n": 10},
        "metrics_label": "measured",
        "cost": {"components": [{"component": "Dense", "label": "not recorded", "seconds": None}]},
    }


def _terrain(phase: str) -> dict:
    sets = []
    for offset, name in enumerate(("hotpotqa-dev", "multihop-rag", "musique")):
        paired = {
            "system": "x",
            "against": "g-l",
            "wins": 7,
            "losses": 2,
            "ties": 1 + offset,
            "p": 0.0898,
            "state": STATES["x vs g-l"][name],
            "label": "measured",
        }
        sets.append(
            {
                "set": name,
                "n": 10,
                "paired_full_support_at_2048": [paired],
                "systems": {
                    "x": _system(5291 + offset),
                    "g-l": _system(5000),
                    "p14": _system(4000),
                },
            }
        )
    table = {"sets": sets}
    if phase != "02":
        table["outcome"] = {
            "label": "written by code",
            "verdict": "does not advance",
            "states": STATES,
        }
    if phase == "06":
        bar = {
            "class": "L",
            "own": "x",
            "own_fs": 5291,
            "strongest_ghost": "g-l",
            "ghosts_fs": {"g-l": 5000},
            "wins": 7,
            "losses": 2,
            "ties": 1,
            "p": 0.0898,
            "state": "win",
            "on": "full set",
            "label": "measured",
        }
        for entry in sets:
            entry["literature_bar"] = [bar]
    return table


def _exam() -> tuple[dict, dict]:
    row = {
        "cost": {"label": "not recorded"},
        "fs": 928,
        "fs_multi": 96,
        "fs_single": 832,
        "fs_within": None,
        "role": "entrant",
        "upper": None,
    }
    verdict = {
        "class": "L",
        "own": "rrf4",
        "own_fs": 928,
        "strongest_ghost": "G-L",
        "ghosts_fs": {"G-L": 1058},
        "wins": 55,
        "losses": 185,
        "ties": 1097,
        "p": 1.2441220987657834e-17,
        "state": "loss",
        "label": "measured",
    }
    exam = {
        "label": "measured",
        "n": 1337,
        "n_multi": 231,
        "n_single": 1106,
        "expected": 1451,
        "dry_run": None,
        "rows": {"rrf4": row},
        "verdict": [verdict],
    }
    cost = {
        "offline_seconds": 1.0,
        "online_seconds_per_question": 0.04,
        "unsplit_seconds": 0,
        "usd": 0.0,
        "hardware": ["laptop"],
        "label": "derived",
        "not_recorded": [{"kind": "not recorded", "name": "fusion", "why": "untimed"}],
    }
    annex = {"costs": {"rrf4": cost}, "context_mcnemar": [], "empty_rankings": {}}
    return exam, annex


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    exam, annex = _exam()
    bodies = {key: _terrain(key) for key in report.TERRAIN}
    bodies |= {"07": exam, "07-annex": annex}
    paths = {}
    for key, body in bodies.items():
        paths[key] = tmp_path / key / "results.json"
        paths[key].parent.mkdir()
        paths[key].write_text(json.dumps(body), "utf-8")
    return paths


def test_rerun_is_byte_equal_and_records_every_input_digest(inputs):
    first, second = report.dump(report.build(inputs)), report.dump(report.build(inputs))
    assert first == second
    recorded = json.loads(first)["inputs"]
    for key, path in inputs.items():
        assert recorded[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_states_counts_and_cost_labels_are_copied_from_the_run_files(inputs):
    export = report.build(inputs)
    for key in report.CANDIDATES:
        written = json.loads(inputs[key].read_text("utf-8"))
        assert export["phases"][key]["outcome"] == written["outcome"]
        for entry in written["sets"]:
            copied = export["phases"][key]["sets"][entry["set"]]
            assert copied["paired_full_support_at_2048"] == entry["paired_full_support_at_2048"]
    assert export["phases"]["07"]["verdict"][0]["state"] == "loss"
    assert export["phases"]["07"]["rows"]["rrf4"]["cost"]["label"] == "not recorded"
    assert export["phases"]["02"]["sets"]["musique"]["systems"]["p14"]["report_role"] == (
        report.CONTEXT
    )
    page = "\n".join(report.t_unrecorded(export))
    assert "not recorded" in page and "fusion" in page


def _report(prose: str) -> str:
    return (
        f"# Report\n\n{prose}\n\n<!-- BEGIN GENERATED: states -->\nstale\n<!-- END GENERATED -->\n"
    )


def test_render_fills_the_tables_and_keeps_the_prose(inputs):
    export = json.loads(report.dump(report.build(inputs)))
    text = report.render(_report("Class L loses (measured)."), export)
    assert "Class L loses (measured)." in text and "| 03 | x vs g-l | win | tie | loss |" in text
    assert report.render(text, export) == text


def test_checker_passes_traced_figures_and_fails_a_planted_one(inputs):
    export = json.loads(report.dump(report.build(inputs)))
    good = "rrf4 reaches 928 of 1,337 against 1,058, exact p 1.24e-17 (measured)."
    assert report.check(report.render(_report(good), export), export) == []
    planted = report.check(report.render(_report(good.replace("928", "929")), export), export)
    assert planted == ["line 3: untraced figure 929"]
    unlabelled = report.check(_report("rrf4 reaches 928."), export)
    assert unlabelled == ["line 3: figures without a label"]


def test_a_pointer_traces_a_figure_only_if_the_page_holds_it(inputs):
    export = json.loads(report.dump(report.build(inputs)))
    page = "docs/plans/fase-07-exam/results.md"
    held = f"The hop is empty on some questions; 1,106 single-evidence (measured) [src: {page}]."
    assert report.check(_report(held), export) == []
    absent = report.check(_report(f"A count of 98,765 (measured) [src: {page}]."), export)
    assert absent == ["line 3: untraced figure 98,765"]
    assert report.check(_report("A count of 1,106 (measured) [src: data/x.md]."), export)
