import pytest

from edge_rag import fusion_results as fr


def test_state_needs_the_direction_and_p_below_alpha():
    assert fr.state(None) == "not run"
    assert fr.state({"wins": 30, "losses": 10, "p": 0.001}) == "win"
    assert fr.state({"wins": 10, "losses": 30, "p": 0.001}) == "loss"
    assert fr.state({"wins": 30, "losses": 10, "p": 0.05}) == "tie"
    assert fr.state({"wins": 5, "losses": 5, "p": 1.0}) == "tie"


def test_not_run_hotpotqa_with_one_win_does_not_advance():
    # HotpotQA not run, F3 wins MuSiQue and ties MultiHop-RAG (spec C5).
    g_l = ["not run", "win", "tie"]
    assert fr.verdict(g_l, ["not run", "win", "win"], ["not run", "tie", "tie"]) == (
        "does not advance"
    )
    assert fr.verdict(["not run", "win", "win"], ["not run", "win", "tie"], ["tie"] * 3) == (
        "entrant"
    )


def test_each_bar_of_the_selection_rule():
    win3 = ["win", "win", "win"]
    assert fr.verdict(["win", "win", "loss"], win3, win3) == "does not advance"
    assert fr.verdict(["win", "win", "tie"], ["tie", "win", "tie"], ["tie"] * 3) == "entrant"
    assert fr.verdict(win3, ["tie"] * 3, ["tie"] * 3) == (
        "advances, no entrant (no win against RRF3)"
    )
    assert fr.verdict(win3, ["win", "loss", "tie"], ["tie"] * 3) == (
        "advances, no entrant (a loss to RRF3)"
    )
    assert fr.verdict(win3, ["win", "tie", "tie"], ["loss", "tie", "tie"]) == (
        "advances, no entrant (a loss to the best light-class system so far)"
    )


def _cost(system, gpu_online):
    comp = {
        "hardware": "laptop CPU",
        "offline_seconds": {"bm25_build": 6.0},
        "question_encoding": {"seconds_per_question": 0.02, "questions": 200},
        "retrieval_seconds_per_question": {"dense": 0.004, "bm25": 0.001},
    }
    fused = {
        "offline_seconds": {"boilerplate_flags": 0.5},
        "online_seconds_per_question": {"rrf": 0.0001, "diversity": 0.01},
    }
    g_l = {
        "offline_seconds": 1000.0,
        "offline_usd": 0.2,
        "online_seconds_per_question": gpu_online,
        "online_usd_per_question": 0.00001,
    }
    return fr.candidate_cost(system, comp, fused, g_l, "RTX 4090", "musique")


def test_class_check_is_per_hardware_and_f3_adds_its_components():
    rrf3, f3 = _cost("rrf3", 0.025), _cost("f3", 0.025)
    laptop = f3["totals_per_hardware"]["online laptop CPU"]["seconds"]
    assert laptop == pytest.approx(0.02 + 0.004 + 0.001 + 0.0001 + 0.01)
    assert rrf3["totals_per_hardware"]["online laptop CPU"]["seconds"] == pytest.approx(
        laptop - 0.01
    )
    assert f3["totals_per_hardware"]["offline laptop CPU"]["seconds"] == pytest.approx(6.5)
    assert f3["class_check"]["inside_light_class"]
    outside = _cost("f3", 0.358)["class_check"]
    assert outside["laptop_inside"] and not outside["gpu_inside"]
    assert not outside["inside_light_class"]


def test_outcome_reads_states_and_marks_missing_sets_not_run():
    def table(f3_vs_g_l_wins):
        tests = [
            {"system": s, "against": a, "wins": w, "losses": 0, "p": 0.001 if w else 1.0}
            for s, a, w in [
                ("f3", "g-l", f3_vs_g_l_wins),
                ("rrf3", "g-l", 50),
                ("f3", "rrf3", 0),
                ("f3", "p14", 0),
                ("rrf3", "p14", 0),
                ("f3", "j-union", 0),
                ("rrf3", "j-union", 0),
            ]
        ]
        return {
            "best_light_class_so_far": "p14",
            "best_so_far": "j-union",
            "paired_full_support_at_2048": tests,
        }

    out = fr.outcome({"hotpotqa-dev": None, "musique": table(50), "multihop-rag": table(0)})
    assert out["states"]["f3 vs g-l"] == {
        "hotpotqa-dev": "not run",
        "musique": "win",
        "multihop-rag": "tie",
    }
    assert out["f3_verdict"] == "does not advance"
    assert out["rrf3_context_verdict"] == "advances"
    assert out["exam_entrant"] == "none from this phase"


def test_markdown_renders_the_same_from_results_json_as_written():
    # Spec C4: results.md regenerates byte-equal from results.json, whose keys are sorted.
    import json

    metrics = {
        "full_support_at_budget": {"1024": 1, "2048": 2, "4096": 3},
        "full_support_at_k": {"2": 1, "5": 2, "20": 3},
        "recall_at_k": {k: 0.5 for k in ("2", "5", "10", "20", "100")},
        "gold_share_at_5": 0.25,
        "ndcg_at_10": 0.5,
    }
    ref = {"label": "measured", "offline_seconds": 1.0, "offline_usd": 0.1,
           "online_seconds_per_question": 0.01, "online_usd_per_question": 0.0}  # fmt: skip
    systems = {
        "rrf3": {"metrics": metrics, "cost": _cost("rrf3", 0.025)},
        "f3": {"metrics": metrics, "cost": _cost("f3", 0.025)},
        "p14": {
            "metrics": metrics,
            "cost": {**ref, "handover": fr.handover_cost("musique", "p14")},
        },
        "g-l": {"metrics": metrics, "cost": ref},
        "j-union": {"metrics": metrics, "cost": ref},
    }
    tests = [
        {"system": s, "against": a, "wins": 1, "losses": 0, "ties": 0, "p": 1.0, "state": "tie"}
        for s, a in [("rrf3", "g-l"), ("f3", "g-l"), ("f3", "rrf3"), ("f3", "p14")]
    ]
    rules = {"source cap": 3, "boilerplate": 1, "near-duplicate": 2}
    entry = {
        "set": "musique",
        "n": 4,
        "best_light_class_so_far": "p14",
        "best_so_far": "j-union",
        "systems": systems,
        "paired_full_support_at_2048": tests,
        "boilerplate_units": 6,
        "demotions_per_rule": rules,
        "demotions_per_rule_scope": "the union",
        "demotions_per_rule_in_rrf3_top_100": rules,
        "notes": ["a note"],
        "exploratory_gold_diagnostics": {
            "questions_with_a_gold_unit_demoted": {
                r: {"in_context_at_budget": 0, "in_top_100": 1} for r in rules
            },
            "gold_units_flagged_boilerplate": 0,
            "gold_units": 9,
        },
        "peak_rss_mb": {"components": 1.0, "fuse": 2.0},
    }
    states = {"musique": "tie", "hotpotqa-dev": "not run"}
    table = {
        "outcome": {
            "f3_verdict": "does not advance",
            "exam_entrant": "none from this phase",
            "rrf3_context_verdict": "does not advance",
            "states": {name: dict(states) for name in reversed(fr.STATE_NAMES)},
        },
        "not_run": {"hotpotqa-dev": "no Phase 03 fuse manifest"},
        "sets": [entry],
    }
    page = fr.markdown(table)
    assert fr.markdown(json.loads(json.dumps(table, sort_keys=True))) == page
    assert page.index("| f3 | ") < page.index("| g-l | ") < page.index("| j-union | ")
    assert page.index("| f3 vs g-l |") < page.index("| rrf3 vs g-l |")
    assert "invoiced 0.356 USD (measured (invoice)" in page
