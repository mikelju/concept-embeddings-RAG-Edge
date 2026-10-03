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
