"""Phase 04 states, verdict and entrant (spec C8): each bar and the `not run` case."""

from edge_rag import judge_results as jr

SETS = ("hotpotqa-dev", "multihop-rag", "musique")
WIN = {"wins": 30, "losses": 5, "ties": 0, "p": 0.001}
LOSS = {"wins": 5, "losses": 30, "ties": 0, "p": 0.001}
TIE = {"wins": 10, "losses": 9, "ties": 0, "p": 1.0}


def table(**states):
    """One set's table: every comparison a tie unless named (`g_r`, `j_rrf3`, `in_bound`)."""
    by = {"g-r": states.get("g_r", TIE), jr.CONTROL: states.get("j_rrf3", TIE),
          "j-p10b": states.get("in_bound", TIE)}  # fmt: skip
    tests = []
    for system, against in jr.COMPARISONS:
        target = {"best_in_bound": "j-p10b", "best_so_far": "j-union"}.get(against, against)
        test = by.get(target, TIE) if system == jr.CANDIDATE else TIE
        tests.append({"system": system, "against": target, **test})
    return {
        "best_in_bound": "j-p10b",
        "best_so_far": "j-union",
        "paired_full_support_at_2048": tests,
    }


def verdict(*tables):
    return jr.outcome(dict(zip(SETS, tables, strict=True)))


def test_entrant_when_every_bar_is_met():
    out = verdict(table(g_r=WIN, j_rrf3=WIN), table(g_r=WIN), table(g_r=TIE))
    assert out["verdict"] == "entrant"
    assert out["exam_entrant"] == jr.CANDIDATE
    assert out["states"]["j-rrf4 vs g-r"] == dict(zip(SETS, ("win", "win", "tie"), strict=True))


def test_gate_needs_two_wins_and_no_loss_to_g_r():
    assert verdict(table(g_r=WIN), table(g_r=TIE), table(g_r=TIE))["verdict"] == "does not advance"
    out = verdict(table(g_r=WIN, j_rrf3=WIN), table(g_r=WIN), table(g_r=LOSS))
    assert out["verdict"] == "does not advance"


def test_each_entrant_bar_is_named():
    out = verdict(table(g_r=WIN), table(g_r=WIN), table())
    assert out["verdict"] == "advances, no entrant (no win against j-rrf3)"
    out = verdict(table(g_r=WIN, j_rrf3=WIN), table(g_r=WIN, j_rrf3=LOSS), table())
    assert out["verdict"] == "advances, no entrant (a loss to j-rrf3)"
    out = verdict(table(g_r=WIN, j_rrf3=WIN), table(g_r=WIN, in_bound=LOSS), table())
    assert out["verdict"] == (
        "advances, no entrant (a loss to the best in-bound rerank-class system so far)"
    )
    assert out["exam_entrant"] == "none from this phase"


def test_spec_not_run_case_does_not_advance():
    # HotpotQA not run, j-rrf4 wins MuSiQue against G-R and ties MultiHop-RAG.
    out = verdict(None, table(g_r=TIE), table(g_r=WIN))
    assert out["states"]["j-rrf4 vs g-r"] == {
        "hotpotqa-dev": "not run",
        "multihop-rag": "tie",
        "musique": "win",
    }
    assert out["verdict"] == "does not advance"
    assert out["j-rrf3_context_verdict"] == "does not advance"


def test_page_renders_with_every_set_not_run():
    out = jr.outcome(dict.fromkeys(SETS))
    page = jr.markdown({"outcome": out, "not_run": dict.fromkeys(SETS, "no judge"), "sets": []})
    assert "j-rrf4 verdict: **does not advance**" in page
    assert page.count("not run (no judge)") == 3


def test_judge_online_row_is_labelled_derived():
    """Plan D9: the per-100-pairs J-strong row is arithmetic on the timing sample, not measured."""
    laptop = {
        "offline_seconds": {"bm25_build": 1.0},
        "question_encoding": {"seconds_per_question": 0.1},
        "retrieval_seconds_per_question": {"dense": 0.1, "bm25": 0.1},
        "hardware": "laptop",
    }
    pod = {"cost_per_hr_usd": 0.74, "gpu": "RTX 4090", "seconds": {"load_model": 10.0},
           "timing_sample": {"seconds_per_100_pairs": 0.3}}
    g_l = {"offline_seconds": 1.0, "offline_usd": 0.0, "online_seconds_per_question": 0.02,
           "online_usd_per_question": 0.0}
    fused = {"online_seconds_per_question": {"rrf": 0.001}}
    cost = jr.system_cost(jr.CONTROL, "musique", laptop, fused, {}, g_l, "RTX 4090", pod)
    row = next(r for r in cost["components"] if r["component"] == "J-strong over 100 units")
    assert row["label"].startswith("seconds derived")
    assert cost["class_check"]["gpu_online_seconds_per_question"] == {"RTX 4090": 0.32}
