"""Phase 05 states, verdict and entrant (spec C6): each bar, the context rows and `not run`."""

import pytest

from edge_rag.artifacts import ArtifactError
from edge_rag.phase_results import Phase05 as hr

SETS = ("hotpotqa-dev", "multihop-rag", "musique")
WIN = {"wins": 30, "losses": 5, "ties": 0, "p": 0.001}
LOSS = {"wins": 5, "losses": 30, "ties": 0, "p": 0.001}
TIE = {"wins": 10, "losses": 9, "ties": 0, "p": 1.0}
LIGHT, BEST = "p14", "g-a1"


def table(**states):
    """One set's table: every comparison a tie unless named by its STATE_NAMES index."""
    tests = []
    for i, (system, against) in enumerate(hr.COMPARISONS):
        target = {"best_light_class_so_far": LIGHT, "best_so_far": BEST}.get(against, against)
        tests.append({"system": system, "against": target, **states.get(f"c{i}", TIE)})
    return {
        "best_light_class_so_far": LIGHT,
        "best_so_far": BEST,
        "paired_full_support_at_2048": tests,
    }


def entrant_sets(**override):
    """mch wins against G-L and `rrf-prf` on every set; `override` replaces one set's table."""
    tables = {s: table(c0=WIN, c1=WIN) for s in SETS}
    tables.update(override)
    return tables


def test_state_names_follow_the_spec_comparisons():
    assert hr.STATE_NAMES == (
        "mch vs g-l",
        "mch vs rrf-prf",
        "mch vs rrf-1s",
        "mch vs best light-class so far",
        "mch vs best so far",
        "rrf-prf vs g-l",
        "rrf-prf vs best so far",
    )


def test_every_bar_met_makes_mch_the_entrant():
    out = hr.outcome(entrant_sets())
    assert out["verdict"] == "entrant"
    assert out["exam_entrant"] == "mch"


def test_one_win_against_g_l_does_not_advance():
    tables = {s: table() for s in SETS}
    tables["musique"] = table(c0=WIN, c1=WIN)
    out = hr.outcome(tables)
    assert out["verdict"] == "does not advance"
    assert out["exam_entrant"] == "none from this phase"


def test_a_loss_to_g_l_does_not_advance_even_with_two_wins():
    out = hr.outcome(entrant_sets(musique=table(c0=LOSS, c1=WIN)))
    assert out["verdict"] == "does not advance"


def test_no_win_against_rrf_prf_is_a_failed_bar():
    tables = {s: table(c0=WIN) for s in SETS}
    out = hr.outcome(tables)
    assert out["verdict"] == "advances, no entrant (no win against rrf-prf)"
    assert out["exam_entrant"] == "none from this phase"


def test_a_loss_to_rrf_prf_is_a_failed_bar():
    out = hr.outcome(entrant_sets(musique=table(c0=WIN, c1=LOSS)))
    assert out["verdict"] == "advances, no entrant (a loss to rrf-prf)"


def test_a_loss_to_the_best_light_class_system_is_a_failed_bar():
    out = hr.outcome(entrant_sets(musique=table(c0=WIN, c1=WIN, c3=LOSS)))
    assert out["verdict"] == ("advances, no entrant (a loss to the best light-class system so far)")


def test_context_comparisons_never_change_the_verdict():
    """Losses to `rrf-1s` and to the best system so far are context, not bars."""
    out = hr.outcome(entrant_sets(musique=table(c0=WIN, c1=WIN, c2=LOSS, c4=LOSS, c6=LOSS)))
    assert out["verdict"] == "entrant"
    assert out["states"]["mch vs rrf-1s"]["musique"] == "loss"


def test_rrf_prf_context_verdict_uses_the_advance_rule():
    tables = {s: table(c5=WIN) for s in SETS}
    assert hr.outcome(tables)["rrf-prf_context_verdict"] == "advances"
    tables["musique"] = table(c5=LOSS)
    assert hr.outcome(tables)["rrf-prf_context_verdict"] == "does not advance"


def test_hotpotqa_not_run_with_a_win_and_a_tie_does_not_advance():
    """The plan's case: HotpotQA not run, mch wins MuSiQue and ties MultiHop-RAG."""
    tables = {
        "hotpotqa-dev": None,
        "musique": table(c0=WIN, c1=WIN),
        "multihop-rag": table(c1=WIN),
    }
    out = hr.outcome(tables)
    assert out["states"]["mch vs g-l"] == {
        "hotpotqa-dev": "not run",
        "musique": "win",
        "multihop-rag": "tie",
    }
    assert out["verdict"] == "does not advance"
    assert out["exam_entrant"] == "none from this phase"


def test_a_set_not_run_needs_wins_on_both_other_sets():
    out = hr.outcome(entrant_sets(**{"hotpotqa-dev": None}))
    assert out["states"]["mch vs rrf-prf"]["hotpotqa-dev"] == "not run"
    assert out["verdict"] == "entrant"


def test_best_systems_come_from_earlier_rows_by_fs_at_2048():
    def row(fs):
        return {"metrics": {"full_support_at_budget": {"2048": fs}}}

    earlier = {"p10-a": row(400), "p14": row(761), "g-l": row(533), "g-a1": row(1101),
               "rrf3": row(594), "j-rrf4": row(831), "rrf4": row(665)}  # fmt: skip
    assert hr.best_of(earlier, hr.LIGHT) == ("p14", "g-a1")
    # Systems outside the light-class list never become the light bar, however strong.
    assert "j-rrf4" not in hr.LIGHT and "g-a1" not in hr.LIGHT
    # A tie keeps the system met first (the earlier phase).
    earlier["rrf4"] = row(761)
    assert hr.best_of(earlier, hr.LIGHT)[0] == "p14"


def test_light_class_list_is_the_spec_list():
    assert hr.LIGHT == ("p10-a", "p10-b", "p10-c", "p14", "g-l", "rrf3", "f3", "rrf4")


def test_bare_run_refuses_to_overwrite_stored_results(tmp_path, monkeypatch):
    stored = tmp_path / "results.json"
    stored.write_text("{}", "utf-8")
    monkeypatch.setattr(hr, "RESULTS_JSON", stored)
    with pytest.raises(ArtifactError, match="written once"):
        hr.run(SETS)
    assert stored.read_text("utf-8") == "{}"


def test_earlier_results_must_match_their_pinned_sha256(tmp_path, monkeypatch):
    changed = tmp_path / "results.json"
    changed.write_text('{"sets": []}', "utf-8")
    monkeypatch.setattr(hr, "EARLIER", (("Phase 03", changed, tmp_path),))
    with pytest.raises(ArtifactError, match="not the pinned Phase 03"):
        hr.earlier("musique")
