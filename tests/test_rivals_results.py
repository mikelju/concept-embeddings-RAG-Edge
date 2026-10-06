"""Phase 06 results (spec C5, C6): states, the literature bar, the bests and the not-run rows,
on hand-built examples; the generated table, when present, for the 1,000-qid pairing."""

import json

import pytest

from edge_rag import config
from edge_rag import phase_results as pr
from edge_rag.artifacts import ArtifactError
from edge_rag.phase_results import Phase06 as rr

SETS = ("hotpotqa-dev", "multihop-rag", "musique")


def hits(n_ones: int, n: int = 40) -> list[int]:
    return [1] * n_ones + [0] * (n - n_ones)


def test_comparisons_follow_the_spec():
    assert rr.COMPARISONS == (
        ("g-r2", "best_so_far"), ("g-r2", "best_own"), ("g-r2", "g-r"),
        ("g-a1", "best_so_far"), ("g-a1", "best_own"), ("g-a1", "g-a2"),
        ("g-a2", "best_so_far"), ("g-a2", "best_own"), ("g-a2", "g-a1"),
    )  # fmt: skip
    assert rr.CLASSES == {"L": ("g-l",), "R": ("g-r", "g-r2"), "A": ("g-a1", "g-a2")}


def test_bar_takes_the_strongest_measured_ghost_and_states_a_win():
    own = hits(30)
    bar = pr.literature_bar("R", {"g-r": hits(10), "g-r2": hits(12)}, "j-rrf4", own, pr.FULL_SET)
    assert bar["strongest_ghost"] == "g-r2"
    assert (bar["wins"], bar["losses"], bar["ties"]) == (18, 0, 22)
    assert bar["state"] == "win"
    assert bar["not_measured"] == []


def test_bar_states_a_loss_and_a_tie():
    loss = pr.literature_bar("A", {"g-a1": hits(30), "g-a2": None}, "x", hits(10), pr.FULL_SET)
    assert loss["strongest_ghost"] == "g-a1"
    assert loss["state"] == "loss"
    assert loss["not_measured"] == ["g-a2"]
    tie = pr.literature_bar("L", {"g-l": hits(10)}, "x", hits(11), pr.FULL_SET)
    assert tie["state"] == "tie"


def test_bar_without_a_measured_ghost_is_not_run():
    bar = pr.literature_bar("A", {"g-a1": None, "g-a2": None}, "x", hits(10), pr.FULL_SET)
    assert bar["strongest_ghost"] is None
    assert bar["state"] == pr.NOT_RUN


def test_only_a_won_bar_is_a_literature_claim():
    bars = {"musique": {"L": "win", "R": "tie", "A": "loss"}, "hotpotqa-dev": {"A": "not run"}}
    assert pr.literature_claims(bars) == ["class L on musique"]


def test_bests_by_fs_and_by_class_or_cheaper():
    def row(fs):
        return {"metrics": {"full_support_at_budget": {"2048": fs}}}

    earlier = {"g-l": row(5), "g-a1": row(50), "p10-a": row(8), "rrf4": row(9), "j-rrf4": row(20)}
    bests = rr.bests(earlier)
    assert bests["best_so_far"] == "g-a1"
    assert bests["best_own"] == "j-rrf4"
    assert bests["best_own_by_class"] == {"L": "rrf4", "R": "j-rrf4", "A": "j-rrf4"}
    with pytest.raises(ArtifactError, match="without a cost class"):
        rr.bests({**earlier, "new-system": row(1)})


def test_outcome_marks_a_set_not_run_and_reads_states():
    table = {
        f"paired_full_support_at_{config.BUDGET}": [{"name": "g-r2 vs best own", "state": "loss"}],
        "literature_bar": [{"class": "L", "state": "win"}],
    }
    out = rr.outcome({"musique": table, "hotpotqa-dev": None})
    assert out["states"]["g-r2 vs best own"] == {"musique": "loss", "hotpotqa-dev": pr.NOT_RUN}
    assert out["states"]["g-a2 vs g-a1"]["musique"] == pr.NOT_RUN
    assert out["literature_bar"]["musique"] == {"L": "win", "R": pr.NOT_RUN, "A": pr.NOT_RUN}
    assert out["beats_the_literature"] == ["class L on musique"]


def test_g_a2_not_run_carries_the_cost_gate_reason():
    reason = rr.NOT_RUN_REASONS[("multihop-rag", "g-a2")]
    assert "7.10 USD" in reason and "4.5 USD" in reason and "06.3" in reason


def test_a_rival_on_other_qids_is_refused(tmp_path):
    name = "g-r2.jsonl.gz"
    (tmp_path / "manifest.json").write_text(json.dumps({name: "abc"}), "utf-8")
    manifest = {"outputs_sha256": {name: "abc"}, "subset": {"sha256": "0" * 64}}
    (tmp_path / "g-r2.manifest.json").write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ArtifactError, match="not the preregistered qids"):
        rr.rivals_of("hotpotqa-dev", tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({name: "other"}), "utf-8")
    with pytest.raises(ArtifactError, match="manifests name other rankings"):
        rr.rivals_of("hotpotqa-dev", tmp_path)


def test_bare_run_refuses_to_overwrite_stored_results(tmp_path, monkeypatch):
    stored = tmp_path / "results.json"
    stored.write_text("{}", "utf-8")
    monkeypatch.setattr(rr, "RESULTS_JSON", stored)
    with pytest.raises(ArtifactError, match="written once"):
        rr.run(SETS)
    assert stored.read_text("utf-8") == "{}"


@pytest.mark.skipif(not rr.RESULTS_JSON.exists(), reason="data/phase06/results.json not present")
def test_stored_hotpotqa_rival_tests_pair_the_same_1000_qids():
    table = json.loads(rr.RESULTS_JSON.read_text("utf-8"))
    entry = next(e for e in table["sets"] if e["set"] == "hotpotqa-dev")
    assert entry["subset"]["questions"] == 1000
    tested = [
        t for t in entry[f"paired_full_support_at_{config.BUDGET}"] if t["state"] != pr.NOT_RUN
    ]
    assert {t["system"] for t in tested} == {"g-r2", "g-a1"}
    assert all(t["on"] == pr.SUBSET and t["n"] == 1000 for t in tested)
    bars = {b["class"]: b for b in entry["literature_bar"]}
    assert bars["R"]["on"] == bars["A"]["on"] == pr.SUBSET
