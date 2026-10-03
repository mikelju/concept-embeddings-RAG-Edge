import pytest

from edge_rag import metrics, scoring
from edge_rag.artifacts import ArtifactError
from edge_rag.corpus import Question


def _question(qid, *gold):
    return Question(qid=qid, question="?", gold_unit_ids=gold, split="dev")


def test_full_support_at_k_units():
    ranked = ["x", "g1", "y", "g2", "z"]
    assert metrics.full_support_at_k(ranked, ["g1", "g2"], 2) == 0
    assert metrics.full_support_at_k(ranked, ["g1", "g2"], 4) == 1
    assert metrics.full_support_at_k(ranked, ["g1"], 2) == 1


def test_gold_share_is_the_per_question_fraction_of_gold_in_the_top_k():
    ranked = ["g1", "x", "y", "z", "w", "g2"]
    assert metrics.gold_share_at_k(ranked, ["g1", "g2"], 5) == 0.5
    assert metrics.gold_share_at_k(ranked, ["g1", "g2", "g3", "g4"], 5) == 0.25
    assert metrics.gold_share_at_k(ranked, ["g1", "g2"], 6) == 1.0


def test_score_counts_budgets_units_and_share():
    counts = {"a": 1500, "b": 600, "c": 500, "g": 100}
    questions = [_question("q1", "a", "c"), _question("q2", "b", "g")]
    rankings = {"q1": ["a", "b", "c", "g"], "q2": ["a", "b", "g", "c"]}
    summary, record = scoring.score(questions, rankings, counts)
    assert summary["n"] == 2
    # 1,024 tokens: q1 holds b and g, q2 holds b and g; 2,048: q1 holds a and c, q2 a and g.
    assert summary["full_support_at_budget"] == {"1024": 1, "2048": 1, "4096": 2}
    assert record == [1, 0]
    assert summary["full_support_at_k"] == {"2": 0, "5": 2, "20": 2}
    assert summary["gold_share_at_5"] == 1.0


def test_missing_or_extra_qids_are_an_error():
    counts = {"a": 1}
    questions = [_question("q1", "a"), _question("q2", "a")]
    with pytest.raises(ArtifactError, match="1 missing"):
        scoring.score(questions, {"q1": ["a"]}, counts)
    with pytest.raises(ArtifactError, match="1 extra"):
        scoring.score(questions, {"q1": ["a"], "q2": ["a"], "q3": ["a"]}, counts)


def test_ranking_file_round_trip_and_write_once(tmp_path):
    path = tmp_path / "set" / "system.jsonl.gz"
    rows = [("q1", ["a", "b"]), ("q2", ["c"])]
    digest = scoring.write_rankings(path, rows)
    assert scoring.read_rankings(path) == {"q1": ["a", "b"], "q2": ["c"]}
    assert scoring.write_rankings(path, rows) == digest
    assert '"system.jsonl.gz": "' + digest in (path.parent / "manifest.json").read_text()
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        scoring.write_rankings(path, [("q1", ["b", "a"]), ("q2", ["c"])])
    assert scoring.read_rankings(path) == {"q1": ["a", "b"], "q2": ["c"]}
