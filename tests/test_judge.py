import pytest

from edge_rag.artifacts import ArtifactError
from edge_rag.judge import judged, pair_row, score_table, uncached, union
from edge_rag.pod.common import Unit


def test_cache_split_sends_only_unstored_pairs():
    stored = {("q1", "a"): 1.0, ("q1", "c"): 0.5, ("q2", "b"): 2.0}
    assert uncached("q1", ["a", "b", "c", "d"], stored) == ["b", "d"]
    # The same unit under another question is another pair.
    assert uncached("q2", ["a", "b"], stored) == ["a"]
    assert uncached("q1", ["a", "c"], stored) == []


def test_judge_order_breaks_ties_by_pool_rank():
    table = {("q", "x"): 1.0, ("q", "y"): 3.0, ("q", "z"): 1.0, ("q", "w"): 1.0}
    assert judged("q", ["z", "x", "y", "w"], table) == ["y", "z", "x", "w"]
    assert judged("q", ["x", "w", "z", "y"], table) == ["y", "x", "w", "z"]


def test_assembly_from_mixed_cached_and_new_scores():
    stored = {("q", "a"): 0.2, ("q", "b"): 0.9}
    fresh = {("q", "c"): 0.5}
    pool = ["a", "b", "c"]
    table = score_table((("q", u) for u in pool), stored, fresh)
    assert table == {("q", "a"): 0.2, ("q", "b"): 0.9, ("q", "c"): 0.5}
    assert judged("q", pool, table) == ["b", "c", "a"]
    with pytest.raises(ArtifactError, match="no score"):
        score_table([("q", "d")], stored, fresh)
    with pytest.raises(ArtifactError, match="stored score and a pod score"):
        score_table([("q", "a")], stored, {("q", "a"): 0.2})


def test_one_score_per_pair_shared_by_both_systems():
    rrf3, rrf4 = ["a", "b", "c"], ["c", "d", "a"]
    both = union(rrf3, rrf4)
    assert both == ["a", "b", "c", "d"]
    stored = {("q", "a"): 1.0}
    fresh = {("q", u): float(i) for i, u in enumerate(uncached("q", both, stored))}
    pairs = [("q", u) for u in rrf3 + rrf4]
    table = score_table(pairs, stored, fresh)
    assert len(table) == len(both) == len(set(pairs))
    assert judged("q", rrf3, table) == ["a", "c", "b"]
    assert judged("q", rrf4, table) == ["d", "c", "a"]


def test_pairs_file_text_is_the_corpus_title_and_sentences():
    sentences = ["First sentence.", "Second one."]
    units = {"u1": Unit("u1", "A Title", " ".join(sentences))}
    row = pair_row("q", "Who?", ["u1"], units)
    assert row == {
        "qid": "q",
        "question": "Who?",
        "unit_ids": ["u1"],
        "texts": [f"A Title. {' '.join(sentences)}"],
    }
