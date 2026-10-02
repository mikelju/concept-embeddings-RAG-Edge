import math

import pytest
from scipy.stats import binomtest

from edge_rag import metrics
from edge_rag.retrieval.fusion import fuse_lists, min_max_normalize


def test_exact_mcnemar_small_values():
    assert metrics.exact_two_sided_p(0, 0) == 1.0
    assert metrics.exact_two_sided_p(1, 1) == 1.0
    assert metrics.exact_two_sided_p(0, 5) == 2 / 32
    assert metrics.exact_two_sided_p(5, 0) == 2 / 32


def test_exact_mcnemar_past_1024_discordant_does_not_overflow():
    wins, losses = 600, 700
    n = wins + losses
    tail = sum(math.comb(n, i) for i in range(wins + 1))
    with pytest.raises(OverflowError):
        _ = 2.0 * tail / 2**n  # the pre-9a9ad3f form
    p = metrics.exact_two_sided_p(wins, losses)
    assert p == pytest.approx(binomtest(wins, n, 0.5).pvalue, rel=1e-9)


def test_paired_counts():
    assert metrics.paired([1, 0, 1, 0], [1, 1, 0, 1]) == {
        "wins": 2,
        "losses": 1,
        "ties": 1,
        "p": 1.0,
    }


def test_fill_context_skips_a_unit_that_does_not_fit_and_carries_on():
    counts = {"a": 1500, "b": 600, "c": 500, "d": 48}
    assert metrics.fill_context(["a", "b", "c", "d"], counts, 2048) == ["a", "c", "d"]
    assert metrics.full_support(["a", "c"], ["a", "c"]) == 1
    assert metrics.full_support(["a", "c"], ["a", "b"]) == 0


def test_recall_and_ndcg():
    ranked = ["x", "g1", "y", "g2"]
    assert metrics.recall_at_k(ranked, ["g1", "g2"], 2) == 0.5
    assert metrics.recall_at_k(ranked, ["g1", "g2"], 100) == 1.0
    ideal = 1 + 1 / math.log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / ideal
    assert metrics.ndcg_at_k(ranked, ["g1", "g2"], 10) == pytest.approx(expected)
    assert metrics.ndcg_at_k(["g1", "g2"], ["g1", "g2"], 10) == 1.0
    assert metrics.ndcg_at_k(ranked, [], 10) == 0.0


def test_min_max_absent_floor_and_constant_list():
    assert min_max_normalize([("a", 3.0), ("b", 1.0)], ["a", "b", "c"]) == {
        "a": 1.0,
        "b": 0.0,
        "c": 0.0,
    }
    assert min_max_normalize([("a", 2.0), ("b", 2.0)], ["a", "b"]) == {"a": 0.0, "b": 0.0}


def test_fusion_weights_and_ties_by_unit_id():
    dense = [("b", 0.9), ("a", 0.5)]
    bm25 = [("a", 10.0), ("b", 2.0)]
    # Both units score 0.5: the tie goes to the smaller unit id.
    assert fuse_lists([dense, bm25], (0.5, 0.5), top_k=10) == [("a", 0.5), ("b", 0.5)]
    with pytest.raises(ValueError):
        fuse_lists([dense, bm25], (0.6, 0.6), top_k=10)
    with pytest.raises(ValueError):
        fuse_lists([[("a", 1.0), ("a", 2.0)], bm25], (0.5, 0.5), top_k=10)
