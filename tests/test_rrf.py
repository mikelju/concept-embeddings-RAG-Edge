from fractions import Fraction

from edge_rag.retrieval.rrf import (
    BOILERPLATE,
    NEAR_DUPLICATE,
    SOURCE_CAP,
    boilerplate_flags,
    diversify,
    jaccard,
    rrf,
    rrf_scores,
    shingles,
)


def test_rrf_scores_and_order_by_hand():
    lists = [["a", "b", "c"], ["b", "d"], ["c", "a"]]
    scores = rrf_scores(lists)
    assert scores["a"] == Fraction(1, 61) + Fraction(1, 62)
    assert scores["c"] == Fraction(1, 61) + Fraction(1, 63)
    # d is in one list only: the lists without it add nothing.
    assert scores["d"] == Fraction(1, 62)
    # a and b both score 1/61 + 1/62: the smaller unit id goes first.
    assert scores["a"] == scores["b"]
    assert rrf(lists) == ["a", "b", "c", "d"]
    assert rrf([["z"], ["y"]]) == ["y", "z"]


def test_three_list_tie_goes_to_the_smaller_unit_id():
    # x holds ranks 1, 7, 2 and z ranks 7, 2, 1: equal exact scores, but the float sums in list
    # order differ in the last bit and would put z first.
    fill = [f"f{i}" for i in range(10)]
    lists = [["x", *fill[0:5], "z"], [fill[5], "z", *fill[6:10], "x"], ["z", "x"]]
    assert 1 / 61 + 1 / 67 + 1 / 62 < 1 / 67 + 1 / 62 + 1 / 61
    scores = rrf_scores(lists)
    assert scores["x"] == scores["z"]
    assert rrf(lists)[:2] == ["x", "z"]


def test_rrf_truncates_at_depth():
    lists = [[f"{name}{i:03d}" for i in range(100)] for name in "xyz"]
    assert len(rrf(lists)) == 300
    assert len(rrf(lists, depth=100)) == 100


def test_shingles_and_jaccard():
    assert shingles("One two, three four five six!") == {
        ("one", "two", "three", "four", "five"),
        ("two", "three", "four", "five", "six"),
    }
    assert shingles("Hello, World") == {("hello", "world")}
    assert shingles("  ") == frozenset()
    a = shingles("a b c d e f g h")
    assert jaccard(a, a) == 1.0
    assert jaccard(a, shingles("a b c d e f g x")) == 3 / 5


def test_boilerplate_flag():
    bodies = ["Advertisement", "advertisement!", "unique text here", "", "same", "Same."]
    titles = ["T1", "T2", "T3", "T4", "T5", "T5"]
    assert boilerplate_flags(bodies, titles).tolist() == [True, True, False, True, False, False]


def test_diversify_rule_order_kept_then_demoted_and_truncation():
    long = "the quick brown fox jumps over the lazy dog today"
    units = {
        "u1": ("T1", long, False),
        "u2": ("T9", "advertisement", True),
        "u3": ("T2", long + ".", False),  # near-duplicate of u1
        "u4": ("T1", "a second paragraph of the first article here", False),
        "u5": ("T1", "a third paragraph that is not like the others", False),  # cap
        "u6": ("T1", long, False),  # near-duplicate before the cap
        "u7": ("T1", long, True),  # boilerplate before both
        "u8": ("T3", "another source entirely with its own words", False),
    }
    order = list(units)
    titles = {u: v[0] for u, v in units.items()}
    bodies = {u: v[1] for u, v in units.items()}
    flags = {u: v[2] for u, v in units.items()}
    output, demoted = diversify(order, titles, bodies, flags)
    assert output == ["u1", "u4", "u8", "u2", "u3", "u5", "u6", "u7"]
    assert demoted == {
        "u2": BOILERPLATE,
        "u3": NEAR_DUPLICATE,
        "u5": SOURCE_CAP,
        "u6": NEAR_DUPLICATE,
        "u7": BOILERPLATE,
    }
    assert diversify(order, titles, bodies, flags, depth=4)[0] == ["u1", "u4", "u8", "u2"]
    many = [f"v{i:03d}" for i in range(300)]
    output, _ = diversify(
        many,
        dict.fromkeys(many, "T") | {u: u for u in many},
        {u: f"body of unit {u}" for u in many},
        dict.fromkeys(many, False),
    )
    assert output == many[:100]


def test_gold_demotions_count_context_and_top_100_losses():
    from edge_rag.corpus import Question
    from edge_rag.fuse import gold_demotions

    questions = [Question("q", "?", ("g1", "g2"), "dev")]
    rrf3 = {"q": ["g1", "x", "g2"]}
    f3 = {"q": ["x", "g2", "g1"]}  # g1 demoted by the cap, still in the top 100
    counts = {"g1": 1000, "x": 1000, "g2": 1000}
    found = gold_demotions(questions, rrf3, f3, {"q": {"g1": SOURCE_CAP}}, counts)
    # RRF3 reads g1 and x within 2,048 tokens; F3 reads x and g2: g1 left the context.
    assert found[SOURCE_CAP] == {"in_context_at_budget": 1, "in_top_100": 0}
    assert found[BOILERPLATE] == {"in_context_at_budget": 0, "in_top_100": 0}
