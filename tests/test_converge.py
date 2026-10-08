import numpy as np
import pytest
from scipy import sparse

from edge_rag import config
from edge_rag.converge import fusion, hop_ms, refined_query, seeds
from edge_rag.retrieval.hop import Hops, NodeIndex


def _hops(rows, types, ids=None):
    indptr = np.cumsum([0, *(len(r) for r in rows)])
    indices = np.concatenate([np.array(r, dtype=int) for r in rows])
    shape = (len(rows), len(types))
    incidence = sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=shape)
    ids = ids or tuple(f"u{i}" for i in range(len(rows)))
    index = NodeIndex(tuple(ids), tuple(types), incidence, "")
    return Hops(index, np.eye(len(rows), dtype=np.float32))


def test_seeds_take_dense_top_5_then_new_bm25_top_5_in_order():
    dense = ["a", "b", "c", "d", "e", "f"]
    bm25 = ["c", "x", "a", "y", "z", "w"]
    assert seeds(dense, bm25) == ["a", "b", "c", "d", "e", "x", "y", "z"]
    assert seeds(dense, ["e", "d", "c", "b", "a"]) == ["a", "b", "c", "d", "e"]
    assert len(seeds(dense, ["p", "q", "r", "s", "t"])) == 2 * config.SEEDS_PER_LIST


def test_seed_hop_scores_rarity_excludes_the_given_units_and_ignores_concepts():
    # Nodes 0, 1 are entities, node 2 a concept; df 3 for each entity over N = 5 units.
    hops = _hops([[0, 1, 2], [0], [1, 2], [2], [0, 1]], ("entity", "entity", "concept"))
    weight = float(np.float32(np.log1p(5 / 4)))
    hop = hops.seed_hop("u0", {"u0", "u1"}, 100)
    assert [unit_id for unit_id, _ in hop] == ["u4", "u2"]
    assert hop[0][1] == pytest.approx(2 * weight)
    assert hop[1][1] == pytest.approx(weight)
    assert hops.seed_hop("u3", {"u3"}, 100) == []  # a seed with no entity node
    assert [u for u, _ in hops.seed_hop("u0", {"u0"}, 2)] == ["u4", "u1"]  # tie u1, u2 by id


def test_seed_hop_from_dense_first_unit_equals_entity_hop():
    rng = np.random.default_rng(5)
    n, nodes = 40, 8
    rows = [sorted(rng.choice(nodes, size=rng.integers(0, 4), replace=False)) for _ in range(n)]
    types = ("entity",) * 6 + ("concept",) * 2
    hops = _hops(rows, types, tuple(f"u{i:02d}" for i in range(n)))
    for trial in range(20):
        order = rng.permutation(n)
        first = [(f"u{i:02d}", 1.0) for i in order[:15]]
        read = [u for u, _ in first[: config.READ_DEPTH]]
        for depth in (3, 100):
            assert hops.seed_hop(read[0], read, depth) == hops.entity_hop(first, depth), trial


def test_two_seeds_at_rank_50_beat_one_seed_at_rank_1():
    first = [f"a{i:02d}" for i in range(1, 50)] + ["x"]
    second = [f"b{i:02d}" for i in range(1, 50)] + ["x"]
    convergent = hop_ms([first, second, ["y"]])
    # x: 2 / 110 = 0.0182; y, a01 and b01: 1 / 61 = 0.0164.
    assert convergent[0] == "x"
    assert convergent[1:4] == ["a01", "b01", "y"]
    assert len(convergent) == config.DEPTH


def test_an_empty_per_seed_list_adds_nothing():
    lists = [["a", "b"], ["c", "a"]]
    assert hop_ms([lists[0], [], lists[1]]) == hop_ms(lists) == ["a", "c", "b"]
    assert hop_ms([[], []]) == []


def test_refined_query_on_two_dimensions():
    question = np.array([1.0, 0.0], dtype=np.float32)
    same = refined_query(question, np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32))
    assert same.dtype == np.float32
    assert same == pytest.approx([2**-0.5, 2**-0.5])
    # c = (0.5, 0.5), q + c = (1.5, 0.5), norm sqrt(2.5).
    mixed = refined_query(question, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert mixed == pytest.approx([1.5 / 2.5**0.5, 0.5 / 2.5**0.5])


def test_four_list_rrf_with_an_empty_hop_ms_is_rrf_prf():
    lists = {"p10-a": ["a", "b"], "bm25": ["b", "c"], "dense-prf": ["a"], "hop": ["d"]}
    # a: 2 / 61; b: 1 / 61 + 1 / 62; c: 1 / 62.
    assert fusion("rrf-prf", lists) == ["a", "b", "c"]
    assert fusion("mch", {**lists, "hop-ms": []}) == ["a", "b", "c"]
    # c from hop-ms ties b at 1 / 61 + 1 / 62 and follows it by id.
    assert fusion("mch", {**lists, "hop-ms": ["c"]}) == ["a", "b", "c"]
    # d: 1 / 61 > c: 1 / 62.
    assert fusion("rrf-1s", lists) == ["a", "b", "d", "c"]


def _seed_case(top_rows):
    """15 units, Dense order u00..u14; `top_rows` are the read set's rows (ranks 1-10),
    u11 shares entity 0, u12 entity 1; node 2 is a concept."""
    rows = [*top_rows, [], [0], [1], [], []]
    hops = _hops(rows, ("entity", "entity", "concept"), tuple(f"u{i:02d}" for i in range(15)))
    first = [(f"u{i:02d}", 1.0 - i / 100) for i in range(15)]
    return hops, first


def test_first_entity_seed_skips_dense_units_without_an_entity():
    top = [[2], [], [0], [1], [], [], [], [], [], []]  # u00 concept only, u02 first entity
    hops, first = _seed_case(top)
    assert hops.entity_hop(first, 100) == []  # default P1 seed: u00 has no entity
    assert [u for u, _ in hops.entity_hop(first, 100, seed="first-entity")] == ["u11"]


def test_first_entity_seed_is_empty_when_no_read_unit_has_an_entity():
    hops, first = _seed_case([[2]] + [[]] * 9)  # entities only at u11, u12 (rank 12, 13)
    assert hops.entity_hop(first, 100, seed="first-entity") == []
    with pytest.raises(ValueError):
        hops.entity_hop(first, 100, seed="p2")


def test_first_entity_seed_equals_p1_when_p1_has_an_entity():
    hops, first = _seed_case([[1, 2], [0]] + [[]] * 8)
    p1 = hops.entity_hop(first, 100)
    assert [u for u, _ in p1] == ["u12"]
    assert hops.entity_hop(first, 100, seed="first-entity") == p1
