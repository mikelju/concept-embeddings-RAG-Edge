import pytest

from edge_rag.results import ghost_cost

MANIFESTS = {
    "g-l": {
        "seconds": {"encode": 3000.0, "index": 600.0, "search": 10.0},
        "search_seconds_per_question": 0.01,
        "cost_per_hr_usd": 0.72,
    },
    "g-r": {"rerank_seconds_per_question": 0.5, "cost_per_hr_usd": 0.72},
    "g-a1": {"seconds_per_question": 2.0, "cost_per_hr_usd": 0.72},
}


def test_g_l_cost_is_encode_and_index_offline_and_search_online() -> None:
    cost = ghost_cost("g-l", MANIFESTS)
    assert cost["offline_seconds"] == 3600.0
    assert cost["offline_usd"] == pytest.approx(0.72)
    assert cost["online_seconds_per_question"] == pytest.approx(0.01)


def test_reranker_adds_its_time_to_the_search_and_agent_counts_its_own_run() -> None:
    assert ghost_cost("g-r", MANIFESTS)["online_seconds_per_question"] == pytest.approx(0.51)
    agent = ghost_cost("g-a1", MANIFESTS)
    assert agent["online_seconds_per_question"] == pytest.approx(2.0)
    assert agent["online_usd_per_question"] == pytest.approx(2.0 * 0.72 / 3600)
    assert agent["offline_seconds"] == 3600.0
