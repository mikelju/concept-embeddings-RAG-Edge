import pytest

from edge_rag.results import ghost_cost, markdown, spread_entry

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


def test_agent_offline_cost_follows_the_g_l_index_it_searched() -> None:
    searched = {**MANIFESTS["g-l"], "seconds": {"encode": 50.0, "index": 8.0}}
    cost = ghost_cost("g-a1", {**MANIFESTS, "g-l": searched})
    assert cost["offline_seconds"] == 58.0


def test_spread_wins_are_questions_only_the_rebuild_supports() -> None:
    entry = spread_entry("s4", "s4/g-l.jsonl.gz", "x", 2, reported=[1, 0, 0], rebuild=[0, 1, 1])
    assert entry["against_reported_g_l"]["wins"] == 2
    assert entry["against_reported_g_l"]["losses"] == 1


def test_markdown_states_unmeasured_ghosts_and_skips_an_empty_paired_table() -> None:
    old = {
        "metrics": {
            "full_support_at_budget": {"1024": 1, "2048": 2, "4096": 3},
            "full_support_at_k": {"2": 0, "5": 1, "20": 2},
            "recall_at_k": {"5": 0.5, "100": 0.9},
            "ndcg_at_10": 0.4,
        },
        "cost": {"label": "old"},
    }
    entry = {
        "set": "hotpotqa-dev",
        "n": 3,
        "best_old_by_fs_at_budget": "p14",
        "systems": {"p14": old},
        "ghosts_not_measured": {"g-l": "stopped (F8)"},
        "paired_full_support_at_2048": [],
        "g_l_build_spread_at_2048": [],
    }
    text = markdown({"sets": [entry]})
    assert "g-l: not measured on this set (stopped (F8))." in text
    assert "McNemar" not in text
