"""The budget fill, Full Support, Recall@k, nDCG@k and the exact paired test.

Copied from the old `evaluation/budget.py`, `evaluation/metrics.py` and
`evaluation/fullwiki.exact_two_sided_p` at commit 9a9ad3f (integer true division).
"""

import math
from collections.abc import Mapping, Sequence


def fill_context(ranked: Sequence[str], token_counts: Mapping[str, int], budget: int) -> list[str]:
    """Walk the ranking; include a unit if it fits whole, skip it if not, and carry on."""
    selected: list[str] = []
    used = 0
    for unit_id in ranked:
        cost = token_counts[unit_id]
        if used + cost <= budget:
            selected.append(unit_id)
            used += cost
    return selected


def _need_gold(gold: Sequence[str]) -> None:
    if not gold:
        raise ValueError("a question without gold units has no defined retrieval metric")


def full_support(context: Sequence[str], gold: Sequence[str]) -> int:
    _need_gold(gold)
    return int(set(gold) <= set(context))


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    _need_gold(gold)
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG (the BEIR metric); a gold id counts once, at its first rank."""
    _need_gold(gold)
    wanted = set(gold)
    seen: set[str] = set()
    dcg = 0.0
    for position, unit_id in enumerate(ranked[:k], start=1):
        if unit_id in wanted and unit_id not in seen:
            seen.add(unit_id)
            dcg += 1.0 / math.log2(position + 1)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(wanted), k) + 1))
    return dcg / ideal


def exact_two_sided_p(wins: int, losses: int) -> float:
    """The exact McNemar test: two-sided binomial(n = wins + losses, 1/2) on the discordant."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    # Integer true division: correctly rounded for any n (no float overflow past n = 1,024).
    return min(1.0, 2 * tail / 2**n)


def paired(a: Sequence[int], b: Sequence[int]) -> dict[str, float | int]:
    """b against a over the same questions: wins, losses, ties and the exact p."""
    if len(a) != len(b):
        raise ValueError("paired outcomes must cover the same questions")
    wins = sum(1 for x, y in zip(a, b, strict=True) if y > x)
    losses = sum(1 for x, y in zip(a, b, strict=True) if y < x)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(a) - wins - losses,
        "p": exact_two_sided_p(wins, losses),
    }
