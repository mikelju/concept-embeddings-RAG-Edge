"""Min-max weighted fusion, copied from the old `retrieval/fusion.py` (`fuse_lists`).

Per query, each component list is min-max normalized over its own range; a unit a component
did not return scores 0.0; a constant or empty list contributes 0 everywhere. The weighted sum
is accumulated component by component in fixed order (the float order the figures were
measured with), then ranked by score descending and unit id ascending.
"""

from collections.abc import Sequence

from edge_rag.retrieval.dense_bm25 import Hit

ABSENT_FLOOR = 0.0


def _scored_once(hits: Sequence[Hit]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for unit_id, score in hits:
        if unit_id in scores:
            raise ValueError(f"a component returned {unit_id!r} twice in one query")
        scores[unit_id] = float(score)
    return scores


def min_max_normalize(hits: Sequence[Hit], units: Sequence[str]) -> dict[str, float]:
    scores = _scored_once(hits)
    normalized = dict.fromkeys(units, ABSENT_FLOOR)
    if not scores:
        return normalized
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return normalized
    span = hi - lo
    for unit_id in normalized:
        if unit_id in scores:
            normalized[unit_id] = (scores[unit_id] - lo) / span
    return normalized


def fuse_lists(
    lists: Sequence[Sequence[Hit]], weights: Sequence[float], *, top_k: int
) -> list[Hit]:
    if len(lists) < 2 or len(lists) != len(weights):
        raise ValueError(f"{len(weights)} weights for {len(lists)} lists")
    if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"not a convex combination: {tuple(weights)}")
    units = sorted({unit_id for hits in lists for unit_id, _ in hits})
    fused = dict.fromkeys(units, 0.0)
    for hits, weight in zip(lists, weights, strict=True):
        for unit_id, value in min_max_normalize(hits, units).items():
            fused[unit_id] += weight * value
    ordered = sorted(fused.items(), key=lambda entry: (-entry[1], entry[0]))
    return ordered[:top_k]
