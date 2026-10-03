"""Weight-free fusion of Phase 03 (spec, frozen definition): RRF3 and the diversity pass of F3.

RRF (Cormack, Clarke and Buettcher 2009): a unit scores the sum over the lists holding it of
1 / (k + rank), rank from 1; ties go to the smaller unit id. Scores are exact fractions, so equal
sums are equal whatever the order of the lists (float sums can differ in the last bit). F3 walks
RRF3's whole order once and demotes a unit by the first rule that applies: boilerplate,
near-duplicate of a kept unit, or a third unit of a source already kept twice; kept units come
first, then the demoted, in order.
"""

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction

import numpy as np

from edge_rag import config

WORD = re.compile(r"\w+")
BOILERPLATE, NEAR_DUPLICATE, SOURCE_CAP = "boilerplate", "near-duplicate", "source cap"
RULES = (BOILERPLATE, NEAR_DUPLICATE, SOURCE_CAP)


def rrf_scores(lists: Sequence[Sequence[str]], k: int = config.RRF_K) -> dict[str, Fraction]:
    scores: dict[str, Fraction] = {}
    for ranked in lists:
        if len(set(ranked)) != len(ranked):
            raise ValueError("a fused list holds a unit twice")
        for rank, unit_id in enumerate(ranked, start=1):
            scores[unit_id] = scores.get(unit_id, Fraction(0)) + Fraction(1, k + rank)
    return scores


def rrf(
    lists: Sequence[Sequence[str]], k: int = config.RRF_K, depth: int | None = None
) -> list[str]:
    """The fused order, score descending then unit id ascending; the whole union if no depth."""
    scores = rrf_scores(lists, k)
    order = sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))
    return order if depth is None else order[:depth]


def normalize(body: str) -> str:
    return " ".join(WORD.findall(body.lower()))


def shingles(body: str, size: int = config.SHINGLE_SIZE) -> frozenset[tuple[str, ...]]:
    """Word `size`-grams of the normalized body; a shorter body is one shingle, itself."""
    tokens = tuple(normalize(body).split())
    if len(tokens) < size:
        return frozenset([tokens]) if tokens else frozenset()
    return frozenset(tokens[i : i + size] for i in range(len(tokens) - size + 1))


def jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def body_hash(body: str) -> int:
    digest = hashlib.sha1(normalize(body).encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:8], "little")


def boilerplate_flags(bodies: Iterable[str], titles: Sequence[str]) -> np.ndarray:
    """True for a unit whose normalized body is empty or appears under two or more titles;
    bodies are compared by an 8-byte hash so a 5 M unit corpus fits in memory (plan D4)."""
    hashes = np.fromiter((body_hash(body) for body in bodies), dtype=np.uint64, count=len(titles))
    first_title: dict[int, str] = {}
    repeated: set[int] = set()
    for value, title in zip(hashes.tolist(), titles, strict=True):
        seen = first_title.setdefault(value, title)
        if seen != title:
            repeated.add(value)
    empty = body_hash("")
    return np.fromiter(
        (value == empty or value in repeated for value in hashes.tolist()),
        dtype=bool,
        count=len(hashes),
    )


def diversify(
    order: Sequence[str],
    titles: Mapping[str, str],
    bodies: Mapping[str, str],
    flags: Mapping[str, bool],
    cap: int = config.SOURCE_CAP,
    threshold: float = config.NEAR_DUPLICATE_JACCARD,
    depth: int | None = config.DEPTH,
) -> tuple[list[str], dict[str, str]]:
    """F3's order (kept, then demoted, truncated) and the rule that demoted each unit."""
    kept: list[str] = []
    kept_shingles: list[frozenset[tuple[str, ...]]] = []
    per_title: dict[str, int] = {}
    demoted: dict[str, str] = {}
    for unit_id in order:
        if flags[unit_id]:
            demoted[unit_id] = BOILERPLATE
            continue
        own = shingles(bodies[unit_id])
        size = len(own)
        # Jaccard >= threshold needs the smaller set at least `threshold` of the larger one
        # (the margin keeps a float product from skipping an exact boundary case).
        if any(
            min(size, len(other)) >= threshold * max(size, len(other)) - 1e-9
            and jaccard(own, other) >= threshold
            for other in kept_shingles
        ):
            demoted[unit_id] = NEAR_DUPLICATE
            continue
        title = titles[unit_id]
        if per_title.get(title, 0) >= cap:
            demoted[unit_id] = SOURCE_CAP
            continue
        kept.append(unit_id)
        kept_shingles.append(own)
        per_title[title] = per_title.get(title, 0) + 1
    output = kept + [unit_id for unit_id in order if unit_id in demoted]
    return (output if depth is None else output[:depth]), demoted
