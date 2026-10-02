"""The entity index and the two hops from Dense's first paragraph.

Copied from the old `nodes/index.load_node_index`, `retrieval/conceptual.rarity_weights`,
`evaluation/second_hop.node_hop_columnwise` / `relevance_hop_columnwise` and
`evaluation/phase14` (similarity, min-max, mix, cut). Neither hop reads the question's own
entities: both start from `p1`, Dense's top paragraph, and exclude Dense's top `READ_DEPTH`.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from edge_rag import config
from edge_rag.artifacts import ArtifactError, Checks, OldData, digest_of
from edge_rag.retrieval.dense_bm25 import Hit


@dataclass(frozen=True, eq=False)
class NodeIndex:
    unit_ids: tuple[str, ...]
    node_types: tuple[str, ...]
    incidence: sparse.csr_matrix
    digest: str


def gliner_configuration_digest(configuration: dict[str, Any]) -> str:
    """Old `LocalExtractor.configuration_digest`, recomputed from a recorded manifest."""
    text = json.dumps(configuration, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_node_index(old: OldData, nodes_dir: str, checks: Checks, *, recorded: str) -> NodeIndex:
    extraction = old.json(f"{nodes_dir}/extraction.json")
    checks.expect(
        "GLiNER configuration digest",
        gliner_configuration_digest(extraction["configuration"]),
        config.GLINER_CONFIGURATION_DIGEST,
    )
    stem = f"{nodes_dir}/nodes-{str(extraction['digest'])[:16]}"
    payload = old.json(f"{stem}.json")
    with old.path(f"{stem}.npz").open("rb") as handle, np.load(handle, allow_pickle=False) as z:
        indptr, indices = z["indptr"], z["indices"]
    stored = payload.pop("digest", None)
    measured = digest_of(
        indptr.astype(np.int64),
        indices.astype(np.int64),
        json.dumps(payload, sort_keys=True, ensure_ascii=True),
    )
    checks.expect("entity index sidecar digest", str(stored), recorded)
    checks.expect("entity index digest", measured, recorded)
    shape = tuple(payload["shape"])
    incidence = sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=shape)
    types = tuple(str(node["type"]) for node in payload["nodes"])
    if [int(node["node_id"]) for node in payload["nodes"]] != list(range(len(types))):
        raise ArtifactError("node ids are not 0..n-1 in order")
    return NodeIndex(tuple(payload["unit_ids"]), types, incidence, measured)


def node_weights(index: NodeIndex) -> np.ndarray:
    """`w = log(1 + N / (1 + df))` over every node, float32 (decision D7 of the old line)."""
    stored = sparse.csr_matrix(index.incidence, copy=True)
    stored.eliminate_zeros()
    df = np.bincount(stored.indices, minlength=stored.shape[1]).astype(np.int64)
    n_units = index.incidence.shape[0]
    return np.log1p(n_units / (1.0 + df.astype(np.float64))).astype(np.float32)


@dataclass(frozen=True, eq=False)
class ArmColumns:
    mask: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray


def arm_columns(index: NodeIndex, types: Sequence[str] = config.ENTITY_TYPES) -> ArmColumns:
    if not index.incidence.has_sorted_indices:
        raise ArtifactError("the column-wise hop needs rows that list nodes in order")
    wanted = set(types)
    mask = np.array([node_type in wanted for node_type in index.node_types], dtype=bool)
    csc = index.incidence.tocsc()
    return ArmColumns(mask=mask, indptr=csc.indptr, indices=csc.indices)


class Hops:
    """P10-C's Entity Hop and P14's relevance-ordered hop over one index."""

    def __init__(self, index: NodeIndex, vectors: np.ndarray) -> None:
        self.index = index
        self.weights = node_weights(index)
        self.columns = arm_columns(index)
        self.vectors = vectors  # rows are the index rows, in order
        self._rows = {unit_id: row for row, unit_id in enumerate(index.unit_ids)}

    def _rarity(self, first: Sequence[Hit]) -> tuple[np.ndarray, np.ndarray]:
        """Rarity scores of every row, and the positive rows outside Dense's read set."""
        read = [self._rows[unit_id] for unit_id, _ in first[: config.READ_DEPTH]]
        if not read:
            raise ValueError("the dense list is empty")
        incidence = self.index.incidence
        p1 = read[0]
        p1_nodes = incidence.indices[incidence.indptr[p1] : incidence.indptr[p1 + 1]]
        shared = np.sort(p1_nodes[self.columns.mask[p1_nodes]])
        scores = np.zeros(incidence.shape[0])
        for node in shared:
            rows = self.columns.indices[self.columns.indptr[node] : self.columns.indptr[node + 1]]
            scores[rows] += self.weights[node]
        excluded = np.zeros(incidence.shape[0], dtype=bool)
        excluded[read] = True
        return scores, np.nonzero((scores > 0.0) & ~excluded)[0]

    def entity_hop(self, first: Sequence[Hit], depth: int) -> list[Hit]:
        scores, positive = self._rarity(first)
        kept = positive
        if positive.size > depth:
            values = scores[positive]
            threshold = -np.partition(-values, depth - 1)[depth - 1]
            kept = positive[values >= threshold]
        unit_ids = self.index.unit_ids
        ordered = sorted((int(r) for r in kept), key=lambda r: (-float(scores[r]), unit_ids[r]))
        return [(unit_ids[r], float(scores[r])) for r in ordered[:depth]]

    def relevance_hop(
        self, first: Sequence[Hit], question_vector: np.ndarray, depth: int, alpha: float
    ) -> list[Hit]:
        if alpha <= 0.0:
            raise ValueError("alpha = 0 is the entity hop")
        scores, positive = self._rarity(first)
        if positive.size == 0:
            return []
        gathered = self.vectors[positive].astype(np.float32, copy=False)
        similarity = gathered @ np.asarray(question_vector, dtype=np.float32)
        mixed = (1.0 - alpha) * _min_max(scores[positive]) + alpha * _min_max(similarity)
        kept = np.arange(positive.size)
        if positive.size > depth:
            threshold = -np.partition(-mixed, depth - 1)[depth - 1]
            kept = np.nonzero(mixed >= threshold)[0]
        unit_ids = self.index.unit_ids
        ordered = sorted(
            (int(i) for i in kept), key=lambda i: (-float(mixed[i]), unit_ids[int(positive[i])])
        )
        return [(unit_ids[int(positive[i])], float(mixed[i])) for i in ordered[:depth]]


def _min_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = values.min(), values.max()
    if high == low:
        return np.zeros_like(values)
    return (values - low) / (high - low)
