"""Dense and BM25, copied from the old `retrieval/dense.py` and `retrieval/bm25.py`.

Ties are broken by unit id. Dense is an exact per-query matrix-vector product, as recorded:
a batched product could change the float accumulation order and so a tie.
"""

import json
from collections.abc import Sequence

import bm25s
import numpy as np

from edge_rag.artifacts import digest_of

Hit = tuple[str, float]


class Dense:
    def __init__(self, vectors: np.ndarray, unit_ids: Sequence[str]) -> None:
        if vectors.shape[0] != len(unit_ids):
            raise ValueError(f"{vectors.shape[0]} vectors for {len(unit_ids)} unit ids")
        self.vectors = vectors
        self.unit_ids = list(unit_ids)

    def retrieve(self, query_vector: np.ndarray, top_k: int) -> list[Hit]:
        scores = self.vectors @ query_vector
        k = min(top_k, len(self.unit_ids))
        if k < len(self.unit_ids):
            candidates = np.argpartition(-scores, k - 1)[:k]
        else:
            candidates = np.arange(len(self.unit_ids))
        ordered = sorted(candidates, key=lambda i: (-float(scores[i]), self.unit_ids[i]))
        return [(self.unit_ids[i], float(scores[i])) for i in ordered]


class BM25:
    """bm25s 0.3.11 defaults, English stopwords, over the same `indexable_text` as Dense."""

    def __init__(self, texts: Sequence[str], unit_ids: Sequence[str], stopwords: str = "en"):
        self.unit_ids = list(unit_ids)
        self.stopwords = stopwords
        tokens = bm25s.tokenize(list(texts), stopwords=stopwords, show_progress=False)
        self._index = bm25s.BM25()
        self._index.index(tokens, show_progress=False)

    def digest(self) -> str:
        """The old `phase9.bm25_digest`: score arrays, document count, vocabulary, unit ids."""
        scores = self._index.scores
        arrays = [np.asarray(scores[name]) for name in ("data", "indices", "indptr")]
        vocabulary = sorted(self._index.vocab_dict.items())
        return digest_of(
            *arrays,
            int(scores["num_docs"]),
            json.dumps(vocabulary, ensure_ascii=True),
            *self.unit_ids,
        )

    def retrieve(self, query: str, top_k: int) -> list[Hit]:
        k = min(top_k, len(self.unit_ids))
        tokens = bm25s.tokenize(query, stopwords=self.stopwords, show_progress=False)
        indices, scores = self._index.retrieve(tokens, k=k, show_progress=False)
        # Zero-score hits are kept, as recorded.
        return [
            (self.unit_ids[int(index)], float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
        ]
