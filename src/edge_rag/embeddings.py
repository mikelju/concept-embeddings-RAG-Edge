"""The embedding cache: reading the inherited vectors, keying new ones per corpus.

Copied from the old `embeddings/cache.py`. The old `question_cache_key` does not include the
corpus (handover 5.1, "Gotchas"), so the key this harness writes, `question_cache_key`, adds
the corpus unit-set hash, and every corpus gets its own cache directory. `legacy_*` keys
reproduce the old names, so an inherited file is found by the key it was written under.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from edge_rag import config
from edge_rag.artifacts import ArtifactError, guard_write


def _sha1_16(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def legacy_corpus_cache_key(model: str, revision: str, corpus_hash: str) -> str:
    return _sha1_16(f"{model}|{revision}|{corpus_hash}|1")


def question_set_hash(qids: Sequence[str]) -> str:
    return _sha1_16("\n".join(sorted(qids)))


def legacy_question_cache_key(model: str, revision: str, question_set: str, split: str) -> str:
    return _sha1_16(f"question|{model}|{revision}|{question_set}|{split}|1")


def question_cache_key(
    model: str, revision: str, corpus_hash: str, question_set: str, split: str
) -> str:
    """The key this harness writes: the corpus is part of it, unlike the legacy key."""
    return _sha1_16(f"question|{model}|{revision}|corpus={corpus_hash}|{question_set}|{split}|1")


def cache_dir(set_name: str) -> Path:
    """Where this harness writes caches for one corpus: never shared across corpora."""
    if set_name not in config.SETS:
        raise ArtifactError(f"unknown set {set_name!r}")
    return guard_write(config.DATA_DIR / "cache" / set_name)


def load_vectors(path: Path) -> tuple[np.ndarray, list[str]]:
    """An `embeddings-<key>.npz`: float32 `vectors` and their `unit_ids`, no pickle."""
    with np.load(path, allow_pickle=False) as payload:
        vectors = payload["vectors"]
        unit_ids = [str(unit_id) for unit_id in payload["unit_ids"]]
    if vectors.dtype != np.float32 or vectors.shape[0] != len(unit_ids):
        raise ArtifactError(f"{path.name}: {vectors.dtype} {vectors.shape} for {len(unit_ids)} ids")
    return vectors, unit_ids


def vectors_digest(vectors: np.ndarray) -> str:
    """sha256 over the float32 vector bytes, read in place (old `phase9.vectors_digest`)."""
    block = np.ascontiguousarray(vectors, dtype=np.float32)
    return hashlib.sha256(block.reshape(-1).view(np.uint8)).hexdigest()


class QueryTable:
    """Cached question vectors served by qid; the model is never called."""

    def __init__(self, vectors: np.ndarray, qids: Sequence[str]) -> None:
        self._rows = {qid: row for row, qid in enumerate(qids)}
        if len(self._rows) != len(qids):
            raise ArtifactError("a qid appears twice in the question cache")
        self._vectors = vectors

    def vector(self, qid: str) -> np.ndarray:
        row = self._rows.get(qid)
        if row is None:
            raise ArtifactError(f"question {qid!r} is not in the cache")
        return self._vectors[row]
