import gzip
import hashlib
import json
import os

import numpy as np
import pytest
from scipy import sparse

from edge_rag import config
from edge_rag.artifacts import (
    ArtifactError,
    Checks,
    OldData,
    digest_of,
    guard_write,
    write_bytes,
    write_json,
)
from edge_rag.corpus import load_corpus, unit_id_for, unit_set_hash
from edge_rag.embeddings import cache_dir, legacy_corpus_cache_key, question_cache_key
from edge_rag.retrieval.dense_bm25 import Dense
from edge_rag.retrieval.hop import Hops, NodeIndex


@pytest.fixture
def old_root(tmp_path, monkeypatch):
    root = tmp_path / "old"
    root.mkdir()
    monkeypatch.setenv(config.OLD_DATA_ROOT_ENV, str(root))
    return root


def test_writes_under_the_old_data_root_are_refused(old_root, tmp_path):
    with pytest.raises(ArtifactError):
        write_json(old_root / "phase16" / "x.json", {})
    with pytest.raises(ArtifactError):
        guard_write(old_root / "a" / ".." / "b.json")
    assert write_json(tmp_path / "new" / "x.json", {"a": 1}).exists()


def test_paths_cannot_escape_the_set_directory(old_root):
    (old_root / "phase16").mkdir()
    with pytest.raises(ArtifactError):
        OldData("phase16", {}).path("../secret.json")
    with pytest.raises(ArtifactError):
        OldData("../elsewhere", {})


def test_a_mismatched_digest_is_refused():
    checks = Checks()
    checks.expect("ok", "abc", "abc")
    with pytest.raises(ArtifactError):
        checks.expect("bad", "abc", "abd")
    assert checks.records == [{"artifact": "ok", "digest": "abc"}]


def test_the_temporary_file_is_guarded_too(old_root, tmp_path):
    with pytest.raises(ArtifactError):
        write_bytes(old_root / "phase16" / "x.json.tmp", b"{}")
    (old_root / "victim.json").write_text("old", encoding="utf-8")
    outside = tmp_path / "new"
    outside.mkdir()
    try:
        os.symlink(old_root / "victim.json", outside / "x.json.tmp")
    except OSError:
        pytest.skip("symlinks need privileges on this machine")
    with pytest.raises(ArtifactError):
        write_json(outside / "x.json", {"a": 1})
    assert (old_root / "victim.json").read_text(encoding="utf-8") == "old"


def test_every_old_read_is_hashed_before_it_is_parsed(old_root):
    directory = old_root / "set"
    directory.mkdir()
    body = json.dumps({"a": 1}).encode("utf-8")
    (directory / "a.json").write_bytes(body)
    npz = directory / "b.npz"
    np.savez(npz, x=np.arange(3))
    pins = {
        "a.json": hashlib.sha256(body).hexdigest(),
        "b.npz": hashlib.sha256(npz.read_bytes()).hexdigest(),
    }
    checks = Checks()
    old = OldData("set", pins, checks)
    assert old.json("a.json") == {"a": 1}
    assert list(old.arrays("b.npz", "x")[0]) == [0, 1, 2]
    assert [r["artifact"] for r in checks.records] == ["sha256 a.json", "sha256 b.npz"]
    (directory / "a.json").write_bytes(b'{"a": 2}')
    (directory / "b.npz").write_bytes(b"not an npz")
    with pytest.raises(ArtifactError, match="sha256"):
        old.json("a.json")
    with pytest.raises(ArtifactError, match="sha256"):
        old.arrays("b.npz", "x")
    with pytest.raises(ArtifactError, match="no pinned"):
        OldData("set", {}).json("a.json")


def _write_corpus(directory, entries, ordered):
    directory.mkdir(parents=True)
    with gzip.open(directory / "corpus.jsonl.gz", "wt", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    manifest = {"corpus_file": "corpus.jsonl.gz", "ordered_unit_digest": ordered}
    (directory / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")


def _pinned(name):
    base = config.old_data_root() / name
    pins = {f: hashlib.sha256((base / f).read_bytes()).hexdigest() for f in os.listdir(base)}
    return OldData(name, pins)


def test_corpus_ids_and_order_are_verified(old_root):
    units = [("A", ["one.", "two."]), ("B", ["three."])]
    entries = [{"title": t, "sentences": s, "unit_id": unit_id_for(t, s)} for t, s in units]
    ids = [entry["unit_id"] for entry in entries]
    _write_corpus(old_root / "good", entries, digest_of(*ids))
    corpus = load_corpus(
        _pinned("good"), Checks(), ordered_digest=digest_of(*ids), set_hash=unit_set_hash(ids)
    )
    assert corpus.texts == ["A. one. two.", "B. three."]
    tampered = [dict(entries[0], sentences=["changed."]), entries[1]]
    _write_corpus(old_root / "bad", tampered, digest_of(*ids))
    with pytest.raises(ArtifactError):
        load_corpus(
            _pinned("bad"), Checks(), ordered_digest=digest_of(*ids), set_hash=unit_set_hash(ids)
        )


def test_legacy_keys_reproduce_the_recorded_cache_names():
    for spec in config.SETS.values():
        key = legacy_corpus_cache_key(
            config.EMBEDDING_MODEL, config.EMBEDDING_REVISION, spec.unit_set_hash
        )
        assert spec.vectors_file == f"cache/embeddings-{key}.npz"


def test_caches_are_per_corpus():
    keys = {
        question_cache_key("m", "r", spec.unit_set_hash, "same-questions", "dev")
        for spec in config.SETS.values()
    }
    assert len(keys) == len(config.SETS)
    assert len({cache_dir(name) for name in config.SETS}) == len(config.SETS)


def _toy_hops():
    # Rows: u0 (p1), u1, u2, u3. Nodes 0, 1 are entities, node 2 is a concept.
    rows = [[0, 1, 2], [0], [1, 2], [2]]
    indptr = np.cumsum([0, *(len(r) for r in rows)])
    indices = np.concatenate([np.array(r) for r in rows])
    incidence = sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(4, 3))
    index = NodeIndex(("u0", "u1", "u2", "u3"), ("entity", "entity", "concept"), incidence, "")
    vectors = np.eye(4, dtype=np.float32)
    return Hops(index, vectors), Dense(vectors, index.unit_ids)


def test_entity_hop_excludes_read_rows_and_ignores_concepts():
    hops, _dense = _toy_hops()
    first = [("u0", 1.0)]
    hop = hops.entity_hop(first, 100)
    # u3 shares only the concept node; u1 and u2 share one entity each, of equal df.
    assert [unit_id for unit_id, _ in hop] == ["u1", "u2"]
    weight = float(np.float32(np.log1p(4 / 3)))
    assert hop[0][1] == pytest.approx(weight)


def test_relevance_hop_mixes_rarity_with_similarity():
    hops, _dense = _toy_hops()
    question = np.array([0, 0, 1, 0], dtype=np.float32)
    hop = hops.relevance_hop([("u0", 1.0)], question, 100, 0.75)
    assert hop == [("u2", 0.75), ("u1", 0.0)]
    with pytest.raises(ValueError):
        hops.relevance_hop([("u0", 1.0)], question, 100, 0.0)


def test_dense_breaks_ties_by_unit_id():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dense = Dense(vectors, ["b", "a", "c"])
    assert dense.retrieve(np.array([1.0, 0.0], dtype=np.float32), 2) == [("a", 1.0), ("b", 1.0)]


def test_hop_candidates_exclude_dense_top_read_depth_and_include_the_next():
    depth = config.READ_DEPTH
    n = depth + 3
    # Every unit holds entity node 0, so all share it with p1 = u0.
    indptr = np.arange(n + 1)
    incidence = sparse.csr_matrix((np.ones(n), np.zeros(n, dtype=int), indptr), shape=(n, 1))
    ids = tuple(f"u{i:02d}" for i in range(n))
    index = NodeIndex(ids, ("entity",), incidence, "")
    vectors = np.eye(n, dtype=np.float32)
    hops = Hops(index, vectors)
    first = [(unit_id, 1.0) for unit_id in ids[: depth + 1]]  # dense ranks 1..depth+1
    question = np.ones(n, dtype=np.float32)
    expected = set(ids[depth:])  # the unit at rank depth + 1 and the ones Dense never listed
    assert {u for u, _ in hops.entity_hop(first, 100)} == expected
    assert {u for u, _ in hops.relevance_hop(first, question, 100, 0.75)} == expected
    assert not set(ids[:depth]) & expected
