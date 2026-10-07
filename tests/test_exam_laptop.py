import json
from pathlib import Path

import numpy as np
import pytest

from edge_rag import exam_laptop, old_nodes, scoring
from edge_rag.artifacts import ArtifactError
from edge_rag.old_nodes import ExtractionRecord

UNITS = [
    ("a1", "pa", "Alpha parsing", "We train a parser on Penn Treebank."),
    ("a2", "pa", "Alpha parsing", "Penn Treebank results beat the baseline."),
    ("a3", "pa", "Alpha parsing", "FLOAT SELECTED: Table 1: scores."),
    ("b1", "pb", "Beta speech", "We record speech in Basque villages."),
    ("b2", "pb", "Beta speech", "The baseline is a Kaldi recognizer."),
]
QUESTIONS = [
    ("q1", "pa", "What baseline do they use?"),
    ("q2", "pb", "What baseline do they use?"),
    ("q3", "pb", "Where is speech recorded?"),
]
ENTITIES = {"a1": ("Penn Treebank",), "a2": ("penn treebank", "The Baseline"), "b2": ("Kaldi",)}


def encode(texts):
    vectors = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().split():
            vectors[row, sum(map(ord, word)) % 64] += 1.0
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def records():
    return {
        uid: ExtractionRecord(uid, "m", "d", old_nodes.OK, ENTITIES.get(uid, ()), (), None, 0, 0)
        for uid, *_ in UNITS
    }


def split(tmp_path: Path) -> Path:
    directory = tmp_path / "dev"
    directory.mkdir(exist_ok=True)
    units = [
        json.dumps({"unit_id": u, "paper_id": p, "title": t, "body": b, "tokens": 9})
        for u, p, t, b in UNITS
    ]
    titles = {p: t for _, p, t, _ in UNITS}
    questions = [
        json.dumps({"qid": q, "paper_id": p, "question": s, "pooled_query": f"{titles[p]}. {s}"})
        for q, p, s in QUESTIONS
    ]
    (directory / "units.jsonl").write_text("\n".join(units) + "\n", "utf-8")
    (directory / "questions.jsonl").write_text("\n".join(questions) + "\n", "utf-8")
    return directory  # no gold.json: ranking must not need it


def test_node_index_normalizes_and_keeps_unit_order():
    index = old_nodes.build_node_index(records(), [u for u, *_ in UNITS])
    assert index.unit_ids == ("a1", "a2", "a3", "b1", "b2")
    # "Penn Treebank" and "penn treebank" are one node; "The Baseline" drops its article.
    assert index.incidence.shape == (5, 3)
    assert index.incidence[0].nnz == 1 and index.incidence[1].nnz == 2
    with pytest.raises(ValueError):
        old_nodes.build_node_index(records(), ["a1", "zz"])


def test_local_extraction_record_dedupes_and_bounds():
    from edge_rag import local_extraction

    ok = local_extraction.record_from_forms(
        "u", [" X ", "X", "Y"], model="m", configuration_digest="d"
    )
    assert ok.status == old_nodes.OK and ok.entities == ("X", "Y")
    long = local_extraction.record_from_forms("u", ["x" * 121], model="m", configuration_digest="d")
    assert long.status == old_nodes.FAILED and long.failure == "bounds"


def test_run_writes_every_system_with_markers(tmp_path):
    out = tmp_path / "rankings"
    summary = exam_laptop.run(split(tmp_path), out, records(), encode, say=lambda _: None)
    assert summary == {"units": 5, "questions": 3, "seconds": summary["seconds"]}
    for setting, names in (("pooled", exam_laptop.POOLED), ("within", exam_laptop.WITHIN)):
        for name in names:
            assert exam_laptop.is_complete(out / setting, name, 3)
            ranked = scoring.read_rankings(out / setting / f"{name}.jsonl.gz")
            assert list(ranked) == ["q1", "q2", "q3"]
    within = scoring.read_rankings(out / "within" / "bm25.jsonl.gz")
    assert set(within["q1"]) <= {"a1", "a2", "a3"} and set(within["q2"]) <= {"b1", "b2"}
    pooled = scoring.read_rankings(out / "pooled" / "bm25.jsonl.gz")
    # D8: the same question ranks its own paper first once the title is in the query.
    assert pooled["q1"][0].startswith("a") and pooled["q2"][0].startswith("b")
    hop = scoring.read_rankings(out / "pooled" / "hop.jsonl.gz")
    assert all(len(v) <= 5 for v in hop.values())


def test_rerun_is_identical_and_tampering_breaks_completion(tmp_path):
    out = tmp_path / "rankings"
    exam_laptop.run(split(tmp_path), out, records(), encode, say=lambda _: None)
    exam_laptop.run(split(tmp_path), out, records(), encode, say=lambda _: None)
    path = out / "pooled" / "dense.jsonl.gz"
    path.write_bytes(scoring.ranking_bytes([("q1", ["a1"])]))
    assert not exam_laptop.is_complete(out / "pooled", "dense", 3)
    assert not exam_laptop.is_complete(out / "pooled", "g-l", 3)


def test_write_system_refuses_a_missing_question(tmp_path):
    with pytest.raises(ArtifactError):
        exam_laptop.write_system(tmp_path, "dense", {"q1": ["a1"]}, ["q1", "q2"])
    assert not (tmp_path / "dense.complete.json").exists()


def test_rrf4_lists_fuses_in_terrain_order():
    dense = {"q": ["a", "b"]}
    bm25 = {"q": ["b", "c"]}
    gl = {"q": ["b", "a"]}
    hop = {"q": ["d"]}
    assert exam_laptop.rrf4_lists(dense, bm25, gl, hop)["q"] == [
        "b",
        "a",
        "d",
        "c",
    ]  # hop rank 1 > BM25 rank 2
    with pytest.raises(ArtifactError):
        exam_laptop.rrf4_lists(dense, bm25, {}, hop)


def test_run_without_records_skips_only_the_entity_systems(tmp_path):
    out = tmp_path / "rankings"
    exam_laptop.run(split(tmp_path), out, None, encode, say=lambda _: None)
    for name in exam_laptop.POOLED:
        written = exam_laptop.is_complete(out / "pooled", name, 3)
        assert written == (name not in exam_laptop.ENTITY), name
    for name in exam_laptop.WITHIN:
        assert exam_laptop.is_complete(out / "within", name, 3)


def test_entrant_hop_uses_first_entity_seed_and_p14_keeps_p1(tmp_path, monkeypatch):
    seeds = []
    rarity = exam_laptop.Hops._rarity

    def spy(self, first, seed="p1"):
        seeds.append(seed)
        return rarity(self, first, seed)

    monkeypatch.setattr(exam_laptop.Hops, "_rarity", spy)
    exam_laptop.run(split(tmp_path), tmp_path / "r", records(), encode, say=lambda _: None)
    # per question: the entrant hop (spec pre-download change 3), then p14's relevance hop
    assert seeds == ["first-entity", "p1"] * len(QUESTIONS)
