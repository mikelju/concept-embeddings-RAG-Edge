"""The pod's shared pieces: corpus stream, ranking files, manifests, rerank order, C3 pairs."""

import dataclasses
import gzip
import hashlib
import json

import pytest

from edge_rag import config
from edge_rag.artifacts import ArtifactError, OldData, digest_of, sha256_file
from edge_rag.corpus import unit_id_for, unit_set_hash
from edge_rag.pod import common, rerank


def make_corpus(tmp_path, entries):
    directory = tmp_path / "old" / "toy"
    directory.mkdir(parents=True)
    lines = []
    ids = []
    for title, sentences in entries:
        unit_id = unit_id_for(title, sentences)
        ids.append(unit_id)
        lines.append(json.dumps({"unit_id": unit_id, "title": title, "sentences": sentences}))
    corpus = gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))
    (directory / "corpus.jsonl.gz").write_bytes(corpus)
    manifest = json.dumps(
        {"corpus_file": "corpus.jsonl.gz", "ordered_unit_digest": digest_of(*ids)}
    ).encode("utf-8")
    (directory / "corpus.json").write_bytes(manifest)
    pins = {
        "corpus.jsonl.gz": hashlib.sha256(corpus).hexdigest(),
        "corpus.json": hashlib.sha256(manifest).hexdigest(),
    }
    spec = dataclasses.replace(
        config.SETS["musique"],
        name="toy",
        directory="toy",
        ordered_unit_digest=digest_of(*ids),
        unit_set_hash=unit_set_hash(ids),
        file_sha256=pins,
    )
    return spec, OldData("toy", pins, root=tmp_path / "old"), ids


ENTRIES = [("Paris", ["Capital.", "Of France."]), ("Lyon", ["A city."]), ("Nice", ["Sea."])]


def test_stream_matches_the_harness_text_and_digests(tmp_path):
    spec, old, ids = make_corpus(tmp_path, ENTRIES)
    units = list(common.iter_units(old, spec))
    assert [u.unit_id for u in units] == ids
    assert units[0].text == "Paris. Capital. Of France."
    assert [u.unit_id for u in common.iter_units(old, spec, limit=2)] == ids[:2]
    picked = common.units_by_id(old, spec, {ids[2]})
    assert list(picked) == [ids[2]]
    with pytest.raises(ArtifactError):
        common.units_by_id(old, spec, {"missing"})


def test_stream_refuses_a_wrong_order(tmp_path):
    spec, old, ids = make_corpus(tmp_path, ENTRIES)
    wrong = dataclasses.replace(spec, ordered_unit_digest=digest_of(*reversed(ids)))
    with pytest.raises(ArtifactError):
        list(common.iter_units(old, wrong))


def test_rankings_round_trip_and_bytes_are_stable(tmp_path):
    rows = [("q1", ["a", "b"]), ("q2", ["c"])]
    first = common.write_rankings(tmp_path / "a" / "g-l.jsonl.gz", rows)
    second = common.write_rankings(tmp_path / "b" / "g-l.jsonl.gz", rows)
    assert first == second
    assert common.read_rankings(tmp_path / "a" / "g-l.jsonl.gz") == rows
    assert common.manifest_path(tmp_path / "g-l.jsonl.gz").name == "g-l.manifest.json"
    probe = common.rankings_path("musique", "g-l", "units1000")
    assert probe.parts[-4:] == ("probe", "units1000", "musique", "g-l.jsonl.gz")


def test_manifest_records_outputs_and_rate(tmp_path, monkeypatch):
    monkeypatch.setenv(common.COST_PER_HR_ENV, "0.74")
    out = tmp_path / "x.jsonl.gz"
    common.write_rankings(out, [("q", ["u"])])
    path = common.write_manifest(tmp_path / "x.manifest.json", {"system": "g-l"}, [out])
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["system"] == "g-l"
    assert body["cost_per_hr_usd"] == 0.74
    assert body["outputs"] == {"x.jsonl.gz": sha256_file(out)}
    assert {"git_commit", "gpu", "written_utc"} <= set(body)


def test_rerank_order_breaks_ties_by_first_stage_rank():
    assert rerank.reorder(["a", "b", "c", "d"], [0.1, 0.5, 0.5, 0.9]) == ["d", "b", "c", "a"]


def test_first_pairs_walks_rows_in_file_order():
    pairs = [
        {"qid": "1", "question": "Q1", "unit_ids": ["a", "b"], "texts": ["A", "B"]},
        {"qid": "2", "question": "Q2", "unit_ids": ["c", "d"], "texts": ["C", "D"]},
    ]
    scores = [
        {"qid": "1", "unit_ids": ["a", "b"], "scores": [1.0, 2.0]},
        {"qid": "2", "unit_ids": ["c", "d"], "scores": [3.0, 4.0]},
    ]
    picked = rerank.first_pairs(pairs, scores, 3)
    assert [(p["question"], p["text"], p["stored"]) for p in picked] == [
        ("Q1", "A", 1.0),
        ("Q1", "B", 2.0),
        ("Q2", "C", 3.0),
    ]
    with pytest.raises(ArtifactError):
        rerank.first_pairs(pairs, scores, 5)
    scores[1]["unit_ids"] = ["d", "c"]
    with pytest.raises(ArtifactError):
        rerank.first_pairs(pairs, scores, 4)


def test_read_pinned_refuses_other_bytes(tmp_path):
    path = tmp_path / "p.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"a": 1}\n{"a": 2}\n'))
    assert rerank.read_pinned(path, sha256_file(path), 1) == [{"a": 1}]
    with pytest.raises(ArtifactError):
        rerank.read_pinned(path, "0" * 64)


def test_spread_chunks_first_chunk_spans_corpus() -> None:
    from edge_rag.pod.colbert import spread_chunks

    chunks = list(spread_chunks(lambda: iter(range(10)), 3, 3))
    assert chunks[0] == [0, 3, 6, 9]
    assert sorted(u for c in chunks for u in c) == list(range(10))
    assert list(spread_chunks(lambda: iter(range(4)), 1, 10)) == [[0, 1, 2, 3]]
