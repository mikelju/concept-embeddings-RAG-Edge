"""Phase 06 G-A2 adapters: the upload bundle (export), the pod's passage groups, token counts
and bundle check, and the laptop's conversion to depth-100 rankings (spec C4)."""

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from edge_rag import ga2, scoring
from edge_rag.artifacts import ArtifactError
from edge_rag.corpus import Question
from edge_rag.pod import hipporag_run
from edge_rag.pod.common import Unit

UNITS = [
    Unit("u1", "A", "one."),
    Unit("u2", "B", "two."),
    Unit("u3", "A", "one."),  # same text as u1: one passage to HippoRAG
    Unit("u4", "C", "three."),
    Unit("u5", "D", "four."),
]
QUESTIONS = [Question("q1", "first?", ("u1",), "test"), Question("q2", "second?", ("u2",), "test")]


class _Checks:
    records = [{"file": "x", "sha256": "y"}]


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ga2, "open_set", lambda name: (None, None, _Checks()))
    monkeypatch.setattr(ga2, "iter_units", lambda old, spec: iter(UNITS))
    monkeypatch.setattr(ga2.scoring, "questions_of", lambda old, checks, spec: QUESTIONS)
    monkeypatch.setattr(ga2, "EXPECTED", {"units": 5, "questions": 2})
    out = tmp_path / "bundle"
    ga2.run_export(out, say=lambda _: None)
    return out


def _rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_export_writes_units_questions_manifest_and_sums(bundle: Path):
    units = _rows(bundle / ga2.UNITS_FILE)
    assert units[0] == {"unit_id": "u1", "text": "A. one."}
    assert [u["unit_id"] for u in units] == ["u1", "u2", "u3", "u4", "u5"]
    assert _rows(bundle / ga2.QUESTIONS_FILE) == [
        {"qid": "q1", "question": "first?"},
        {"qid": "q2", "question": "second?"},
    ]
    manifest = json.loads((bundle / ga2.BUNDLE_MANIFEST).read_text("utf-8"))
    assert manifest["units"] == 5 and manifest["questions"] == 2
    assert manifest["duplicate_texts"] == {"texts": 1, "units": 2}
    for name, digest in manifest["outputs_sha256"].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
    # the pod reads the same digests from the sha256sum file
    assert hipporag_run.check_bundle(bundle) == manifest["outputs_sha256"]


def test_export_refuses_other_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ga2, "open_set", lambda name: (None, None, _Checks()))
    monkeypatch.setattr(ga2, "iter_units", lambda old, spec: iter(UNITS[:4]))
    monkeypatch.setattr(ga2.scoring, "questions_of", lambda old, checks, spec: QUESTIONS)
    monkeypatch.setattr(ga2, "EXPECTED", {"units": 5, "questions": 2})
    with pytest.raises(ArtifactError):
        ga2.run_export(tmp_path, say=lambda _: None)


def test_pod_bundle_check_catches_a_changed_file(bundle: Path):
    (bundle / ga2.QUESTIONS_FILE).write_bytes(b"other")
    with pytest.raises(SystemExit):
        hipporag_run.check_bundle(bundle)


def test_passage_groups_and_rows_keep_corpus_and_hipporag_order():
    units = _units_as_rows()
    groups = hipporag_run.passage_groups(units)
    assert groups["A. one."] == ["u1", "u3"]
    row = hipporag_run.to_row("q1", ["C. three.", "A. one."], [0.9, 0.5], groups)
    assert row == {"qid": "q1", "passages": [["u4"], ["u1", "u3"]], "scores": [0.9, 0.5]}
    with pytest.raises(SystemExit):
        hipporag_run.to_row("q1", ["not a unit"], [0.1], groups)


def _units_as_rows() -> list[dict]:
    return [{"unit_id": u.unit_id, "text": u.text} for u in UNITS]


def test_cache_tokens_split_offline_and_online(tmp_path: Path):
    cache = tmp_path / "cache.sqlite"
    with sqlite3.connect(cache) as conn:
        conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, message TEXT, metadata TEXT)")
        for i, (p, c) in enumerate([(10, 2), (20, 3), (5, 1)]):
            meta = json.dumps({"prompt_tokens": p, "completion_tokens": c})
            conn.execute("INSERT INTO cache VALUES (?, ?, ?)", (f"k{i}", "m", meta))
    assert hipporag_run.max_rowid(cache) == 3
    assert hipporag_run.cache_tokens(cache, 0, 2) == {
        "calls": 2,
        "prompt_tokens": 30,
        "completion_tokens": 5,
    }
    assert hipporag_run.cache_tokens(cache, 2) == {
        "calls": 1,
        "prompt_tokens": 5,
        "completion_tokens": 1,
    }
    assert hipporag_run.cache_tokens(tmp_path / "none.sqlite")["calls"] == 0


def test_flatten_expands_shared_passages_and_cuts_at_depth():
    assert ga2.flatten([["u4"], ["u1", "u3"], ["u2"]], depth=3) == ["u4", "u1", "u3"]
    with pytest.raises(ArtifactError):
        ga2.flatten([["u1"], ["u1"]])


def test_rank_checks_ids_coverage_and_depth():
    ids = {u.unit_id for u in UNITS}
    rows = [
        {"qid": "q1", "passages": [["u4"], ["u1", "u3"]]},
        {"qid": "q2", "passages": [["u2"], ["u5"], ["u4"]]},
    ]
    assert ga2.rank(["q1", "q2"], rows, ids, depth=3) == {
        "q1": ["u4", "u1", "u3"],
        "q2": ["u2", "u5", "u4"],
    }
    with pytest.raises(ArtifactError):  # too short
        ga2.rank(["q1"], [{"qid": "q1", "passages": [["u4"]]}], ids, depth=3)
    with pytest.raises(ArtifactError):  # not a corpus unit
        ga2.rank(["q1"], [{"qid": "q1", "passages": [["x"], ["u1", "u3"]]}], ids, depth=3)
    with pytest.raises(ArtifactError):  # a question missing
        ga2.rank(["q1", "q2"], rows[:1], ids, depth=3)
    with pytest.raises(ArtifactError):  # a question twice
        ga2.rank(["q1"], rows[:1] * 2, ids, depth=3)


def _pod(bundle: Path, tmp_path: Path, **override) -> Path:
    pod = tmp_path / "pod"
    pod.mkdir(parents=True)
    rows = [
        {"qid": "q2", "passages": [["u2"], ["u5"], ["u1", "u3"], ["u4"]], "scores": [4, 3, 2, 1]},
        {"qid": "q1", "passages": [["u4"], ["u1", "u3"], ["u2"], ["u5"]], "scores": [4, 3, 2, 1]},
    ]
    body = gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"), mtime=0)
    (pod / ga2.POD_RANKINGS).write_bytes(body)
    exported = json.loads((bundle / ga2.BUNDLE_MANIFEST).read_text("utf-8"))
    manifest = {
        "models": {"llm": {"name": "L"}, "embedding": {"name": "E"}},
        "hipporag": {"pinned_commit": "c", "head": "c", "status": ""},
        "settings": {"retrieval_top_k": 200},
        "seconds": {"index": 100.0, "retrieve": 20.0},
        "llm_tokens": {"offline": {"calls": 2}, "online": {"calls": 2}},
        "check": {"diff_sha256": hashlib.sha256(b"").hexdigest(), "filter_errors": 0},
        "gpu": "G",
        "cost_per_hr_usd": 0.5,
        "started_utc": "t",
        "inputs_sha256": exported["outputs_sha256"],
        "outputs": {ga2.POD_RANKINGS: hashlib.sha256(body).hexdigest()},
        **override,
    }
    (pod / ga2.POD_MANIFEST).write_text(json.dumps(manifest), "utf-8")
    return pod


def test_run_rank_writes_rankings_in_question_order_with_manifest(bundle: Path, tmp_path: Path):
    pod = _pod(bundle, tmp_path)
    body = ga2.run_rank(bundle, pod, tmp_path / "rankings", say=lambda _: None)
    path = tmp_path / "rankings" / ga2.SET / "g-a2.jsonl.gz"
    assert _rows(path) == [
        {"qid": "q1", "ranked": ["u4", "u1", "u3", "u2", "u5"]},
        {"qid": "q2", "ranked": ["u2", "u5", "u1", "u3", "u4"]},
    ]
    assert body["outputs_sha256"]["g-a2.jsonl.gz"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert body["offline_seconds"] == 100.0 and body["online_seconds_per_question"] == 10.0
    assert body["llm_tokens"]["online"]["calls"] == 2
    assert scoring.MANIFEST in {p.name for p in path.parent.iterdir()}


def test_run_rank_refuses_another_bundle_or_a_changed_file(bundle: Path, tmp_path: Path):
    pod = _pod(bundle, tmp_path, inputs_sha256={"units.jsonl.gz": "other"})
    with pytest.raises(ArtifactError):
        ga2.run_rank(bundle, pod, tmp_path / "rankings", say=lambda _: None)
    good = _pod(bundle, tmp_path / "second")
    (good / ga2.POD_RANKINGS).write_bytes(b"changed")
    with pytest.raises(ArtifactError):
        ga2.run_rank(bundle, good, tmp_path / "rankings", say=lambda _: None)
