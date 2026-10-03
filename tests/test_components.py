import gzip
import hashlib
import json

import numpy as np

from edge_rag import components, config
from edge_rag.artifacts import Checks, OldData, digest_of
from edge_rag.corpus import Question, load_corpus, unit_id_for, unit_set_hash
from edge_rag.embeddings import QueryTable
from edge_rag.retrieval.dense_bm25 import BM25, Dense
from edge_rag.retrieval.fusion import fuse_lists


def test_body_is_the_text_after_the_title_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv(config.OLD_DATA_ROOT_ENV, str(tmp_path))
    units = [("St. Louis. Missouri", ["One. two.", "Three."]), ("Empty", [])]
    entries = [{"title": t, "sentences": s, "unit_id": unit_id_for(t, s)} for t, s in units]
    ids = [entry["unit_id"] for entry in entries]
    directory = tmp_path / "toy"
    directory.mkdir()
    with gzip.open(directory / "corpus.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(entry) + "\n" for entry in entries)
    manifest = {"corpus_file": "corpus.jsonl.gz", "ordered_unit_digest": digest_of(*ids)}
    (directory / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
    pins = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("corpus.json", "corpus.jsonl.gz")
    }
    corpus = load_corpus(
        OldData("toy", pins), Checks(), ordered_digest=digest_of(*ids), set_hash=unit_set_hash(ids)
    )
    assert corpus.titles == ["St. Louis. Missouri", "Empty"]
    assert corpus.body(0) == "One. two. Three."
    assert corpus.body(1) == ""
    assert corpus.texts[1] == "Empty. "


def _toy():
    ids = ["u1", "u2", "u3", "u4"]
    texts = ["red apple pie", "green apple tart", "blue river stone", "red river boat"]
    vectors = np.eye(4, dtype=np.float32)
    questions = [
        Question("q1", "red apple", ("u1",), "dev"),
        Question("q2", "river stone", ("u3",), "dev"),
    ]
    rows = np.array([[0.9, 0.5, 0.1, 0.2], [0.1, 0.2, 0.9, 0.3]], np.float32)
    table = QueryTable(rows, ["q1", "q2"])
    return ids, texts, vectors, questions, table


def test_both_equality_counts_compare_with_the_stored_lists():
    ids, texts, vectors, questions, table = _toy()
    dense, bm25 = Dense(vectors, ids), BM25(texts, ids)
    truth_a, truth_b = {}, {}
    for question in questions:
        first = dense.retrieve(table.vector(question.qid), config.DEPTH)
        second = bm25.retrieve(question.question, config.DEPTH)
        truth_a[question.qid] = [u for u, _ in first]
        fused = fuse_lists([first, second], config.WEIGHTS_P10B, top_k=config.DEPTH)
        truth_b[question.qid] = [u for u, _ in fused]
    rankings, seconds, equal = components.retrieve(
        questions, dense, bm25, table, truth_a, truth_b, say=lambda _: None
    )
    assert equal == {"dense_equals_p10-a": 2, "fusion_equals_p10-b": 2}
    assert rankings["dense"] == truth_a
    assert set(seconds) == {"dense", "bm25"}
    wrong_a = dict(truth_a, q2=list(reversed(truth_a["q2"])))
    wrong_b = dict(truth_b, q1=truth_b["q1"][:1])
    _, _, equal = components.retrieve(
        questions, dense, bm25, table, wrong_a, wrong_b, say=lambda _: None
    )
    assert equal == {"dense_equals_p10-a": 1, "fusion_equals_p10-b": 1}


def test_manifest_divides_seconds_per_question_and_pins_inputs():
    body = components.manifest(
        set_name="toy",
        questions=4,
        checks=[{"artifact": "a", "digest": "d"}],
        inputs_sha256={"p10-a.jsonl.gz": "x"},
        stage_seconds={"BM25 rebuild": 3.5, "retrieval": 1.0},
        retrieval_seconds={"dense": 2.0, "bm25": 1.0},
        encoding={"seconds_per_question": 0.04},
        equal={"dense_equals_p10-a": 4, "fusion_equals_p10-b": 3},
        outputs={"dense.jsonl.gz": "y"},
        peak_rss=12.5,
    )
    assert body["retrieval_seconds_per_question"] == {"dense": 0.5, "bm25": 0.25}
    assert body["offline_seconds"] == {"bm25_build": 3.5}
    assert body["equality"]["fusion_equals_p10-b"] == {"equal": 3, "of": 4}
    assert body["inputs_sha256"] == {"p10-a.jsonl.gz": "x"}
    assert body["peak_rss_mb"] == 12.5
    assert body["git_commit"]
