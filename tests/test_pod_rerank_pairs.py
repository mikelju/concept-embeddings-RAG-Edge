"""Phase 04 pod mode `rerank --pairs` with a stub scorer: C4 check, file layout, write-once."""

import gzip
import json

import numpy as np
import pytest

from edge_rag.artifacts import ArtifactError, sha256_file
from edge_rag.pod import common, rerank


class Stub:
    """A scorer whose score is a fixed function of the pair text."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, pairs, **_):
        self.calls += 1
        return np.asarray([len(q) + len(t) / 100 for q, t in pairs], dtype=np.float32)


def stub_score(question, text):
    return float(np.float32(len(question) + len(text) / 100))


def write_gz(path, rows):
    path.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")))
    return sha256_file(path)


def make_inputs(tmp_path, *, offset=0.0):
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    rows = [
        {"qid": "q1", "question": "Who?", "unit_ids": ["a", "b"], "texts": ["A. x", "B. yy"]},
        {"qid": "q2", "question": "Where?", "unit_ids": ["c"], "texts": ["C. zzz"]},
    ]
    timing = [
        {
            "qid": "q1",
            "question": "Who?",
            "unit_ids": ["a", "b", "d"],
            "texts": ["A. x", "B. yy", "D. w"],
            "stored": [None, None, stub_score("Who?", "D. w") + offset],
        },
        {
            "qid": "q2",
            "question": "Where?",
            "unit_ids": ["c", "e"],
            "texts": ["C. zzz", "E. v"],
            "stored": [None, stub_score("Where?", "E. v") - offset],
        },
    ]
    outputs = {
        "toy.jsonl.gz": write_gz(pairs_dir / "toy.jsonl.gz", rows),
        "toy.timing.jsonl.gz": write_gz(pairs_dir / "toy.timing.jsonl.gz", timing),
    }
    (pairs_dir / "toy.manifest.json").write_text(json.dumps({"outputs_sha256": outputs}))
    return pairs_dir, tmp_path / "scores"


def run(pairs_dir, scores_dir, stub):
    return rerank.score_pairs(
        "toy", load=lambda: stub, pairs_dir=pairs_dir, scores_dir=scores_dir, say=lambda _: None
    )


def test_check_passes_then_scores_every_uncached_pair(tmp_path):
    pairs_dir, scores_dir = make_inputs(tmp_path)
    assert run(pairs_dir, scores_dir, Stub()) is True
    check = json.loads((scores_dir / "toy.check.json").read_text())
    assert check["pairs_compared"] == 2
    assert check["max_abs_diff"] == 0.0
    assert check["pass"] is True
    assert check["timing"]["questions"] == 2
    assert check["timing"]["pairs"] == 5
    rows = common.read_jsonl_gz(scores_dir / "toy.jsonl.gz")
    assert [(r["qid"], r["unit_ids"]) for r in rows] == [("q1", ["a", "b"]), ("q2", ["c"])]
    assert rows[0]["scores"] == [stub_score("Who?", "A. x"), stub_score("Who?", "B. yy")]
    assert rows[1]["scores"] == [stub_score("Where?", "C. zzz")]
    manifest = json.loads((scores_dir / "toy.manifest.json").read_text())
    assert manifest["pairs"] == 3
    assert manifest["questions"] == 2
    assert manifest["model"] == rerank.MODEL
    assert manifest["revision"] == rerank.REVISION
    assert manifest["settings"]["weights_sha256"] == rerank.WEIGHTS_SHA256
    assert manifest["check"]["pass"] is True
    assert manifest["outputs"]["toy.jsonl.gz"] == sha256_file(scores_dir / "toy.jsonl.gz")
    assert set(manifest["seconds"]) == {"read", "load_model", "score"}
    assert set(manifest["cgroup_memory"]) == {"memory.current", "memory.peak"}
    assert (
        manifest["inputs_sha256"]
        == json.loads((pairs_dir / "toy.manifest.json").read_text())["outputs_sha256"]
    )


def test_failed_check_stops_before_scoring(tmp_path):
    pairs_dir, scores_dir = make_inputs(tmp_path, offset=0.01)
    assert run(pairs_dir, scores_dir, Stub()) is False
    check = json.loads((scores_dir / "toy.check.json").read_text())
    assert check["pass"] is False
    assert check["max_abs_diff"] == pytest.approx(0.01, abs=1e-5)
    assert not (scores_dir / "toy.jsonl.gz").exists()
    assert not (scores_dir / "toy.manifest.json").exists()
    # A rerun reuses the failed check and still scores nothing.
    assert run(pairs_dir, scores_dir, Stub()) is False
    assert not (scores_dir / "toy.jsonl.gz").exists()


@pytest.mark.parametrize(
    "key", ["inputs_sha256", "revision", "weights_sha256", "tolerance", "git_commit"]
)
def test_check_is_recomputed_when_its_run_differs(tmp_path, key):
    pairs_dir, scores_dir = make_inputs(tmp_path, offset=0.01)
    assert run(pairs_dir, scores_dir, Stub()) is False
    check_path = scores_dir / "toy.check.json"
    stale = json.loads(check_path.read_text())
    current = stale[key]
    stale[key] = "other"
    check_path.write_text(json.dumps(stale))
    stub = Stub()
    assert run(pairs_dir, scores_dir, stub) is False
    assert stub.calls == 1
    assert json.loads(check_path.read_text())[key] == current
    # The same run reuses it without scoring.
    again = Stub()
    assert run(pairs_dir, scores_dir, again) is False
    assert again.calls == 0


def test_scores_are_written_once(tmp_path):
    pairs_dir, scores_dir = make_inputs(tmp_path)
    assert run(pairs_dir, scores_dir, Stub()) is True
    before = sha256_file(scores_dir / "toy.jsonl.gz")
    again = Stub()
    assert run(pairs_dir, scores_dir, again) is True
    assert again.calls == 0
    assert sha256_file(scores_dir / "toy.jsonl.gz") == before
    (scores_dir / "toy.manifest.json").unlink()
    with pytest.raises(ArtifactError, match="without its manifest"):
        run(pairs_dir, scores_dir, Stub())


def test_inputs_must_match_the_pinned_sha256(tmp_path):
    pairs_dir, scores_dir = make_inputs(tmp_path)
    write_gz(
        pairs_dir / "toy.jsonl.gz", [{"qid": "q9", "question": "?", "unit_ids": [], "texts": []}]
    )
    with pytest.raises(ArtifactError, match="pinned"):
        run(pairs_dir, scores_dir, Stub())
    assert not scores_dir.exists()


def test_cgroup_memory_reads_v2_files(tmp_path):
    (tmp_path / "memory.current").write_text("123\n")
    assert rerank.cgroup_memory(tmp_path) == {"memory.current": 123, "memory.peak": None}
