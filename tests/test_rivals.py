"""Phase 06 laptop parts: the subsample rule (C2), G-R2 ranking order and manifest (C4), and
the C3 comparison of the pod check."""

import gzip
import hashlib
import json
import random

import numpy as np
import pytest

from edge_rag import config, rivals
from edge_rag.artifacts import ArtifactError, sha256_file
from edge_rag.pod.qwen_rerank import compare

SUBSAMPLE_SHA256 = "6cebd41ff9e5e9f02e27ebb019620de0e37b77794fc6a783a2ff0d80fb3a62bd"


def test_subsample_is_the_frozen_rule_over_sorted_qids():
    qids = [f"q{i:05d}" for i in range(3000)]
    shuffled = qids[::-1]
    chosen = rivals.subsample(shuffled)
    assert chosen == random.Random(20261002).sample(sorted(qids), 1000)  # noqa: S311
    assert len(set(chosen)) == 1000
    assert rivals.subsample_bytes(["a", "b"]) == b"a\nb\n"


def test_subsample_file_recomputes_from_the_rule():
    path = config.PHASE06_SUBSAMPLE_DIR / rivals.SUBSAMPLE_FILE
    if not path.exists() or not config.old_data_root().exists():
        pytest.skip("the subsample file or the old data root is not on this machine")
    from edge_rag import scoring
    from edge_rag.pod.common import open_set

    spec, old, checks = open_set("hotpotqa-dev")
    qids = [q.qid for q in scoring.questions_of(old, checks, spec)]
    body = rivals.subsample_bytes(rivals.subsample(qids))
    assert hashlib.sha256(body).hexdigest() == SUBSAMPLE_SHA256 == sha256_file(path)


def test_rank_orders_by_score_then_gl_rank():
    tops = {"q1": ["a", "b", "c", "d"], "q2": ["x", "y"]}
    scored = [
        {"qid": "q1", "unit_ids": ["a", "b", "c", "d"], "scores": [-3.0, -0.5, -0.5, -0.1]},
        {"qid": "q2", "unit_ids": ["x", "y"], "scores": [-2.0, -2.0]},
    ]
    # b and c tie: G-L rank keeps b first; x and y tie: G-L order kept.
    assert rivals.rank(tops, scored) == {"q1": ["d", "b", "c", "a"], "q2": ["x", "y"]}
    with pytest.raises(ArtifactError, match="not G-L's top"):
        rivals.rank(tops, [{**scored[0], "unit_ids": ["b", "a", "c", "d"]}, scored[1]])
    with pytest.raises(ArtifactError, match="no scores"):
        rivals.rank(tops, scored[:1])
    with pytest.raises(ArtifactError, match="scored twice"):
        rivals.rank(tops, [scored[0], scored[0], scored[1]])


def _gz(path, rows):
    path.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")))
    return sha256_file(path)


def test_run_rank_writes_rankings_and_manifest(tmp_path, monkeypatch):
    tops = {"q1": ["a", "b"], "q2": ["c", "d"]}
    monkeypatch.setattr(rivals, "gl_tops", lambda _set: tops)
    monkeypatch.setitem(config.GL_SHA256, "toy", "e" * 64)
    pairs_dir, scores_dir, rankings_dir = (tmp_path / n for n in ("pairs", "scores", "rankings"))
    pairs_dir.mkdir()
    scores_dir.mkdir()
    pins = {"toy.jsonl.gz": "f" * 64}
    (pairs_dir / "toy.manifest.json").write_text(json.dumps({"outputs_sha256": pins}))
    rows = [
        {"qid": "q2", "unit_ids": ["c", "d"], "scores": [-1.0, -0.2]},
        {"qid": "q1", "unit_ids": ["a", "b"], "scores": [-0.3, -0.3]},
    ]
    pod = {
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "revision": "r",
        "settings": {"dtype": "bfloat16"},
        "gpu": "RTX 4090",
        "cost_per_hr_usd": 0.74,
        "git_commit": "c",
        "git_src_changes": [],
        "inputs_sha256": pins,
        "check": {"pass": True},
        "seconds": {"score": 4.0},
        "outputs": {"toy.jsonl.gz": _gz(scores_dir / "toy.jsonl.gz", rows)},
    }
    (scores_dir / "toy.manifest.json").write_text(json.dumps(pod))
    body = rivals.run_rank(
        "toy",
        scores_dir=scores_dir,
        pairs_dir=pairs_dir,
        rankings_dir=rankings_dir,
        say=lambda _line: None,
    )
    with gzip.open(rankings_dir / "toy" / "g-r2.jsonl.gz", "rt", encoding="utf-8") as handle:
        written = [json.loads(line) for line in handle]
    assert written == [{"qid": "q2", "ranked": ["d", "c"]}, {"qid": "q1", "ranked": ["a", "b"]}]
    manifest = json.loads((rankings_dir / "toy" / "g-r2.manifest.json").read_text("utf-8"))
    assert manifest == body
    assert manifest["outputs_sha256"]["g-r2.jsonl.gz"] == sha256_file(
        rankings_dir / "toy" / "g-r2.jsonl.gz"
    )
    assert manifest["pod"]["cost_per_hr_usd"] == 0.74
    assert manifest["online_seconds_per_question"] == 2.0
    assert manifest["revision"] == "r"

    (scores_dir / "toy.manifest.json").write_text(json.dumps({**pod, "check": {"pass": False}}))
    with pytest.raises(ArtifactError, match="did not pass"):
        rivals.run_rank(
            "toy", scores_dir=scores_dir, pairs_dir=pairs_dir, rankings_dir=tmp_path, say=print
        )
    other = {**pod, "inputs_sha256": {"toy.jsonl.gz": "0" * 64}}
    (scores_dir / "toy.manifest.json").write_text(json.dumps(other))
    with pytest.raises(ArtifactError, match="other pairs files"):
        rivals.run_rank(
            "toy", scores_dir=scores_dir, pairs_dir=pairs_dir, rankings_dir=tmp_path, say=print
        )


def test_c3_compare_works_on_p_yes():
    reference = np.asarray([0.5, 0.01, 0.999])
    batched = np.log(reference + np.asarray([0.0005, 0.0, -0.0002])).astype(np.float32)
    same = compare(batched, batched.copy(), reference)
    assert same["pass"] and same["determinism_max_abs_diff"] == 0.0
    assert same["fidelity_max_abs_diff"] == pytest.approx(5e-4, abs=1e-6)
    shifted = batched.copy()
    shifted[0] = np.log(0.503)
    report = compare(batched, shifted, reference)
    assert report["fidelity_pass"] and not report["determinism_pass"] and not report["pass"]
