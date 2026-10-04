import gzip
import json

import pytest

from edge_rag.artifacts import ArtifactError, sha256_file
from edge_rag.judge import assemble, judged, pair_row, pod_scores, score_table, uncached, union
from edge_rag.pod.common import Unit


def test_cache_split_sends_only_unstored_pairs():
    stored = {("q1", "a"): 1.0, ("q1", "c"): 0.5, ("q2", "b"): 2.0}
    assert uncached("q1", ["a", "b", "c", "d"], stored) == ["b", "d"]
    # The same unit under another question is another pair.
    assert uncached("q2", ["a", "b"], stored) == ["a"]
    assert uncached("q1", ["a", "c"], stored) == []


def test_judge_order_breaks_ties_by_pool_rank():
    table = {("q", "x"): 1.0, ("q", "y"): 3.0, ("q", "z"): 1.0, ("q", "w"): 1.0}
    assert judged("q", ["z", "x", "y", "w"], table) == ["y", "z", "x", "w"]
    assert judged("q", ["x", "w", "z", "y"], table) == ["y", "x", "w", "z"]


def test_assembly_from_mixed_cached_and_new_scores():
    stored = {("q", "a"): 0.2, ("q", "b"): 0.9}
    fresh = {("q", "c"): 0.5}
    pool = ["a", "b", "c"]
    table = score_table((("q", u) for u in pool), stored, fresh)
    assert table == {("q", "a"): 0.2, ("q", "b"): 0.9, ("q", "c"): 0.5}
    assert judged("q", pool, table) == ["b", "c", "a"]
    with pytest.raises(ArtifactError, match="no score"):
        score_table([("q", "d")], stored, fresh)
    with pytest.raises(ArtifactError, match="stored score and a pod score"):
        score_table([("q", "a")], stored, {("q", "a"): 0.2})


def test_one_score_per_pair_shared_by_both_systems():
    rrf3, rrf4 = ["a", "b", "c"], ["c", "d", "a"]
    both = union(rrf3, rrf4)
    assert both == ["a", "b", "c", "d"]
    stored = {("q", "a"): 1.0}
    fresh = {("q", u): float(i) for i, u in enumerate(uncached("q", both, stored))}
    pairs = [("q", u) for u in rrf3 + rrf4]
    table = score_table(pairs, stored, fresh)
    assert len(table) == len(both) == len(set(pairs))
    assert judged("q", rrf3, table) == ["a", "c", "b"]
    assert judged("q", rrf4, table) == ["d", "c", "a"]


def test_pairs_file_text_is_the_corpus_title_and_sentences():
    sentences = ["First sentence.", "Second one."]
    units = {"u1": Unit("u1", "A Title", " ".join(sentences))}
    row = pair_row("q", "Who?", ["u1"], units)
    assert row == {
        "qid": "q",
        "question": "Who?",
        "unit_ids": ["u1"],
        "texts": [f"A Title. {' '.join(sentences)}"],
    }


def test_assemble_on_a_toy_set_with_both_pools():
    pools = {
        "rrf3": {"q1": ["a", "b", "c"], "q2": ["x", "y"]},
        "rrf4": {"q1": ["c", "d", "a"], "q2": ["y", "z"]},
    }
    stored = {("q1", "a"): 0.1, ("q1", "c"): 0.7, ("q2", "y"): 0.0}
    fresh = {("q1", "b"): 0.9, ("q1", "d"): 0.7, ("q2", "x"): 0.0, ("q2", "z"): 0.5}
    rankings, counts = assemble(pools, stored, fresh)
    assert rankings["rrf3"] == {"q1": ["b", "c", "a"], "q2": ["x", "y"]}
    # c and d tie at 0.7: RRF4's rank keeps c first.
    assert rankings["rrf4"] == {"q1": ["c", "d", "a"], "q2": ["z", "y"]}
    assert counts == {
        "questions": 2,
        "permutation_of_pool": {"rrf3": 2, "rrf4": 2},
        "pairs": 7,
        "pairs_with_one_score": 7,
        "from_cache": 3,
        "from_pod": 4,
    }
    with pytest.raises(ArtifactError, match="belong to no pool pair"):
        assemble(pools, stored, {**fresh, ("q2", "w"): 1.0})
    with pytest.raises(ArtifactError, match="no score"):
        assemble(pools, stored, {k: v for k, v in fresh.items() if k != ("q2", "z")})


def _write(path, body):
    path.write_bytes(body)
    return sha256_file(path)


def test_pod_scores_read_from_a_fabricated_scores_file(tmp_path):
    scores_dir, pairs_dir = tmp_path / "scores", tmp_path / "pairs"
    scores_dir.mkdir()
    pairs_dir.mkdir()
    pins = {"toy.jsonl.gz": "p" * 64, "toy.timing.jsonl.gz": "t" * 64}
    (pairs_dir / "toy.manifest.json").write_text(json.dumps({"outputs_sha256": pins}))
    rows = [{"qid": "q1", "unit_ids": ["b", "d"], "scores": [0.9, 0.7]}]
    body = gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"))
    digest = _write(scores_dir / "toy.jsonl.gz", body)

    def manifest(**changes):
        entry = {
            "check": {"pass": True},
            "inputs_sha256": pins,
            "outputs": {"toy.jsonl.gz": digest},
        }
        entry.update(changes)
        (scores_dir / "toy.manifest.json").write_text(json.dumps(entry))

    manifest()
    fresh, sha = pod_scores("toy", scores_dir, pairs_dir)
    assert fresh == {("q1", "b"): 0.9, ("q1", "d"): 0.7}
    assert sha["toy.jsonl.gz"] == digest
    manifest(check={"pass": False})
    with pytest.raises(ArtifactError, match="C4"):
        pod_scores("toy", scores_dir, pairs_dir)
    manifest(inputs_sha256={**pins, "toy.jsonl.gz": "0" * 64})
    with pytest.raises(ArtifactError, match="other pairs"):
        pod_scores("toy", scores_dir, pairs_dir)
    manifest(outputs={"toy.jsonl.gz": "0" * 64})
    with pytest.raises(ArtifactError, match="recorded"):
        pod_scores("toy", scores_dir, pairs_dir)
