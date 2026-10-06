"""G-A1 on the HotpotQA 1,000 (Phase 06 increment 5): qid restriction of the unchanged driver,
pace gate, and conversion of the pod's evidence lists, on hand-built examples."""

import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from edge_rag import config, ga1_hotpot
from edge_rag.artifacts import ArtifactError
from edge_rag.pod import common, ga1_pace, ga1_subset, searchr1

QIDS = [f"q{i:04d}" for i in range(1000)]


def qid_file(path: Path, qids: list[str] = QIDS) -> str:
    body = "".join(f"{q}\n" for q in qids).encode("utf-8")
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def test_qid_list_is_refused_unless_its_digest_and_size_match(tmp_path):
    digest = qid_file(tmp_path / "q.txt")
    assert ga1_subset.read_qids(tmp_path / "q.txt", digest) == QIDS
    with pytest.raises(ArtifactError, match="sha256"):
        ga1_subset.read_qids(tmp_path / "q.txt", "0" * 64)
    short = qid_file(tmp_path / "s.txt", QIDS[:999])
    with pytest.raises(ArtifactError, match="999 qids"):
        ga1_subset.read_qids(tmp_path / "s.txt", short)


def test_restriction_runs_the_original_and_keeps_question_file_order():
    calls = []
    questions = [SimpleNamespace(qid=q) for q in ("c", "a", "x", "b")]

    def original(*args):
        calls.append(args)
        return questions, {"a": "1"}

    kept, answers = ga1_subset.restricted(original, ["b", "a", "c"])("old", "checks", "spec")
    assert [q.qid for q in kept] == ["c", "a", "b"]
    assert answers == {"a": "1"} and calls == [("old", "checks", "spec")]
    with pytest.raises(ArtifactError, match="1 listed qids"):
        ga1_subset.restricted(original, ["a", "z"])()


def test_wrapper_calls_the_unchanged_driver_with_the_restriction(tmp_path, monkeypatch):
    digest = qid_file(tmp_path / "q.txt")
    seen = {}

    def fake_main():
        seen["argv"] = sys.argv[1:]
        seen["util"] = searchr1.GPU_MEMORY_UTILIZATION
        seen["patched"] = common.set_questions.__qualname__

    monkeypatch.setattr(searchr1, "main", fake_main)
    monkeypatch.setattr(searchr1, "GPU_MEMORY_UTILIZATION", 0.80)
    monkeypatch.setattr(common, "set_questions", common.set_questions)
    monkeypatch.setattr(sys, "argv", ["x"])
    monkeypatch.setattr(searchr1, "output_dir", lambda s, t: tmp_path / s / t)
    ga1_subset.main(
        [
            "--qids",
            str(tmp_path / "q.txt"),
            "--qids-sha256",
            digest,
            "--gpu-memory-utilization",
            "0.45",
        ]
    )
    assert seen == {
        "argv": ["--set", "hotpotqa-dev"],
        "util": 0.45,
        "patched": "restricted.<locals>.set_questions",
    }
    record = json.loads((tmp_path / "hotpotqa-dev" / "g-a1" / "subset.json").read_text("utf-8"))
    assert record["qids"] == 1000 and record["gpu_memory_utilization"] == 0.45
    assert record["driver_gpu_memory_utilization"] == 0.80
    assert set(record["driver_sha256"]) == {"searchr1.py", "common.py"}


def test_encode_pace_projects_only_after_ten_minutes():
    assert ga1_pace.encode_remaining([(0.0, 0), (300.0, 500_000)]) is None
    left = ga1_pace.encode_remaining([(0.0, 233_329), (1000.0, 1_233_329)])
    assert left == pytest.approx(4000.0)  # 4,000,000 units left at 1,000 units/s


def test_turn_projection_is_mean_turn_times_turns_left():
    lines = [
        "noise",
        "[TURN] 1 active 1000 searches 990 generate_s 50.0 retrieve_s 50.0 elapsed_s 100.0",
        "[TURN] 2 active 900 searches 800 generate_s 40.0 retrieve_s 40.0 elapsed_s 180.0",
    ]
    assert ga1_pace.turns_remaining(lines) == pytest.approx(90.0 * 7)
    assert ga1_pace.turns_remaining(["nothing"]) is None


def test_pace_gate_stops_past_the_cut(tmp_path, capsys):
    pace = tmp_path / "pace"
    pace.write_text("0 233329\n1000 1233329\n", "utf-8")
    base = [
        "--stage",
        "g-l",
        "--progress",
        str(tmp_path / "none"),
        "--pace",
        str(pace),
        "--rate",
        "1.59",
    ]
    # 1,800 s spent + 4,000 s encode + the recipe's rest (4,456 s) = 10,256 s, 4.53 USD
    assert ga1_pace.main([*base, "--spent", "1800", "--cut", "6.9"]) == 0
    assert "projected_usd 4.53" in capsys.readouterr().out
    assert ga1_pace.main([*base, "--spent", "1800", "--cut", "4.0"]) == 1
    other = [
        "--stage",
        "sync",
        "--progress",
        str(tmp_path / "none"),
        "--pace",
        str(pace),
        "--rate",
        "1.59",
    ]
    assert ga1_pace.main([*other, "--spent", "15624", "--cut", "6.9"]) == 1  # 6.90 USD spent


def test_evidence_lists_are_checked():
    rows = [{"qid": "a", "ranked": ["u1", "u2"]}, {"qid": "b", "ranked": []}]
    assert ga1_hotpot.evidence(rows, ["a", "b"], {"u1", "u2"}) == {"a": ["u1", "u2"], "b": []}
    with pytest.raises(ArtifactError, match="twice in its evidence"):
        ga1_hotpot.evidence([{"qid": "a", "ranked": ["u1", "u1"]}], ["a"], {"u1"})
    with pytest.raises(ArtifactError, match="not corpus units"):
        ga1_hotpot.evidence([{"qid": "a", "ranked": ["u9"]}], ["a"], {"u1"})
    with pytest.raises(ArtifactError, match="more than the driver"):
        ga1_hotpot.evidence(
            [{"qid": "a", "ranked": [f"u{i}" for i in range(25)]}],
            ["a"],
            {f"u{i}" for i in range(25)},
        )
    with pytest.raises(ArtifactError, match="differ from the qid list"):
        ga1_hotpot.evidence(rows, ["a"], {"u1", "u2"})


def write_pod(pod: Path, qids: list[str], digest: str, *, diff: str = "") -> None:
    pod.mkdir()
    ranking = common.gz_jsonl(
        {"qid": q, "ranked": ["u1", "u2"] if i % 2 else ["u2"]} for i, q in enumerate(qids)
    )
    (pod / "g-a1.jsonl.gz").write_bytes(ranking)
    sha = hashlib.sha256(ranking).hexdigest()
    driver = {
        "system": "g-a1",
        "set": "hotpotqa-dev",
        "engine": "vllm",
        "model": searchr1.MODEL,
        "revision": searchr1.REVISION,
        "settings": {"topk": 3},
        "status_counts": {"answered": len(qids)},
        "searches_total": 3,
        "tokens_generated": 9,
        "answer_em": {"graded": 0},
        "seconds": {"run": 50.0},
        "seconds_per_question": 0.05,
        "gpu": "A100",
        "cost_per_hr_usd": 1.59,
        "git_commit": "abc",
        "outputs": {"g-a1.jsonl.gz": sha, "trace.jsonl.gz": "t"},
    }
    files = {
        "g-a1.manifest.json": json.dumps(driver),
        "subset.json": json.dumps(
            {"qids_sha256": digest, "gpu_memory_utilization": 0.45, "driver_sha256": {}}
        ),
        "c3-g-a1.diff": diff,
        "g-l.manifest.json": json.dumps(
            {
                **ga1_hotpot.D17,
                "seconds": {"encode": 3877.0, "index": 685.0, "search": 9.0},
                "git_commit": "abc",
            }
        ),
        "index.sha256": "x  ./metadata.json\n",
    }
    for name, text in files.items():
        (pod / name).write_text(text, "utf-8")
    sums = {
        n: hashlib.sha256((pod / n).read_bytes()).hexdigest() for n in [*files, "g-a1.jsonl.gz"]
    }
    (pod / "sha256sums.txt").write_text(
        "".join(f"{d}  {n}\n" for n, d in sorted(sums.items())), "utf-8"
    )


def test_rank_writes_the_evidence_lists_with_a_manifest(tmp_path, monkeypatch):
    qids = QIDS
    digest = qid_file(tmp_path / ga1_hotpot.QIDS_FILE)
    monkeypatch.setattr(ga1_hotpot, "QIDS_SHA256", digest)
    write_pod(tmp_path / "pod", qids, digest)
    body = ga1_hotpot.run_rank(
        tmp_path, tmp_path / "pod", tmp_path / "rankings", unit_ids={"u1", "u2"}, say=lambda _: None
    )
    assert body["questions"] == 1000 and body["evidence_units"] == {"mean": 1.5, "min": 1, "max": 2}
    assert body["offline_seconds"] == 3877.0 + 685.0 and body["online_seconds"] == 50.0
    assert body["settings"]["gpu_memory_utilization"] == 0.45
    out = tmp_path / "rankings" / "hotpotqa-dev" / "g-a1.jsonl.gz"
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    assert first == {"qid": "q0000", "ranked": ["u2"]}


def test_rank_refuses_a_driver_diff_or_a_changed_file(tmp_path, monkeypatch):
    digest = qid_file(tmp_path / ga1_hotpot.QIDS_FILE)
    monkeypatch.setattr(ga1_hotpot, "QIDS_SHA256", digest)
    write_pod(tmp_path / "pod", QIDS, digest, diff="--- a/searchr1.py\n")
    with pytest.raises(ArtifactError, match="C3"):
        ga1_hotpot.check_pod(tmp_path / "pod")
    (tmp_path / "pod" / "subset.json").write_text("{}", "utf-8")
    with pytest.raises(ArtifactError, match="sha256 differs"):
        ga1_hotpot.check_pod(tmp_path / "pod")


@pytest.mark.skipif(
    not (config.old_data_root() / "phase9" / "corpus.jsonl.gz").exists(), reason="old data absent"
)
def test_export_refuses_qids_outside_the_question_file(tmp_path, monkeypatch):
    digest = qid_file(tmp_path / "list.txt")  # made-up qids, absent from the question file
    monkeypatch.setattr(ga1_hotpot, "QIDS_SHA256", digest)
    with pytest.raises(ArtifactError, match="not in the question file"):
        ga1_hotpot.run_export(tmp_path / "out", tmp_path / "list.txt", say=lambda _: None)
