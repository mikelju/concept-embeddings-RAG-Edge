"""Phase 07 increment 5c: the QASPER pod adapters' input and output shapes, no model."""

import json
from pathlib import Path

import pytest

from edge_rag.artifacts import ArtifactError
from edge_rag.pod import common
from edge_rag.pod import exam_qasper as eq


def split(tmp_path: Path) -> Path:
    units = [
        {"unit_id": "u1", "paper_id": "p1", "title": "Paper one", "body": "alpha"},
        {"unit_id": "u2", "paper_id": "p1", "title": "Paper one", "body": "beta"},
        {"unit_id": "u3", "paper_id": "p2", "title": "Paper two", "body": "gamma"},
    ]
    questions = [
        {"qid": "q1", "paper_id": "p1", "question": "what?", "pooled_query": "Paper one. what?"},
        {"qid": "q2", "paper_id": "p2", "question": "why?", "pooled_query": "Paper two. why?"},
    ]
    for name, rows in (("units", units), ("questions", questions)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_readers_give_the_driver_shapes(tmp_path: Path) -> None:
    d = split(tmp_path)
    units = eq.read_units(d)
    assert units[0] == ("p1", common.Unit("u1", "Paper one", "alpha"))
    assert units[0][1].text == "Paper one. alpha"
    assert [p for p, _ in eq.read_units(d, 1)] == ["p1"]
    driver = eq.as_driver_questions(eq.read_questions(d))
    assert [(q.qid, q.question, q.gold_unit_ids) for q in driver] == [
        ("q1", "Paper one. what?", ()),
        ("q2", "Paper two. why?", ()),
    ]


def test_pooled_rerank_uses_d8_query_and_keeps_first_stage_ties(tmp_path: Path) -> None:
    d = split(tmp_path)
    questions = eq.read_questions(d)
    texts = {u.unit_id: u.text for _p, u in eq.read_units(d)}
    tops = {"q1": ["u3", "u1", "u2"], "q2": ["u2", "u3"]}
    seen: list[tuple[str, str]] = []

    def score(pairs):
        seen.extend(pairs)
        return [1.0 if t.endswith("beta") else 0.0 for _q, t in pairs]

    out = eq.rerank_lists(eq.pooled_jobs(questions, tops), texts, score, shard=1)
    assert out == {"q1": ["u2", "u3", "u1"], "q2": ["u2", "u3"]}
    assert seen[0] == ("Paper one. what?", "Paper two. gamma")
    with pytest.raises(ArtifactError):
        eq.pooled_jobs(questions, {"q1": ["u1"]})


def test_within_jobs_stay_in_own_paper_with_question_alone(tmp_path: Path) -> None:
    d = split(tmp_path)
    jobs = eq.within_jobs(eq.read_questions(d), eq.read_units(d))
    assert jobs == [("q1", "what?", ["u1", "u2"]), ("q2", "why?", ["u3"])]


def test_fused_first_stage_rrf3_and_rrf4() -> None:
    lists = {
        "dense": {"q": ["a", "b"]},
        "bm25": {"q": ["b", "a"]},
        "g-l": {"q": ["b", "c"]},
        "hop": {"q": ["c", "a"]},
    }
    assert eq.fused_first_stage("rrf3", lists)["q"] == ["b", "a", "c"]
    from edge_rag.pool import rrf4

    expected = rrf4([lists["dense"]["q"], lists["bm25"]["q"], lists["g-l"]["q"]], lists["hop"]["q"])
    assert eq.fused_first_stage("rrf4", lists)["q"] == expected
    with pytest.raises(ArtifactError):
        eq.fused_first_stage("rrf3", {**lists, "bm25": {"x": ["a"]}})


def test_rerank_command_writes_each_system(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "s").mkdir()
    d = split(tmp_path / "s")
    first = tmp_path / "g-l.jsonl.gz"
    common.write_rankings(first, [("q1", ["u2", "u1"]), ("q2", ["u3"])])
    monkeypatch.setattr(eq, "scorer", lambda qwen: (lambda p: [0.0] * len(p), {"model": "m"}))
    monkeypatch.setattr(common, "gpu_name", lambda: "none")
    out = tmp_path / "out"
    eq.main(
        [
            "rerank",
            str(d),
            str(out),
            "--first",
            f"g-l={first}",
            "--systems",
            "g-r",
            "within-j-strong",
        ]
    )
    assert common.read_rankings(out / "g-r.jsonl.gz") == [("q1", ["u2", "u1"]), ("q2", ["u3"])]
    assert common.read_rankings(out / "within-j-strong.jsonl.gz")[0] == ("q1", ["u1", "u2"])
    assert json.loads((out / "g-r.manifest.json").read_text())["pairs"] == 3


def test_patch_drivers_points_searchr1_at_qasper(tmp_path: Path, monkeypatch) -> None:
    d = split(tmp_path)
    for name in ("SETS", "PHASE_DIR", "open_set", "units_by_id", "set_questions"):
        target = common.config if name == "SETS" else common
        monkeypatch.setattr(target, name, getattr(target, name))
    from edge_rag.pod import colbert

    monkeypatch.setattr(colbert, "open_index", colbert.open_index)
    eq.patch_drivers(d, tmp_path / "index", tmp_path / "out")
    questions, answers = common.set_questions(None, None, None)
    assert [q.question for q in questions] == ["Paper one. what?", "Paper two. why?"]
    assert answers == {"q1": "", "q2": ""}
    assert set(common.units_by_id(None, None, {"u3"})) == {"u3"}
    assert "qasper" in common.config.SETS
    assert tmp_path / "out" == common.PHASE_DIR
