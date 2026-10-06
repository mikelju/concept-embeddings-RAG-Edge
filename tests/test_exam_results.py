"""Phase 07 exam rows (spec C6): the verdict rule on hand-built examples, no exam data."""

import pytest

from edge_rag import phase_results as pr

QIDS = [f"q{i:02d}" for i in range(40)]
# q00-q29 single-evidence, q30-q39 multi-evidence, q40 out of scope.
GOLD = {q: [["u-" + q]] for q in QIDS[:30]}
GOLD |= {q: [["a-" + q, "b-" + q]] for q in QIDS[30:]}
GOLD["q40"] = []


def rec(n_ones: int) -> dict[str, int]:
    return {q: int(i < n_ones) for i, q in enumerate(QIDS)}


def full() -> dict[str, dict[str, int] | None]:
    """Every line-up system run: j-rrf4 clearly beats every ghost and j-rrf3 in R and A."""
    return {
        "rrf4": rec(30), "j-rrf4": rec(36), "j-rrf3": rec(20), "p10-b": rec(12),
        "p14": rec(13), "G-L": rec(10), "G-R": rec(18), "G-R2": rec(20), "G-A1": rec(22),
    }  # fmt: skip


def table(pooled, upper=None):
    return pr.exam_table(pooled, {}, GOLD, {}, upper or {})


def by_class(t):
    return {bar["class"]: bar for bar in t["verdict"]}


def line(t, c):
    return next(x for x in pr.exam_verdict_lines(t) if x.startswith(f"- **Class {c}"))


def test_entrants_are_hard_coded():
    assert pr.EXAM_ENTRANTS == {"L": "rrf4", "R": "j-rrf4", "A": "j-rrf4"}
    assert list(pr.EXAM_GHOSTS) == ["G-L", "G-R", "G-R2", "G-A1"]
    assert pr.EXAM_QUESTIONS == 1_451


@pytest.mark.parametrize("context", ["p14", "p10-b", "j-rrf3"])
def test_context_row_beating_the_entrant_leaves_the_verdict(context):
    base = full()
    states = {c: b["state"] for c, b in by_class(table(base)).items()}
    beaten = base | {context: rec(40)}
    after = by_class(table(beaten))
    assert {c: b["state"] for c, b in after.items()} == states
    assert states == {"L": "win", "R": "win", "A": "win"}
    assert all(b["strongest_ghost"] != context for b in after.values())


def test_g_a1_not_run_gives_class_a_not_run():
    t = by_class(table(full() | {"G-A1": None}))
    assert t["A"]["state"] == pr.NOT_RUN and t["A"]["not_run"] == ["G-A1"]
    assert t["A"]["strongest_ghost"] is None  # no cheaper ghost stands in
    assert t["L"]["state"] == "win" and t["R"]["state"] == "win"


def test_bar_tie_resolved_after_fs_by_class_then_fixed_order():
    tied = full() | {"G-L": rec(20), "G-R": rec(20), "G-R2": rec(20), "G-A1": rec(20)}
    assert by_class(table(tied))["A"]["strongest_ghost"] == "G-L"
    tied |= {"G-L": rec(19)}
    assert by_class(table(tied))["A"]["strongest_ghost"] == "G-R"
    assert by_class(table(tied))["R"]["strongest_ghost"] == "G-R"
    tied |= {"G-R": rec(19)}
    assert by_class(table(tied))["A"]["strongest_ghost"] == "G-R2"
    tied |= {"G-R2": rec(19)}
    assert by_class(table(tied))["A"]["strongest_ghost"] == "G-A1"


def test_upper_reference_label_printed_without_changing_the_state():
    plain = table(full())
    labelled = table(full(), {"j-rrf4": "bge-reranker-v2-m3", "G-R": "bge-reranker-v2-m3"})
    assert [b["state"] for b in labelled["verdict"]] == [b["state"] for b in plain["verdict"]]
    assert "keeps its place on the exam in class R (upper reference: bge-reranker-v2-m3)" in (
        line(labelled, "R")
    )
    assert "upper reference" not in line(plain, "R")
    both = full() | {"G-R": rec(30), "G-R2": rec(10), "G-A1": rec(10)}
    labels = {"j-rrf4": "bge-reranker-v2-m3", "G-R": "bge-reranker-v2-m3"}
    assert "Upper reference on both sides." in line(table(both, labels), "R")
    assert table(both, labels)["rows"]["G-R"]["upper"] == "bge-reranker-v2-m3"


def test_fewer_than_1451_rankings_is_not_run_and_never_scored():
    ids = [f"t{i:04d}" for i in range(pr.EXAM_QUESTIONS)]
    gold = {q: [["u-" + q]] for q in ids}
    counts = {"u-" + q: 100 for q in ids}
    rankings = {q: ["u-" + q] for q in ids}
    assert sum(pr.exam_hits(rankings, counts, gold).values()) == pr.EXAM_QUESTIONS
    short = {q: rankings[q] for q in ids[:1_450]}
    assert pr.exam_hits(short, counts, gold) is None
    t = table(full() | {"G-R2": None})
    assert t["rows"]["G-R2"]["fs"] is None
    assert "| G-R2 | ghost | not run | not run | not run |" in pr.exam_markdown(t)


def test_in_scope_filter_and_several_annotators_at_scoring():
    ids = [f"t{i:04d}" for i in range(pr.EXAM_QUESTIONS)]
    rankings = {q: ["x"] for q in ids}
    gold = {q: [] for q in ids} | {ids[0]: [["x", "y"], ["x"]], ids[1]: [["y"]]}
    hits = pr.exam_hits(rankings, {"x": 10, "y": 10}, gold)
    assert hits == {ids[0]: 1, ids[1]: 0}


@pytest.mark.parametrize(
    ("missing", "classes"),
    [("rrf4", {"L"}), ("j-rrf4", {"R", "A"}), ("G-L", {"L", "R", "A"}), ("G-R", {"R", "A"})],
)
def test_entrant_or_needed_ghost_not_run_gives_the_class_not_run(missing, classes):
    t = by_class(table(full() | {missing: None}))
    assert {c for c, b in t.items() if b["state"] == pr.NOT_RUN} == classes


def test_control_state_printed_on_r_and_a_without_changing_them():
    ahead = full()
    behind = full() | {"j-rrf3": rec(40)}
    for c in ("R", "A"):
        assert by_class(table(ahead))[c]["state"] == by_class(table(behind))[c]["state"] == "win"
        assert by_class(table(ahead))[c]["control"]["state"] == "win"
        assert by_class(table(behind))[c]["control"]["state"] == "tie"  # 4 losses only
        assert "Against the literature control j-rrf3 (report only): tie" in line(table(behind), c)
    assert "control" not in by_class(table(ahead))["L"]
    assert "literature control" not in line(table(ahead), "L")
    lost = by_class(table(full() | {"j-rrf3": rec(40), "j-rrf4": rec(30)}))  # 10 losses
    assert lost["R"]["state"] == "win" and lost["R"]["control"]["state"] == "loss"


def test_breakdown_within_paper_and_page():
    t = pr.exam_table(full(), {"j-rrf4": rec(38), "bm25": rec(25)}, GOLD, {}, {})
    assert (t["n"], t["n_single"], t["n_multi"]) == (40, 30, 10)
    row = t["rows"]["j-rrf4"]
    assert (row["fs"], row["fs_single"], row["fs_multi"], row["fs_within"]) == (36, 30, 6, 38)
    assert t["rows"]["p14"]["role"] == "context" and t["rows"]["j-rrf3"]["role"] == "control"
    page = pr.exam_markdown(t)
    assert "| j-rrf4 | entrant | 36 | 30 | 6 | 38 |" in page
    assert f"| bm25 | {pr.WITHIN_ONLY} | - | - | - | 25 |" in page


def test_page_regenerates_from_the_json(tmp_path):
    t = table(full() | {"G-A1": None})
    pr.write_pair(t, pr.exam_markdown, tmp_path / "results.json", tmp_path / "results.md")
    pr.regenerate(pr.exam_markdown, tmp_path / "results.json", tmp_path / "results.md")
