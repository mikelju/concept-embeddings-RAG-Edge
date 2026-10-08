"""Phase 07 increment 2: the QASPER loader on a hand-built two-paper example."""

import json

from edge_rag import qasper

SHARED = "A paragraph both papers print word for word."


def _answer(evidence, unanswerable=False):
    return {"answer": {"unanswerable": unanswerable, "evidence": evidence}}


def _papers():
    return {
        "p2": {
            "title": "Paper Two",
            "full_text": [{"section_name": "Intro", "paragraphs": [SHARED, "Only in two."]}],
            "figures_and_tables": [],
            "qas": [],
        },
        "p1": {
            "title": "Paper  One",
            "full_text": [
                {"section_name": "Intro", "paragraphs": ["First  para.", "", "Second para."]},
                {"section_name": "Method", "paragraphs": ["First para."]},
            ],
            "figures_and_tables": [{"caption": "Table 1: Data.", "file": "t1.png"}],
            "qas": [
                {  # two annotators: one maps fully (two units), one cites a heading
                    "question_id": "q-multi",
                    "question": "What  data?",
                    "answers": [
                        _answer(["First para.", "FLOAT SELECTED: Table 1: Data."]),
                        _answer(["Intro", "Second para."]),
                    ],
                },
                {  # a string that is a unit of the other paper never maps
                    "question_id": "q-cross",
                    "question": "Shared?",
                    "answers": [_answer(["Only in two."])],
                },
                {  # the same string twice is one unit: single evidence
                    "question_id": "q-twice",
                    "question": "Twice?",
                    "answers": [_answer(["Second para.", "Second  para."])],
                },
                {
                    "question_id": "q-none",
                    "question": "Unanswerable?",
                    "answers": [_answer([], unanswerable=True)],
                },
            ],
        },
    }


def test_units_titles_and_the_float_prefix():
    split = qasper.load(_papers())
    one = [u for u in split.units if u.paper_id == "p1"]
    # empty paragraph dropped; "First para." repeated in Method is one unit (its first place)
    assert [u.body for u in one] == [
        "First para.",
        "Second para.",
        "FLOAT SELECTED: Table 1: Data.",
    ]
    assert one[0].title == "Paper One - Intro"
    assert one[0].text == "Paper One - Intro. First para."
    assert one[2].title == "Paper One"
    # the same text in two papers is two units
    ids = [u.unit_id for u in split.units]
    assert len(set(ids)) == len(ids) == 5


def test_gold_maps_inside_the_own_paper_only():
    split = qasper.load(_papers())
    by_body = {u.body: u.unit_id for u in split.units if u.paper_id == "p1"}
    first, table = by_body["First para."], by_body["FLOAT SELECTED: Table 1: Data."]
    assert split.gold["q-multi"] == [tuple(sorted((first, table)))]
    assert split.gold["q-cross"] == []
    assert split.gold["q-twice"] == [(by_body["Second para."],)]
    assert split.gold["q-none"] == []
    assert split.unmapped_sets == 2
    assert (split.evidence_strings, split.evidence_matched) == (7, 5)
    assert (split.float_strings, split.float_matched) == (1, 1)


def test_pooled_query_names_the_paper():
    split = qasper.load(_papers())
    question = next(q for q in split.questions if q.qid == "q-multi")
    assert question.question == "What data?"
    assert question.pooled_query == "Paper One. What data?"


def test_c2_counts_and_batched_tokens():
    split = qasper.load(_papers())
    calls = []

    def encode(batch):
        calls.append(len(batch))
        return [text.split() for text in batch]

    qasper.TOKEN_BATCH, saved = 2, qasper.TOKEN_BATCH
    try:
        tokens = qasper.token_counts([u.text for u in split.units], encode)
    finally:
        qasper.TOKEN_BATCH = saved
    assert calls == [2, 2, 1]
    assert tokens[0] == 6  # "Paper One - Intro. First para." in words
    counts = qasper.c2_counts(split, tokens)
    assert (counts["papers"], counts["units"], counts["caption_units"]) == (2, 5, 1)
    assert (counts["in_scope"], counts["multi_evidence"], counts["single_evidence"]) == (2, 1, 1)
    assert counts["annotators"] == {"1": 3, "2": 1}


def test_write_keeps_gold_apart(tmp_path):
    split = qasper.load(_papers())
    digests = qasper.write(tmp_path, split, [1] * len(split.units))
    questions = (tmp_path / "questions.jsonl").read_text(encoding="utf-8")
    assert "gold" not in questions and "First para." not in questions
    gold = json.loads((tmp_path / "gold.json").read_text(encoding="utf-8"))
    assert sorted(gold) == ["q-cross", "q-multi", "q-none", "q-twice"]
    assert set(digests) == {
        "ordered_unit_digest",
        "token_counts_digest",
        "question_digest",
        "gold_digest",
    }
    assert json.loads((tmp_path / "qasper.json").read_text(encoding="utf-8")) == digests
