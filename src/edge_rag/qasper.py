"""Phase 07: the QASPER loader (spec "Rules fixed before the exam opens", criterion C2).

Units: one per non-empty full-text paragraph, titled `<paper title> - <section name>`, and one
per table or figure, body `FLOAT SELECTED: <caption>` (the prefix kept so evidence strings
match), titled with the paper title. A unit's indexed text is `title. body`, as on the terrain
(`corpus.Corpus.texts`), and its budget token count is taken over that whole text with the
terrain's tokenizer, in batches. Unit ids are `corpus.unit_id_for` with the paper id prefixed, so
two papers never share a unit; a text repeated inside one paper is one unit (its first place),
so an evidence string maps to exactly one unit.

Gold: an evidence string maps only to a unit of the question's own paper, by exact match after
whitespace normalization. A question is in scope when at least one annotator marks it answerable
with a non-empty evidence set that maps fully; it is multi-evidence when every such set has two or
more units. Questions and gold are kept apart: ranking reads `questions.jsonl` only.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_rag import config
from edge_rag.artifacts import digest_of, write_bytes, write_json
from edge_rag.corpus import unit_id_for

FLOAT_PREFIX = "FLOAT SELECTED: "
TOKEN_BATCH = 1024


@dataclass(frozen=True)
class Unit:
    unit_id: str
    paper_id: str
    title: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}"


@dataclass(frozen=True)
class QasperQuestion:
    qid: str
    paper_id: str
    question: str
    # D8: the pooled query names the paper; the within-paper control keeps the question alone.
    pooled_query: str


@dataclass(frozen=True)
class Split:
    papers: int
    units: list[Unit]
    questions: list[QasperQuestion]
    # qid -> every fully mapped answerable annotator set (sorted unit ids); [] = out of scope.
    gold: dict[str, list[tuple[str, ...]]]
    evidence_strings: int
    evidence_matched: int
    float_strings: int
    float_matched: int
    unmapped_sets: int
    annotators: dict[str, int]


def normalize(text: str) -> str:
    return " ".join(text.split())


def paper_units(paper_id: str, paper: Mapping[str, Any]) -> list[Unit]:
    title = normalize(str(paper["title"]))
    units: list[Unit] = []
    seen: set[str] = set()

    def add(unit_title: str, body: str) -> None:
        body = normalize(body)
        if body and body not in seen:
            seen.add(body)
            unit_id = unit_id_for(f"{paper_id}\n{unit_title}", [body])
            units.append(Unit(unit_id, paper_id, unit_title, body))

    for section in paper["full_text"]:
        name = normalize(str(section["section_name"] or ""))
        for paragraph in section["paragraphs"]:
            add(f"{title} - {name}" if name else title, str(paragraph))
    for item in paper["figures_and_tables"]:
        add(title, FLOAT_PREFIX + str(item["caption"]))
    return units


def load(papers: Mapping[str, Mapping[str, Any]]) -> Split:
    """Units, questions and gold of one released split (`qasper-<split>-v0.3.json`)."""
    units: list[Unit] = []
    questions: list[QasperQuestion] = []
    gold: dict[str, list[tuple[str, ...]]] = {}
    strings = matched = float_strings = float_matched = unmapped = 0
    annotators: Counter[int] = Counter()
    for paper_id in sorted(papers):
        paper = papers[paper_id]
        own = paper_units(paper_id, paper)
        units.extend(own)
        by_body = {unit.body: unit.unit_id for unit in own}
        title = normalize(str(paper["title"]))
        for qa in paper["qas"]:
            qid, question = str(qa["question_id"]), normalize(str(qa["question"]))
            questions.append(QasperQuestion(qid, paper_id, question, f"{title}. {question}"))
            annotators[len(qa["answers"])] += 1
            sets: list[tuple[str, ...]] = []
            for answer in qa["answers"]:
                body = answer["answer"]
                evidence = [normalize(str(e)) for e in body["evidence"]]
                if body["unanswerable"] or not evidence:
                    continue
                ids = [by_body.get(e) for e in evidence]
                strings += len(evidence)
                matched += sum(i is not None for i in ids)
                floats = [i for e, i in zip(evidence, ids, strict=True) if e.startswith("FLOAT")]
                float_strings += len(floats)
                float_matched += sum(i is not None for i in floats)
                if all(i is not None for i in ids):
                    sets.append(tuple(sorted({str(i) for i in ids})))
                else:
                    unmapped += 1
            gold[qid] = sets
    return Split(
        len(papers),
        units,
        questions,
        gold,
        strings,
        matched,
        float_strings,
        float_matched,
        unmapped,
        {str(k): v for k, v in sorted(annotators.items())},
    )


def token_counts(
    texts: Sequence[str], encode: Callable[[list[str]], list[list[int]]] | None = None
) -> list[int]:
    """Budget tokens per text with the terrain's tokenizer, `TOKEN_BATCH` texts per call."""
    if encode is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION
        )

        def encode(batch: list[str]) -> list[list[int]]:
            return tokenizer(batch, add_special_tokens=False)["input_ids"]

    counts: list[int] = []
    for start in range(0, len(texts), TOKEN_BATCH):
        counts.extend(len(ids) for ids in encode(list(texts[start : start + TOKEN_BATCH])))
    return counts


def c2_counts(split: Split, tokens: Sequence[int]) -> dict[str, Any]:
    """The C2 figures of one train or dev split (never called on test before scoring)."""
    papers = split.papers
    in_scope = [sets for sets in split.gold.values() if sets]
    multi = sum(all(len(s) >= 2 for s in sets) for sets in in_scope)
    floats = sum(unit.body.startswith(FLOAT_PREFIX) for unit in split.units)
    return {
        "papers": papers,
        "questions": len(split.questions),
        "units": len(split.units),
        "paragraph_units": len(split.units) - floats,
        "caption_units": floats,
        "units_per_paper": round(len(split.units) / papers, 2),
        "tokens_total": int(sum(tokens)),
        "tokens_per_unit": round(sum(tokens) / len(tokens), 2),
        "evidence_strings": split.evidence_strings,
        "evidence_matched": split.evidence_matched,
        "float_strings": split.float_strings,
        "float_matched": split.float_matched,
        "unmapped_sets": split.unmapped_sets,
        "annotators": split.annotators,
        "in_scope": len(in_scope),
        "multi_evidence": multi,
        "single_evidence": len(in_scope) - multi,
    }


def write(out: Path, split: Split, tokens: Sequence[int]) -> dict[str, str]:
    """`units.jsonl`, `questions.jsonl` and the separate `gold.json`, with their digests."""
    unit_lines = [
        json.dumps(
            {
                "unit_id": u.unit_id,
                "paper_id": u.paper_id,
                "title": u.title,
                "body": u.body,
                "tokens": t,
            },
            sort_keys=True,
        )
        for u, t in zip(split.units, tokens, strict=True)
    ]
    question_lines = [
        json.dumps(
            {
                "qid": q.qid,
                "paper_id": q.paper_id,
                "question": q.question,
                "pooled_query": q.pooled_query,
            },
            sort_keys=True,
        )
        for q in split.questions
    ]
    gold = {qid: [list(s) for s in sets] for qid, sets in sorted(split.gold.items())}
    write_bytes(out / "units.jsonl", ("\n".join(unit_lines) + "\n").encode("utf-8"))
    write_bytes(out / "questions.jsonl", ("\n".join(question_lines) + "\n").encode("utf-8"))
    write_json(out / "gold.json", gold)
    digests = {
        "ordered_unit_digest": digest_of(*(u.unit_id for u in split.units)),
        "token_counts_digest": digest_of(*tokens),
        "question_digest": digest_of(*question_lines),
        "gold_digest": digest_of(json.dumps(gold, sort_keys=True)),
    }
    write_json(out / "qasper.json", digests)
    return digests


def run_c2(train_dev: Path, say: Callable[[str], None]) -> dict[str, Any]:
    """C2 on train and dev: load, count tokens in batches, write each split, report counts."""
    figures: dict[str, Any] = {}
    for name in ("train", "dev"):
        papers = json.loads((train_dev / f"qasper-{name}-v0.3.json").read_text(encoding="utf-8"))
        split = load(papers)
        tokens = token_counts([unit.text for unit in split.units])
        write(config.DATA_DIR / "phase07" / name, split, tokens)
        figures[name] = c2_counts(split, tokens)
        say(f"[{name}] {json.dumps(figures[name], sort_keys=True)}")
    return figures
