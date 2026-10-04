"""The corpus and question loaders, for the three inherited formats (one JSONL layout).

Copied from the old `corpus/fullwiki.load_corpus`, `corpus/pool.unit_id_for`,
`evaluation/fullwiki.load_questions` and `corpus/musique.load_questions`: every unit id is
re-derived from its content, the order is proved by the recorded ordered digest, and the
question and gold mapping digests are recomputed.
"""

import gzip
import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass

from edge_rag.artifacts import ArtifactError, Checks, OldData, digest_of


@dataclass(frozen=True)
class Corpus:
    unit_ids: list[str]
    # `indexable_text = f"{title}. {' '.join(sentences)}"`, what Dense and BM25 index.
    texts: list[str]
    titles: list[str]

    def body(self, row: int) -> str:
        """The unit's sentences joined by spaces: its text after the `title + ". "` prefix."""
        return self.texts[row][len(self.titles[row]) + 2 :]


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    gold_unit_ids: tuple[str, ...]
    split: str


def unit_id_for(title: str, sentences: Sequence[str]) -> str:
    payload = title + "\n" + " ".join(sentences)
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def unit_set_hash(unit_ids: Sequence[str]) -> str:
    joined = "\n".join(sorted(unit_ids))
    return hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def load_corpus(old: OldData, checks: Checks, *, ordered_digest: str, set_hash: str) -> Corpus:
    manifest = old.json("corpus.json")
    checks.expect(
        "corpus.json ordered_unit_digest", manifest["ordered_unit_digest"], ordered_digest
    )
    unit_ids: list[str] = []
    texts: list[str] = []
    titles: list[str] = []
    with (
        old.verified_path(str(manifest["corpus_file"])).open("rb") as raw,
        gzip.GzipFile(fileobj=raw) as gz,
    ):
        for number, line in enumerate(io.TextIOWrapper(gz, encoding="utf-8"), start=1):
            entry = json.loads(line)
            title = str(entry["title"])
            sentences = [str(sentence) for sentence in entry["sentences"]]
            unit_id = unit_id_for(title, sentences)
            if unit_id != entry["unit_id"]:
                raise ArtifactError(f"corpus line {number}: {entry['unit_id']} != {unit_id}")
            unit_ids.append(unit_id)
            texts.append(f"{title}. {' '.join(sentences)}")
            titles.append(title)
    checks.expect("corpus ordered unit digest", digest_of(*unit_ids), ordered_digest)
    checks.expect("corpus unit set hash", unit_set_hash(unit_ids), set_hash)
    return Corpus(unit_ids=unit_ids, texts=texts, titles=titles)


def question_digest(questions: Sequence[Question]) -> str:
    return digest_of(
        *(f"{q.qid}\t{q.split}\t{q.question}" for q in sorted(questions, key=lambda q: q.qid))
    )


def mapping_digest(questions: Sequence[Question], corpus_set_hash: str) -> str:
    """Every gold id of every question; for two gold ids it equals the Phase 9 digest."""
    return digest_of(
        corpus_set_hash,
        *("\t".join((q.qid, *q.gold_unit_ids)) for q in sorted(questions, key=lambda q: q.qid)),
    )


def load_questions(
    old: OldData,
    checks: Checks,
    *,
    corpus_set_hash: str,
    question_digest_recorded: str,
    mapping_digest_recorded: str,
) -> tuple[list[Question], dict]:
    """The frozen questions and the body they came from (its `live_path`, if any, is kept)."""
    body = old.json("questions.json")
    if body.get("terminal_state") is not None:
        raise ArtifactError("questions.json records a DATA_STOP")
    if body["corpus_unit_set_hash"] != corpus_set_hash:
        raise ArtifactError("questions.json was mapped against another corpus")
    questions = [
        Question(
            qid=str(entry["qid"]),
            question=str(entry["question"]),
            gold_unit_ids=tuple(str(unit_id) for unit_id in entry["gold_unit_ids"]),
            split=str(entry["cohort"] if "cohort" in entry else body["split"]),
        )
        for entry in body["questions"]
    ]
    checks.expect(
        "questions.json question digest", body["question_digest"], question_digest_recorded
    )
    checks.expect("question digest", question_digest(questions), question_digest_recorded)
    checks.expect(
        "gold mapping digest", mapping_digest(questions, corpus_set_hash), mapping_digest_recorded
    )
    return questions, body
