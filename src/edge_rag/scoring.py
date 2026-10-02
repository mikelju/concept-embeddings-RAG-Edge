"""Ranking files and their scores: the one scoring loop that `reproduce` and `score` share."""

import gzip
import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edge_rag import config, metrics
from edge_rag.artifacts import ArtifactError, Checks, OldData, digest_of, write_bytes, write_json
from edge_rag.corpus import Question, load_questions

MANIFEST = "manifest.json"


def token_counts(old: OldData, checks: Checks, spec: config.CorpusSet) -> dict[str, int]:
    """Token counts keyed by unit id; the ids must be the corpus units in their proved order."""
    manifest = old.json("token-counts.json")
    if manifest["unit_set_hash"] != spec.unit_set_hash:
        raise ArtifactError("the token counts belong to another corpus")
    raw_ids, raw_counts = old.arrays("token-counts.npz", "unit_ids", "counts")
    ids = [str(unit_id) for unit_id in raw_ids]
    counts = raw_counts.astype(np.int32)
    checks.expect("token-count unit order", digest_of(*ids), spec.ordered_unit_digest)
    checks.expect(
        "token-counts.json counts_digest", manifest["counts_digest"], spec.token_counts_digest
    )
    checks.expect("token counts digest", digest_of(counts), spec.token_counts_digest)
    return dict(zip(ids, (int(c) for c in counts), strict=True))


def questions_of(old: OldData, checks: Checks, spec: config.CorpusSet) -> list[Question]:
    questions, _body = load_questions(
        old,
        checks,
        corpus_set_hash=spec.unit_set_hash,
        question_digest_recorded=spec.question_digest,
        mapping_digest_recorded=spec.mapping_digest,
    )
    if len(questions) != spec.questions:
        raise ArtifactError(f"{len(questions)} questions, expected {spec.questions}")
    return questions


def score(
    questions: Sequence[Question],
    rankings: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
) -> tuple[dict[str, Any], list[int]]:
    """Every metric of one ranking per question, and the per-question FS at `config.BUDGET`."""
    wanted = {q.qid for q in questions}
    missing, extra = wanted - rankings.keys(), rankings.keys() - wanted
    if missing or extra:
        raise ArtifactError(
            f"rankings do not cover the questions: {len(missing)} missing, {len(extra)} extra"
        )
    budgets = {b: 0 for b in config.FS_BUDGETS}
    at_k = {k: 0 for k in config.FS_UNIT_KS}
    recall = {k: 0.0 for k in config.RECALL_KS}
    share = ndcg = 0.0
    record: list[int] = []
    for question in questions:
        ranked, gold = list(rankings[question.qid]), question.gold_unit_ids
        for budget in budgets:
            hit = metrics.full_support(metrics.fill_context(ranked, counts, budget), gold)
            budgets[budget] += hit
            if budget == config.BUDGET:
                record.append(hit)
        for k in at_k:
            at_k[k] += metrics.full_support_at_k(ranked, gold, k)
        for k in recall:
            recall[k] += metrics.recall_at_k(ranked, gold, k)
        share += metrics.gold_share_at_k(ranked, gold, config.GOLD_SHARE_K)
        ndcg += metrics.ndcg_at_k(ranked, gold, config.NDCG_K)
    n = len(questions)
    summary = {
        "n": n,
        "full_support_at_budget": {str(b): v for b, v in budgets.items()},
        "full_support_at_k": {str(k): v for k, v in at_k.items()},
        f"gold_share_at_{config.GOLD_SHARE_K}": share / n,
        "recall_at_k": {str(k): v / n for k, v in recall.items()},
        f"ndcg_at_{config.NDCG_K}": ndcg / n,
    }
    return summary, record


def ranking_bytes(rows: Iterable[tuple[str, Sequence[str]]]) -> bytes:
    lines = "".join(json.dumps({"qid": qid, "ranked": list(ranked)}) + "\n" for qid, ranked in rows)
    return gzip.compress(lines.encode("utf-8"), mtime=0)


def write_rankings(path: Path, rows: Iterable[tuple[str, Sequence[str]]]) -> str:
    """Write once: an existing file must already hold these bytes. The sha256 goes to the
    directory's manifest and is returned."""
    body = ranking_bytes(rows)
    digest = hashlib.sha256(body).hexdigest()
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ArtifactError(f"refusing to overwrite {path} with different rankings")
    write_bytes(path, body)
    manifest_path = path.parent / MANIFEST
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    if manifest.get(path.name, digest) != digest:
        raise ArtifactError(f"{manifest_path} records another sha256 for {path.name}")
    manifest[path.name] = digest
    write_json(manifest_path, manifest)
    return digest


def read_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            qid = str(record["qid"])
            if qid in rankings:
                raise ArtifactError(f"{path}: qid {qid} appears twice")
            rankings[qid] = [str(unit_id) for unit_id in record["ranked"]]
    return rankings


def old_references(set_name: str) -> dict[str, str]:
    """J(P10-B) and the union pool of old Phase 17 (J-strong), as ranking files of this repo."""
    files = config.OLD_REFERENCES[set_name]
    old = OldData(config.OLD_REFERENCES_DIR, dict(files.values()))
    written: dict[str, str] = {}
    for name, (relative, _sha) in files.items():
        rows: list[tuple[str, list[str]]] = []
        with old.verified_path(relative).open("rb") as raw, gzip.GzipFile(fileobj=raw) as gz:
            for line in io.TextIOWrapper(gz, encoding="utf-8"):
                record = json.loads(line)
                rows.append((str(record["qid"]), [str(entry[0]) for entry in record["ranked"]]))
        written[name] = write_rankings(config.RANKINGS_DIR / set_name / f"{name}.jsonl.gz", rows)
    return written


def run(set_name: str, rankings_path: Path, against: Path | None = None) -> dict[str, Any]:
    spec = config.SETS[set_name]
    checks = Checks()
    old = OldData(spec.directory, spec.file_sha256, checks)
    questions = questions_of(old, checks, spec)
    counts = token_counts(old, checks, spec)
    summary, record = score(questions, read_rankings(rankings_path), counts)
    result: dict[str, Any] = {
        "set": set_name,
        "rankings": str(rankings_path),
        "rankings_sha256": hashlib.sha256(rankings_path.read_bytes()).hexdigest(),
        **summary,
        "digests_checked": checks.records,
    }
    if against is not None:
        base_summary, base = score(questions, read_rankings(against), counts)
        result["against"] = {
            "rankings": str(against),
            f"full_support_at_{config.BUDGET}": base_summary["full_support_at_budget"][
                str(config.BUDGET)
            ],
            f"paired_full_support_at_{config.BUDGET}": metrics.paired(base, record),
        }
    return result
