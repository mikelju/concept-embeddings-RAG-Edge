"""What every ghost shares: the digest-checked corpus stream, questions, ranking files, manifests.

Imports only the harness modules (`config`, `artifacts`, `corpus`), so it runs in both pod
environments: the `pod` group (G-R, G-A1) and `colbert_env` (G-L).
"""

import gzip
import hashlib
import io
import json
import os
import subprocess  # noqa: S404 - fixed git command, no user input
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_rag import config
from edge_rag.artifacts import ArtifactError, Checks, OldData, sha256_file, write_bytes, write_json
from edge_rag.corpus import Question, load_questions, unit_id_for, unit_set_hash

PHASE_DIR = config.DATA_DIR / "phase02"
# The pod's hourly rate, recorded at pod creation (C5); exported by `pod_run.sh`.
COST_PER_HR_ENV = "POD_COST_PER_HR"


@dataclass(frozen=True)
class Unit:
    unit_id: str
    title: str
    body: str  # `' '.join(sentences)`

    @property
    def text(self) -> str:
        """The harness text: what Dense, BM25 and the old judges read."""
        return f"{self.title}. {self.body}"


def open_set(name: str) -> tuple[config.CorpusSet, OldData, Checks]:
    spec = config.SETS[name]
    checks = Checks()
    return spec, OldData(spec.directory, spec.file_sha256, checks), checks


def iter_units(old: OldData, spec: config.CorpusSet, *, limit: int | None = None) -> Iterator[Unit]:
    """The corpus in file order, one unit at a time, with every id re-derived from its content.

    The ordered digest and the unit set hash are checked once the stream is exhausted (they
    cover the whole corpus, so a `limit` skips them).
    """
    manifest = old.json("corpus.json")
    if manifest["ordered_unit_digest"] != spec.ordered_unit_digest:
        raise ArtifactError("corpus.json ordered_unit_digest differs from the pin")
    ordered = hashlib.sha256()
    unit_ids: list[str] = []
    with (
        old.verified_path(str(manifest["corpus_file"])).open("rb") as raw,
        gzip.GzipFile(fileobj=raw) as gz,
    ):
        for number, line in enumerate(io.TextIOWrapper(gz, encoding="utf-8"), start=1):
            if limit is not None and number > limit:
                return
            entry = json.loads(line)
            title = str(entry["title"])
            sentences = [str(sentence) for sentence in entry["sentences"]]
            unit_id = unit_id_for(title, sentences)
            if unit_id != entry["unit_id"]:
                raise ArtifactError(f"corpus line {number}: {entry['unit_id']} != {unit_id}")
            ordered.update(unit_id.encode("utf-8") + b"\x00")
            unit_ids.append(unit_id)
            yield Unit(unit_id, title, " ".join(sentences))
    if limit is None:
        if ordered.hexdigest() != spec.ordered_unit_digest:
            raise ArtifactError(f"{spec.name}: corpus ordered unit digest differs from the pin")
        if unit_set_hash(unit_ids) != spec.unit_set_hash:
            raise ArtifactError(f"{spec.name}: corpus unit set hash differs from the pin")


def units_by_id(old: OldData, spec: config.CorpusSet, wanted: set[str] | None) -> dict[str, Unit]:
    """The units in `wanted` (all when None), streamed so a large corpus is never held whole."""
    found = {u.unit_id: u for u in iter_units(old, spec) if wanted is None or u.unit_id in wanted}
    if wanted is not None and len(found) != len(wanted):
        raise ArtifactError(f"{spec.name}: {len(wanted) - len(found)} ranked units not in corpus")
    return found


def set_questions(
    old: OldData, checks: Checks, spec: config.CorpusSet
) -> tuple[list[Question], dict[str, str]]:
    """The frozen questions in file order and each one's gold answer ('' when none)."""
    questions, body = load_questions(
        old,
        checks,
        corpus_set_hash=spec.unit_set_hash,
        question_digest_recorded=spec.question_digest,
        mapping_digest_recorded=spec.mapping_digest,
    )
    if len(questions) != spec.questions:
        raise ArtifactError(f"{len(questions)} questions, expected {spec.questions}")
    answers = {str(e["qid"]): str(e.get("answer") or "") for e in body["questions"]}
    return questions, answers


def rankings_path(set_name: str, system: str, tag: str | None = None) -> Path:
    base = PHASE_DIR / ("rankings" if tag is None else f"probe/{tag}") / set_name
    return base / f"{system}.jsonl.gz"


def manifest_path(rankings: Path) -> Path:
    return rankings.with_name(rankings.name.removesuffix(".jsonl.gz") + ".manifest.json")


def gz_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Rows as gzip JSONL with a fixed header (mtime 0), so equal rows give equal bytes."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(dict(row), sort_keys=True) + "\n").encode("utf-8"))
    return buffer.getvalue()


def write_rankings(path: Path, rankings: Sequence[tuple[str, Sequence[str]]]) -> str:
    """`{"qid", "ranked"}` per question, in question order; returns the file's sha256."""
    write_bytes(path, gz_jsonl({"qid": qid, "ranked": list(ranked)} for qid, ranked in rankings))
    return sha256_file(path)


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_rankings(path: Path) -> list[tuple[str, list[str]]]:
    return [(str(r["qid"]), [str(u) for u in r["ranked"]]) for r in read_jsonl_gz(path)]


def git_commit() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=config.REPO_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def git_provenance() -> dict[str, Any]:
    """HEAD and the uncommitted changes under src/, read when a run starts: a manifest whose
    `git_src_changes` is not empty was written by code that no commit holds."""
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain", "--", "src"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=config.REPO_ROOT,
        )
        changes: list[str] | str = [line.strip() for line in out.stdout.splitlines()]
    except (OSError, subprocess.CalledProcessError):
        changes = "unknown"
    return {"git_commit": git_commit(), "git_src_changes": changes}


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        return "none"


def cost_per_hr() -> float | None:
    value = os.environ.get(COST_PER_HR_ENV)
    return float(value) if value else None


def write_manifest(path: Path, body: Mapping[str, Any], outputs: Sequence[Path]) -> Path:
    """The run's manifest: its own fields plus commit, GPU, rate and each output's sha256."""
    full = dict(body)
    full.update(
        {
            "git_commit": git_commit(),
            "gpu": gpu_name(),
            "cost_per_hr_usd": cost_per_hr(),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "outputs": {p.name: sha256_file(p) for p in outputs},
        }
    )
    return write_json(path, full)


class Timer:
    """Named wall-clock seconds, summed over repeated laps."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def add(self, name: str, started: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started
