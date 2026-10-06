"""Phase 06 G-A2 (HippoRAG 2 on MultiHop-RAG), laptop side (spec C3, C4; plan increment 4).

`export` writes the upload bundle: every MultiHop-RAG unit (harness text, corpus file order)
and every question (question file order), gzip JSON lines, with a manifest and a `sha256sum`
file. The pod (`edge_rag.pod.hipporag_run`, in HippoRAG's own environment) indexes the units
in HippoRAG's normal flow and writes, per question, the retrieved passages as groups of unit
ids (units with identical text are one passage to HippoRAG). `rank` checks the pod's manifest
against this bundle, flattens each question's groups to a depth-100 unit ranking and writes it
in the G-R2 format (`scoring.write_rankings` plus a `g-a2.manifest.json`).

    uv run python -m edge_rag.ga2 export --out <dir>
    uv run python -m edge_rag.ga2 rank --bundle <dir> --pod <dir>
"""

import argparse
import gzip
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from edge_rag import config, scoring
from edge_rag.artifacts import ArtifactError, keep_manifest, sha256_file, write_bytes, write_json
from edge_rag.judge import write_once
from edge_rag.pod.common import git_provenance, iter_units, open_set
from edge_rag.reproduce import Say

SET = "multihop-rag"
SYSTEM = "g-a2"
UNITS_FILE = "units.jsonl.gz"
QUESTIONS_FILE = "questions.jsonl.gz"
BUNDLE_MANIFEST = "bundle.manifest.json"
SUMS_FILE = "bundle.sha256"
POD_RANKINGS = "hipporag.jsonl.gz"
POD_MANIFEST = "hipporag.manifest.json"
EXPECTED = {"units": 27_989, "questions": 2_255}


# --- `ga2 export` -----------------------------------------------------------------------------
def duplicate_texts(texts: Iterable[str]) -> dict[str, int]:
    """How many texts are shared by more than one unit, and how many units they cover:
    HippoRAG keys passages by content hash, so each such text is one passage there."""
    shared = [n for n in Counter(texts).values() if n > 1]
    return {"texts": len(shared), "units": sum(shared)}


def sums_bytes(digests: Mapping[str, str]) -> bytes:
    """`sha256sum -c` input, two spaces between digest and name."""
    return "".join(f"{digests[name]}  {name}\n" for name in sorted(digests)).encode("utf-8")


def run_export(out: Path, say: Say = print) -> dict[str, Any]:
    manifest_path = out / BUNDLE_MANIFEST
    keep_manifest(manifest_path)
    started = time.perf_counter()
    spec, old, checks = open_set(SET)
    units = [{"unit_id": u.unit_id, "text": u.text} for u in iter_units(old, spec)]
    questions = [
        {"qid": q.qid, "question": q.question} for q in scoring.questions_of(old, checks, spec)
    ]
    counts = {"units": len(units), "questions": len(questions)}
    if counts != EXPECTED:
        raise ArtifactError(f"{SET}: {counts}, expected {EXPECTED}")
    digests = {
        UNITS_FILE: write_once(out / UNITS_FILE, units),
        QUESTIONS_FILE: write_once(out / QUESTIONS_FILE, questions),
    }
    body: dict[str, Any] = {
        "set": SET,
        "system": SYSTEM,
        **counts,
        "unit_text": "harness text, `title. body` (what Dense, BM25 and G-R2 read)",
        "order": "units in corpus file order, questions in question file order",
        "duplicate_texts": duplicate_texts(u["text"] for u in units),
        **git_provenance(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digests_checked": checks.records,
        "seconds": round(time.perf_counter() - started, 3),
        "bytes": {name: (out / name).stat().st_size for name in digests},
        "outputs_sha256": digests,
    }
    write_json(manifest_path, body)
    write_bytes(out / SUMS_FILE, sums_bytes(digests))
    say(f"[{SET}] {counts['units']} units, {counts['questions']} questions; {digests}")
    return body


# --- `ga2 rank` -------------------------------------------------------------------------------
def flatten(groups: Sequence[Sequence[str]], depth: int = config.DEPTH) -> list[str]:
    """Each passage's units in the passage's rank order, cut at `depth`; a unit seen twice is
    an adapter fault, not a tie."""
    ranked = [unit_id for group in groups for unit_id in group]
    if len(set(ranked)) != len(ranked):
        raise ArtifactError("a unit appears in two retrieved passages")
    return ranked[:depth]


def rank(
    questions: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    unit_ids: set[str],
    depth: int = config.DEPTH,
) -> dict[str, list[str]]:
    """The pod's rows as depth-`depth` unit rankings: one row per question, every id a corpus
    unit, and a full list unless the corpus is smaller."""
    out: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row["qid"])
        if qid in out:
            raise ArtifactError(f"{qid} retrieved twice")
        ranked = flatten(row["passages"], depth)
        if not set(ranked) <= unit_ids:
            raise ArtifactError(f"{qid}: retrieved ids that are not corpus units")
        if len(ranked) < min(depth, len(unit_ids)):
            raise ArtifactError(f"{qid}: {len(ranked)} units, fewer than {depth}")
        out[qid] = ranked
    if set(out) != set(questions):
        raise ArtifactError(f"{len(set(questions) ^ set(out))} questions differ from the bundle")
    return out


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def pod_rows(bundle: Path, pod: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The pod's rows, read only when it indexed this bundle, its C3 diff record is present
    and the rankings file matches the sha256 its manifest recorded."""
    manifest = json.loads((pod / POD_MANIFEST).read_text("utf-8"))
    exported = json.loads((bundle / BUNDLE_MANIFEST).read_text("utf-8"))
    if manifest["inputs_sha256"] != exported["outputs_sha256"]:
        raise ArtifactError("the pod indexed another bundle")
    if not manifest.get("check", {}).get("diff_sha256"):
        raise ArtifactError("the pod manifest has no C3 diff record")
    measured = sha256_file(pod / POD_RANKINGS)
    if measured != manifest["outputs"][POD_RANKINGS]:
        raise ArtifactError(f"{pod / POD_RANKINGS}: sha256 {measured}, not the recorded one")
    return read_rows(pod / POD_RANKINGS), manifest


def run_rank(
    bundle: Path, pod: Path, rankings_dir: Path = config.PHASE06_RANKINGS_DIR, say: Say = print
) -> dict[str, Any]:
    directory = rankings_dir / SET
    manifest_path = directory / f"{SYSTEM}.manifest.json"
    keep_manifest(manifest_path)
    rows, pod_manifest = pod_rows(bundle, pod)
    questions = [str(r["qid"]) for r in read_rows(bundle / QUESTIONS_FILE)]
    unit_ids = {str(r["unit_id"]) for r in read_rows(bundle / UNITS_FILE)}
    ranked = rank(questions, rows, unit_ids)
    name = f"{SYSTEM}.jsonl.gz"
    digest = scoring.write_rankings(directory / name, ((qid, ranked[qid]) for qid in questions))
    seconds = pod_manifest["seconds"]
    body: dict[str, Any] = {
        "system": SYSTEM,
        "set": SET,
        "questions": len(questions),
        "depth": config.DEPTH,
        "order": "HippoRAG 2 passage ranking (retrieve, num_to_retrieve=100); units sharing "
        "a passage text follow in corpus order; cut at 100 units",
        "models": pod_manifest["models"],
        "hipporag": pod_manifest["hipporag"],
        "settings": pod_manifest["settings"],
        "pod": {k: pod_manifest.get(k) for k in ("gpu", "cost_per_hr_usd", "started_utc")},
        "offline_seconds": seconds["index"],
        "online_seconds": seconds["retrieve"],
        "online_seconds_per_question": seconds["retrieve"] / max(len(questions), 1),
        "llm_tokens": pod_manifest["llm_tokens"],
        **git_provenance(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_sha256": {
            **pod_manifest["inputs_sha256"],
            POD_RANKINGS: pod_manifest["outputs"][POD_RANKINGS],
            POD_MANIFEST: sha256_file(pod / POD_MANIFEST),
        },
        "check": pod_manifest["check"],
        "outputs_sha256": {name: digest},
    }
    write_json(manifest_path, body)
    say(f"[{SET}] {len(questions)} questions ranked; {name} {digest}")
    return body


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m edge_rag.ga2", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="write the MultiHop-RAG upload bundle")
    export.add_argument("--out", type=Path, default=config.PHASE06_DIR / "ga2")
    ranked = sub.add_parser("rank", help="convert the pod's HippoRAG output to rankings")
    ranked.add_argument("--bundle", type=Path, default=config.PHASE06_DIR / "ga2")
    ranked.add_argument("--pod", type=Path, default=config.PHASE06_DIR / "ga2" / "pod")
    ranked.add_argument("--rankings", type=Path, default=config.PHASE06_RANKINGS_DIR)
    args = parser.parse_args(argv)
    if args.command == "export":
        run_export(args.out)
    else:
        run_rank(args.bundle, args.pod, args.rankings)


if __name__ == "__main__":
    main()
