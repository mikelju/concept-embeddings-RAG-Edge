"""Phase 06 rivals, laptop side: the HotpotQA subsample (spec C2) and G-R2 (spec C3, C4).

G-R2 is Qwen3-Reranker-0.6B over G-L's pinned top 100 (spec, frozen definition). The laptop
writes the pairs (`rivals-pairs`, Phase 04's `judge-pairs` path: `judge.pair_row`,
`judge.write_once`), the pod scores them (`edge_rag.pod.qwen_rerank`), and the laptop orders
each question by score descending, ties by G-L rank (`rivals-rank`).
"""

import gzip
import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from edge_rag import components, config, scoring
from edge_rag.artifacts import ArtifactError, keep_manifest, sha256_file, write_bytes, write_json
from edge_rag.judge import SUBSAMPLE_SEED, SUBSAMPLE_SET, SUBSAMPLE_SIZE, pair_row, write_once
from edge_rag.pod.common import git_provenance, open_set, units_by_id
from edge_rag.pod.qwen_rerank import SUBSAMPLE_DIR, subset
from edge_rag.pod.rerank import reorder
from edge_rag.process import peak_rss_mb
from edge_rag.reproduce import Say

SUBSAMPLE_FILE = f"{SUBSAMPLE_SET}-{SUBSAMPLE_SIZE}.txt"
SUBSAMPLE_RULE = f"random.Random({SUBSAMPLE_SEED}).sample(sorted(qids), {SUBSAMPLE_SIZE})"
SYSTEM = "g-r2"


# --- `edge-rag rivals-subsample` (spec C2, plan increment 1) ---------------------------------
def subsample(qids: Sequence[str]) -> list[str]:
    """The preregistered rule (master plan, 2026-10-02), in the order the sample returns."""
    return random.Random(SUBSAMPLE_SEED).sample(sorted(qids), SUBSAMPLE_SIZE)  # noqa: S311


def subsample_bytes(chosen: Sequence[str]) -> bytes:
    return "".join(f"{qid}\n" for qid in chosen).encode("utf-8")


def run_subsample(directory: Path = config.PHASE06_SUBSAMPLE_DIR, say: Say = print) -> str:
    """Write the 1,000 qids once, one per line; an existing file must hold the same bytes."""
    spec, old, checks = open_set(SUBSAMPLE_SET)
    qids = sorted(q.qid for q in scoring.questions_of(old, checks, spec))
    body = subsample_bytes(subsample(qids))
    digest = hashlib.sha256(body).hexdigest()
    path = directory / SUBSAMPLE_FILE
    if path.exists() and sha256_file(path) != digest:
        raise ArtifactError(f"refusing to overwrite {path} with other qids")
    write_bytes(path, body)
    manifest = path.with_suffix(".manifest.json")
    if not manifest.exists():
        write_json(
            manifest,
            {
                "set": SUBSAMPLE_SET,
                "rule": SUBSAMPLE_RULE,
                "population": len(qids),
                "population_sha256": hashlib.sha256(subsample_bytes(qids)).hexdigest(),
                "size": SUBSAMPLE_SIZE,
                **git_provenance(),
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "digests_checked": checks.records,
                "outputs_sha256": {SUBSAMPLE_FILE: digest},
            },
        )
    say(f"[{SUBSAMPLE_SET}] {SUBSAMPLE_SIZE} of {len(qids)} qids; {path} sha256 {digest}")
    return digest


# --- `edge-rag rivals-pairs` (spec C3, C4; plan increment 2) ---------------------------------
def gl_tops(set_name: str) -> dict[str, list[str]]:
    """G-L's depth-100 lists, read only after the file matches its pinned sha256."""
    lists = components.phase02_list(set_name, "g-l", config.GL_SHA256[set_name])
    return {qid: ranked[: config.DEPTH] for qid, ranked in lists.items()}


def run_pairs(
    set_name: str, pairs_dir: Path = config.PHASE06_PAIRS_DIR, say: Say = print
) -> dict[str, Any]:
    manifest_path = pairs_dir / f"{set_name}.manifest.json"
    keep_manifest(manifest_path)
    provenance = git_provenance()
    started = time.perf_counter()
    tops = gl_tops(set_name)
    spec, old, checks = open_set(set_name)
    questions = scoring.questions_of(old, checks, spec)
    if sorted(q.qid for q in questions) != sorted(tops):
        raise ArtifactError(f"{set_name}: G-L does not cover the set's questions")
    units = units_by_id(old, spec, {u for ranked in tops.values() for u in ranked})
    name = f"{set_name}.jsonl.gz"
    digest = write_once(
        pairs_dir / name, (pair_row(q.qid, q.question, tops[q.qid], units) for q in questions)
    )
    body: dict[str, Any] = {
        "set": set_name,
        "system": SYSTEM,
        "questions": len(questions),
        "pairs": sum(len(tops[q.qid]) for q in questions),
        "rule": "G-L's top 100 per question in question-file order, harness unit text",
        **provenance,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_sha256": {"g-l.jsonl.gz": config.GL_SHA256[set_name]},
        "digests_checked": checks.records,
        "seconds": round(time.perf_counter() - started, 3),
        "bytes": {name: (pairs_dir / name).stat().st_size},
        "outputs_sha256": {name: digest},
        "peak_rss_mb": peak_rss_mb(),
    }
    write_json(manifest_path, body)
    say(f"[{set_name}] {body['questions']} questions, {body['pairs']} pairs; {name} {digest}")
    return body


# --- `edge-rag rivals-rank` (spec C4, plan increment 3) ---------------------------------------
def rank(
    tops: Mapping[str, Sequence[str]], scored: Sequence[Mapping[str, Any]]
) -> dict[str, list[str]]:
    """Each question's G-L top reordered by its pod scores, ties by G-L rank. The pod must
    have scored exactly the top, in G-L order, once per question."""
    out: dict[str, list[str]] = {}
    for row in scored:
        qid = str(row["qid"])
        if qid in out:
            raise ArtifactError(f"{qid} scored twice")
        if qid not in tops or list(row["unit_ids"]) != list(tops[qid]):
            raise ArtifactError(f"{qid}: the scored units are not G-L's top")
        out[qid] = reorder(tops[qid], row["scores"])
    if set(out) != set(tops):
        raise ArtifactError(f"{len(set(tops) - set(out))} questions have no scores")
    return out


def pod_scores(set_name: str, scores_dir: Path, pairs_dir: Path) -> tuple[list[dict], dict]:
    """The pod's scores, read only when its C3 checks passed, it scored this laptop's pairs
    file and the scores file matches the sha256 its manifest recorded."""
    name = f"{set_name}.jsonl.gz"
    manifest = json.loads((scores_dir / f"{set_name}.manifest.json").read_text("utf-8"))
    pairs = json.loads((pairs_dir / f"{set_name}.manifest.json").read_text("utf-8"))
    if not manifest["check"]["pass"]:
        raise ArtifactError(f"{set_name}: the pod's C3 checks did not pass")
    if manifest["inputs_sha256"] != pairs["outputs_sha256"]:
        raise ArtifactError(f"{set_name}: the pod scored other pairs files")
    measured = sha256_file(scores_dir / name)
    if measured != manifest["outputs"][name]:
        raise ArtifactError(f"{scores_dir / name}: sha256 {measured}, recorded")
    with gzip.open(scores_dir / name, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle], manifest


def run_rank(
    set_name: str,
    *,
    scores_dir: Path = config.PHASE06_SCORES_DIR,
    pairs_dir: Path = config.PHASE06_PAIRS_DIR,
    rankings_dir: Path = config.PHASE06_RANKINGS_DIR,
    subset_dir: Path = SUBSAMPLE_DIR,
    say: Say = print,
) -> dict[str, Any]:
    directory = rankings_dir / set_name
    manifest_path = directory / f"{SYSTEM}.manifest.json"
    keep_manifest(manifest_path)
    tops = gl_tops(set_name)
    scored, pod = pod_scores(set_name, scores_dir, pairs_dir)
    qids, subset_record = subset(set_name, subset_dir)
    if pod.get("subset") != subset_record:
        raise ArtifactError(f"{set_name}: the pod scored subset {pod.get('subset')}")
    if qids is not None:
        if not qids <= set(tops):
            raise ArtifactError(f"{set_name}: subset qids outside G-L's lists")
        tops = {qid: ranked for qid, ranked in tops.items() if qid in qids}
    ranked = rank(tops, scored)
    order = [str(row["qid"]) for row in scored]
    name = f"{SYSTEM}.jsonl.gz"
    digest = scoring.write_rankings(directory / name, ((qid, ranked[qid]) for qid in order))
    body: dict[str, Any] = {
        "system": SYSTEM,
        "set": set_name,
        "questions": len(order),
        "depth": config.DEPTH,
        "order": "Qwen3-Reranker-0.6B log P(yes) descending, ties by G-L rank",
        "model": pod["model"],
        "revision": pod["revision"],
        "settings": pod["settings"],
        "pod": {k: pod.get(k) for k in ("gpu", "cost_per_hr_usd", "git_commit", "git_src_changes")},
        "offline_seconds": 0.0,
        "offline_note": "no index of its own; G-L's offline cost is G-L's row",
        "online_seconds": pod["seconds"]["score"],
        "online_seconds_per_question": pod["seconds"]["score"] / max(len(order), 1),
        **git_provenance(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_sha256": {
            "g-l.jsonl.gz": config.GL_SHA256[set_name],
            **pod["inputs_sha256"],
            f"{set_name}.scores.jsonl.gz": pod["outputs"][f"{set_name}.jsonl.gz"],
            f"{set_name}.scores.manifest.json": sha256_file(
                scores_dir / f"{set_name}.manifest.json"
            ),
        },
        "check": pod["check"],
        "subset": subset_record,
        "outputs_sha256": {name: digest},
    }
    write_json(manifest_path, body)
    say(f"[{set_name}] {len(order)} questions ranked; {name} {digest}")
    return body
