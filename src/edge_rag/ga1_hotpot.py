"""Phase 06 G-A1 (Search-R1) on the HotpotQA dev 1,000, laptop side (spec C3, C4; plan
increment 5, D3).

`export` writes the upload bundle: the preregistered qid list (checked against its recorded
sha256 and against the question file), `upload.sha256` for it, `old.sha256` for the three old
HotpotQA files the pod reads under `OLD_DATA_ROOT` (checked here against their pins first), and
a manifest with sizes and digests. The pod (`scripts/pod_ga1.sh`) rebuilds the G-L index with
the Phase 02 D17 settings and runs the Phase 02 driver, unchanged, through
`edge_rag.pod.ga1_subset`; the driver itself writes each question's evidence list (retrieved
units in retrieval order, later duplicates dropped). `rank` checks the downloaded outputs
against their digests, the C3 diff, the qid list and the corpus, and writes the evidence lists
in the Phase 06 ranking format (`scoring.write_rankings` plus a `g-a1.manifest.json`).

    uv run python -m edge_rag.ga1_hotpot export --out <dir>
    uv run python -m edge_rag.ga1_hotpot rank --bundle <dir> --pod <dir>
"""

import argparse
import gzip
import json
import shutil
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from edge_rag import config, scoring
from edge_rag.artifacts import ArtifactError, keep_manifest, sha256_file, write_bytes, write_json
from edge_rag.ga2 import sums_bytes
from edge_rag.pod import searchr1
from edge_rag.pod.common import git_provenance, iter_units, open_set
from edge_rag.pod.ga1_subset import read_qids
from edge_rag.reproduce import Say

SET = "hotpotqa-dev"
SYSTEM = "g-a1"
QIDS_FILE = "hotpotqa-dev-1000.txt"
# Plan increment 1 (C2): the preregistered subsample, as `rivals-subsample` wrote it.
QIDS_SHA256 = "6cebd41ff9e5e9f02e27ebb019620de0e37b77794fc6a783a2ff0d80fb3a62bd"
OLD_FILES = ("corpus.json", "corpus.jsonl.gz", "questions.json")
UPLOAD_MANIFEST = "upload.manifest.json"
UPLOAD_SUMS = "upload.sha256"
OLD_SUMS = "old.sha256"
# The pod's outputs, as `scripts/pod_ga1.sh` collects them.
POD_RANKINGS = "g-a1.jsonl.gz"
POD_MANIFEST = "g-a1.manifest.json"
POD_SUBSET = "subset.json"
POD_DIFF = "c3-g-a1.diff"
POD_GL_MANIFEST = "g-l.manifest.json"
POD_INDEX_SUMS = "index.sha256"
POD_SUMS = "sha256sums.txt"
# Phase 02 D17: one `create` call over the whole corpus, fp16 embeddings in 200,000-unit batches.
D17 = {
    "chunk_units": 6_000_000,
    "encode_units": 200_000,
    "embeddings_dtype": "float16",
    "index_chunks": 1,
    "units": 5_233_329,
}


# --- `ga1_hotpot export` ------------------------------------------------------------------------
def run_export(
    out: Path, qids_path: Path = config.PHASE06_SUBSAMPLE_DIR / QIDS_FILE, say: Say = print
) -> dict[str, Any]:
    manifest_path = out / UPLOAD_MANIFEST
    keep_manifest(manifest_path)
    started = time.perf_counter()
    qids = read_qids(qids_path, QIDS_SHA256)
    spec, old, checks = open_set(SET)
    known = {q.qid for q in scoring.questions_of(old, checks, spec)}
    if not set(qids) <= known:
        raise ArtifactError(f"{len(set(qids) - known)} listed qids not in the question file")
    old_paths = {f"{spec.directory}/{name}": old.verified_path(name) for name in OLD_FILES}
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(qids_path, out / QIDS_FILE)
    upload = {QIDS_FILE: sha256_file(out / QIDS_FILE)}
    old_digests = {name: sha256_file(path) for name, path in old_paths.items()}
    write_bytes(out / UPLOAD_SUMS, sums_bytes(upload))
    write_bytes(out / OLD_SUMS, sums_bytes(old_digests))
    body: dict[str, Any] = {
        "set": SET,
        "system": SYSTEM,
        "qids": len(qids),
        "questions_in_file": len(known),
        "upload_sha256": upload,
        "old_data_sha256": old_digests,
        "bytes": {
            QIDS_FILE: (out / QIDS_FILE).stat().st_size,
            **{name: path.stat().st_size for name, path in old_paths.items()},
        },
        "old_data_note": "read in place from the old repository (OLD_DATA_ROOT), uploaded as is",
        **git_provenance(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digests_checked": checks.records,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_json(manifest_path, body)
    say(f"[{SET}] {len(qids)} qids; upload {upload}; old data {old_digests}")
    return body


# --- `ga1_hotpot rank` --------------------------------------------------------------------------
def read_sums(path: Path) -> dict[str, str]:
    """`sha256sum` output as {relative name: digest}."""
    rows = [line.split(maxsplit=1) for line in path.read_text("utf-8").splitlines() if line]
    return {name.strip().removeprefix("*").removeprefix("./"): digest for digest, name in rows}


def check_pod(pod: Path) -> dict[str, Any]:
    """The pod's files, read only when each matches the pod's `sha256sums.txt`, the driver's
    ranking matches its manifest, the C3 diff is empty and the run used the listed qids and the
    D17 index settings; returns the manifests."""
    sums = read_sums(pod / POD_SUMS)
    for name in (POD_RANKINGS, POD_MANIFEST, POD_SUBSET, POD_DIFF, POD_GL_MANIFEST, POD_INDEX_SUMS):
        if name not in sums:
            raise ArtifactError(f"{name} is not in {POD_SUMS}")
    for name, digest in sums.items():
        if (pod / name).exists() and sha256_file(pod / name) != digest:
            raise ArtifactError(f"{pod / name}: sha256 differs from {POD_SUMS}")
    driver = json.loads((pod / POD_MANIFEST).read_text("utf-8"))
    subset = json.loads((pod / POD_SUBSET).read_text("utf-8"))
    index = json.loads((pod / POD_GL_MANIFEST).read_text("utf-8"))
    if driver["outputs"].get(POD_RANKINGS) != sums[POD_RANKINGS]:
        raise ArtifactError("the driver manifest records another ranking sha256")
    if (pod / POD_DIFF).stat().st_size != 0:
        raise ArtifactError("C3: the driver or index code differs from Phase 02's commits")
    if subset["qids_sha256"] != QIDS_SHA256:
        raise ArtifactError("the pod ran another qid list")
    wanted = (SET, SYSTEM, "vllm", searchr1.MODEL, searchr1.REVISION)
    found = tuple(driver.get(k) for k in ("set", "system", "engine", "model", "revision"))
    if found != wanted:
        raise ArtifactError(f"driver manifest {found}, expected {wanted}")
    off = {k: index.get(k) for k in D17}
    if off != D17:
        raise ArtifactError(f"the G-L rebuild is not D17's: {off}")
    return {"sums": sums, "driver": driver, "subset": subset, "index": index}


def evidence(
    rows: Sequence[Mapping[str, Any]], qids: Sequence[str], unit_ids: set[str]
) -> dict[str, list[str]]:
    """The driver's evidence lists, one per listed qid, with no repeat and only corpus units."""
    out: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row["qid"])
        ranked = [str(u) for u in row["ranked"]]
        if qid in out:
            raise ArtifactError(f"{qid} appears twice")
        if len(set(ranked)) != len(ranked):
            raise ArtifactError(f"{qid}: a unit appears twice in its evidence list")
        if len(ranked) > searchr1.MAX_SEARCHES * searchr1.TOPK:
            raise ArtifactError(f"{qid}: {len(ranked)} units, more than the driver can retrieve")
        if not set(ranked) <= unit_ids:
            raise ArtifactError(f"{qid}: evidence ids that are not corpus units")
        out[qid] = ranked
    if set(out) != set(qids):
        raise ArtifactError(f"{len(set(qids) ^ set(out))} questions differ from the qid list")
    return out


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def corpus_unit_ids() -> set[str]:
    spec, old, _checks = open_set(SET)
    return {u.unit_id for u in iter_units(old, spec)}


def run_rank(
    bundle: Path,
    pod: Path,
    rankings_dir: Path = config.PHASE06_RANKINGS_DIR,
    unit_ids: set[str] | None = None,
    say: Say = print,
) -> dict[str, Any]:
    directory = rankings_dir / SET
    manifest_path = directory / f"{SYSTEM}.manifest.json"
    keep_manifest(manifest_path)
    qids = read_qids(bundle / QIDS_FILE, QIDS_SHA256)
    pod_files = check_pod(pod)
    rows = read_rows(pod / POD_RANKINGS)
    lists = evidence(rows, qids, corpus_unit_ids() if unit_ids is None else unit_ids)
    order = [str(r["qid"]) for r in rows]  # the driver's: question file order
    name = f"{SYSTEM}.jsonl.gz"
    digest = scoring.write_rankings(directory / name, ((qid, lists[qid]) for qid in order))
    driver, subset, index = pod_files["driver"], pod_files["subset"], pod_files["index"]
    lengths = [len(lists[qid]) for qid in order]
    body: dict[str, Any] = {
        "system": SYSTEM,
        "set": SET,
        "questions": len(order),
        "qids_sha256": QIDS_SHA256,
        "order": "question file order; each list is the driver's evidence ranking: retrieved "
        "units in retrieval order, later duplicates dropped (Phase 02 `dedup_in_order`)",
        "evidence_units": {
            "mean": round(statistics.fmean(lengths), 3),
            "min": min(lengths),
            "max": max(lengths),
        },
        "model": driver["model"],
        "revision": driver["revision"],
        "settings": {
            **driver["settings"],
            "gpu_memory_utilization": subset["gpu_memory_utilization"],
        },
        "status_counts": driver["status_counts"],
        "searches_total": driver["searches_total"],
        "tokens_generated": driver["tokens_generated"],
        "answer_em": driver["answer_em"],
        "searched_g_l_index": {
            "settings": {k: index[k] for k in D17},
            "seconds": index["seconds"],
            "git_commit": index["git_commit"],
            "manifest_sha256": pod_files["sums"][POD_GL_MANIFEST],
            "index_files_sha256": pod_files["sums"][POD_INDEX_SUMS],
            "note": "rebuilt for this run (spec: a new, separately digested artifact); "
            "not the reported Phase 02 session 6 build",
        },
        "pod": {k: driver.get(k) for k in ("gpu", "cost_per_hr_usd", "git_commit")},
        "offline_seconds": index["seconds"].get("encode", 0.0) + index["seconds"].get("index", 0.0),
        "online_seconds": driver["seconds"]["run"],
        "online_seconds_per_question": driver["seconds_per_question"],
        "check": {
            "c3_diff_sha256": pod_files["sums"][POD_DIFF],
            "c3_diff_bytes": 0,
            "driver_sha256": subset["driver_sha256"],
        },
        **git_provenance(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_sha256": {
            QIDS_FILE: QIDS_SHA256,
            **{n: pod_files["sums"][n] for n in (POD_RANKINGS, POD_MANIFEST, POD_SUBSET)},
            POD_SUMS: sha256_file(pod / POD_SUMS),
        },
        "outputs_sha256": {name: digest},
    }
    write_json(manifest_path, body)
    say(f"[{SET}] {len(order)} evidence lists; {name} {digest}")
    return body


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m edge_rag.ga1_hotpot", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="write the HotpotQA 1,000 upload bundle")
    export.add_argument("--out", type=Path, default=config.PHASE06_DIR / "ga1")
    ranked = sub.add_parser("rank", help="convert the pod's evidence lists to rankings")
    ranked.add_argument("--bundle", type=Path, default=config.PHASE06_DIR / "ga1")
    ranked.add_argument("--pod", type=Path, default=config.PHASE06_DIR / "ga1" / "pod")
    ranked.add_argument("--rankings", type=Path, default=config.PHASE06_RANKINGS_DIR)
    args = parser.parse_args(argv)
    if args.command == "export":
        run_export(args.out)
    else:
        run_rank(args.bundle, args.pod, args.rankings)


if __name__ == "__main__":
    main()
