"""Phase 06 G-A2 on the pod: HippoRAG 2 over the MultiHop-RAG bundle (spec C3, C4).

Runs inside HippoRAG's own environment (pinned commit, Python 3.10+), so only the standard
library and `hipporag` are imported, and it is run as a script, not as `edge_rag.pod...`.
The HippoRAG flow is the README's: `HippoRAG(...)`, `index(docs)`, `retrieve(queries,
num_to_retrieve)`, every `BaseConfig` field at its default except the model names, the vLLM
endpoint and `save_dir`; the depth is the `retrieve` argument (100). Our code is the adapter:
bundle in (unit texts as plain-string docs, questions as queries), passages out as groups of
unit ids (units with identical text are one passage to HippoRAG, which keys passages by
content hash).

Resumable: HippoRAG keeps every LLM response in `save_dir/llm_cache/*.sqlite` and its stores
under `save_dir`, so a rerun replays finished OpenIE calls from the cache; an `index.done`
marker skips indexing once it finished; retrieval appends batches of questions to a partial
file and a rerun skips the questions it holds. Token counts are read from the LLM cache: rows
written up to the end of indexing are offline, later rows online (a cache hit is not counted
twice, as it costs no LLM call).

A recognition-memory (fact filter) error makes HippoRAG fall back to dense retrieval silently;
this adapter counts those log records and stops the run at the first one, since the result
would no longer be HippoRAG 2.

    python src/edge_rag/pod/hipporag_run.py --bundle data/phase06/ga2 \
        --save-dir /workspace/hipporag_save --out data/phase06/ga2/pod \
        --llm-base-url http://localhost:6578/v1 --hipporag-dir /workspace/HippoRAG \
        --diff data/phase06/ga2/pod/c3-g-a2.diff
"""

import argparse
import dataclasses
import gzip
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LLM_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
EMBEDDING_NAME = "GritLM/GritLM-7B"
EMBEDDING_REVISION = "cb7f7ffb99c0c24ca2c325d798d7ef5c455d5339"
HIPPORAG_COMMIT = "2bfd831417202b49cda9da7973e141a456e16872"
DEPTH = 100
BATCH = 100
UNITS_FILE = "units.jsonl.gz"
QUESTIONS_FILE = "questions.jsonl.gz"
SUMS_FILE = "bundle.sha256"
RANKINGS = "hipporag.jsonl.gz"
MANIFEST = "hipporag.manifest.json"
PARTIAL = "hipporag.partial.jsonl"
TIMING = "hipporag.timing.json"
INDEX_DONE = "index.done"
FILTER_ERROR = "Error in rerank_facts"
DPR_FALLBACK = "No facts found after reranking"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def check_bundle(bundle: Path) -> dict[str, str]:
    """The bundle files against `bundle.sha256` (written by the laptop's export)."""
    recorded = {}
    for line in (bundle / SUMS_FILE).read_text("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    for name, digest in recorded.items():
        if sha256_file(bundle / name) != digest:
            raise SystemExit(f"{bundle / name}: sha256 differs from {SUMS_FILE}")
    return recorded


def passage_groups(units: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Unit ids per exact text, in corpus order."""
    groups: dict[str, list[str]] = {}
    for unit in units:
        groups.setdefault(unit["text"], []).append(unit["unit_id"])
    return groups


def to_row(qid: str, docs: Sequence[str], scores: Sequence[float], groups: dict[str, list[str]]):
    """One output line: the retrieved passages as unit-id groups, in HippoRAG's order."""
    missing = [doc for doc in docs if doc not in groups]
    if missing:
        raise SystemExit(f"{qid}: {len(missing)} retrieved passages match no unit text")
    return {
        "qid": qid,
        "passages": [groups[doc] for doc in docs],
        "scores": [float(s) for s in scores],
    }


def cache_tokens(cache: Path, after_rowid: int = 0, upto_rowid: int | None = None):
    """Calls and token sums from HippoRAG's LLM cache rows in (after_rowid, upto_rowid]."""
    out = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if not cache.exists():
        return out
    query = "SELECT metadata FROM cache WHERE rowid > ? AND rowid <= ?"
    params = (after_rowid, (1 << 62) if upto_rowid is None else upto_rowid)
    with sqlite3.connect(cache) as conn:
        for (metadata,) in conn.execute(query, params):
            meta = json.loads(metadata) or {}
            out["calls"] += 1
            out["prompt_tokens"] += int(meta.get("prompt_tokens") or 0)
            out["completion_tokens"] += int(meta.get("completion_tokens") or 0)
    return out


def max_rowid(cache: Path) -> int:
    if not cache.exists():
        return 0
    with sqlite3.connect(cache) as conn:
        try:
            return int(conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM cache").fetchone()[0])
        except sqlite3.OperationalError:
            return 0


class Counter(logging.Handler):
    """Counts HippoRAG's filter errors and dense-retrieval fallbacks from its log records."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.filter_errors = 0
        self.dpr_fallbacks = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith(FILTER_ERROR):
            self.filter_errors += 1
        elif message.startswith(DPR_FALLBACK):
            self.dpr_fallbacks += 1


def load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text("utf-8")) if path.exists() else default


def write_json(path: Path, body: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, sort_keys=True, default=str), "utf-8")
    tmp.replace(path)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gpu_name() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git(directory: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True)  # noqa: S603, S607
    return out.stdout.strip()


def endpoint_up(base_url: str) -> None:
    """Fail fast when the vLLM server is not serving the pinned model name."""
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=30) as response:  # noqa: S310
        served = [m["id"] for m in json.loads(response.read())["data"]]
    if LLM_NAME not in served:
        raise SystemExit(f"vLLM serves {served}, not {LLM_NAME}")


def settings(config: Any) -> dict[str, Any]:
    """The full BaseConfig, minus fields that could hold credentials."""
    body = dataclasses.asdict(config)
    return {k: v for k, v in body.items() if "key" not in k.lower() and "secret" not in k.lower()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--hipporag-dir", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True, help="the recorded C3 diff file")
    parser.add_argument("--limit-units", type=int, help="probe: index only the first N units")
    parser.add_argument("--limit-questions", type=int, help="probe: retrieve only the first M")
    parser.add_argument(
        "--force-index-from-scratch",
        action="store_true",
        help="HippoRAG's flag, needed only to rebuild after a crash mid-graph",
    )
    args = parser.parse_args(argv)

    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    args.out.mkdir(parents=True, exist_ok=True)
    inputs = check_bundle(args.bundle)
    units = read_jsonl_gz(args.bundle / UNITS_FILE)[: args.limit_units]
    questions = read_jsonl_gz(args.bundle / QUESTIONS_FILE)[: args.limit_questions]
    groups = passage_groups(units)
    endpoint_up(args.llm_base_url)

    config = BaseConfig(
        save_dir=str(args.save_dir),
        llm_name=LLM_NAME,
        llm_base_url=args.llm_base_url,
        embedding_model_name=EMBEDDING_NAME,
    )
    if args.force_index_from_scratch:
        config.force_index_from_scratch = True
    counter = Counter()
    hippo_logger = logging.getLogger("hipporag")  # module loggers `hipporag.*` propagate here
    hippo_logger.setLevel(logging.INFO)  # so the info-level fallback records reach the counter
    hippo_logger.addHandler(counter)

    timing = load_json(args.out / TIMING, {"index_attempts": [], "retrieve_batches": []})
    with HippoRAG(global_config=config) as rag:
        cache = Path(rag.llm_model.cache_file_name)
        done = args.out / INDEX_DONE
        if not done.exists():
            attempt = {
                "started_utc": utc(),
                "seconds": None,
                "force_index_from_scratch": args.force_index_from_scratch,
            }
            timing["index_attempts"].append(attempt)
            write_json(args.out / TIMING, timing)
            started = time.perf_counter()
            rag.index(docs=[u["text"] for u in units])
            attempt["seconds"] = round(time.perf_counter() - started, 3)
            timing["index_rowid"] = max_rowid(cache)
            write_json(args.out / TIMING, timing)
            done.write_text(utc(), "utf-8")
            print(f"INDEXED {len(units)} units in {attempt['seconds']} s", flush=True)

        partial = args.out / PARTIAL
        rows = {}
        if partial.exists():
            for line in partial.read_text("utf-8").splitlines():
                row = json.loads(line)
                rows[row["qid"]] = row
        todo = [q for q in questions if q["qid"] not in rows]
        for start in range(0, len(todo), BATCH):
            batch = todo[start : start + BATCH]
            errors_before = counter.filter_errors
            started = time.perf_counter()
            solutions = rag.retrieve(queries=[q["question"] for q in batch], num_to_retrieve=DEPTH)
            seconds = round(time.perf_counter() - started, 3)
            if counter.filter_errors > errors_before:
                raise SystemExit(
                    f"STOP: {counter.filter_errors - errors_before} fact-filter errors; "
                    "HippoRAG fell back to dense retrieval"
                )
            new = [
                to_row(q["qid"], s.docs, s.doc_scores, groups)
                for q, s in zip(batch, solutions, strict=True)
            ]
            with open(partial, "a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row) + "\n" for row in new)
            rows.update((row["qid"], row) for row in new)
            timing["retrieve_batches"].append(
                {
                    "questions": len(batch),
                    "seconds": seconds,
                    "dpr_fallbacks_total": counter.dpr_fallbacks,
                }
            )
            write_json(args.out / TIMING, timing)
            print(
                f"RETRIEVED {len(rows)}/{len(questions)} ({seconds} s for {len(batch)})", flush=True
            )

        body = gzip.compress(
            "".join(json.dumps(rows[q["qid"]]) + "\n" for q in questions).encode("utf-8"), mtime=0
        )
        (args.out / RANKINGS).write_bytes(body)
        index_rowid = timing.get("index_rowid", 0)
        diff = args.diff.read_bytes() if args.diff.exists() else None
        rankings_sha256 = hashlib.sha256(body).hexdigest()
        manifest: dict[str, Any] = {
            "system": "g-a2",
            "set": "multihop-rag",
            "scope": {
                "units": len(units),
                "questions": len(questions),
                "probe": args.limit_units is not None or args.limit_questions is not None,
                "passages": len(groups),
            },
            "models": {
                "llm": {
                    "name": LLM_NAME,
                    "revision": LLM_REVISION,
                    "served_by": "vllm (OpenAI-compatible)",
                },
                "embedding": {"name": EMBEDDING_NAME, "revision": EMBEDDING_REVISION},
            },
            "hipporag": {
                "pinned_commit": HIPPORAG_COMMIT,
                "head": git(args.hipporag_dir, "rev-parse", "HEAD"),
                "status": git(args.hipporag_dir, "status", "--porcelain"),
            },
            "settings": settings(rag.global_config),
            "retrieve": {"num_to_retrieve": DEPTH, "batch_questions": BATCH},
            "seconds": {
                "index": sum(a["seconds"] or 0 for a in timing["index_attempts"]),
                "index_attempts": timing["index_attempts"],
                "retrieve": round(sum(b["seconds"] for b in timing["retrieve_batches"]), 3),
            },
            "llm_tokens": {
                "offline": cache_tokens(cache, 0, index_rowid),
                "online": cache_tokens(cache, index_rowid),
                "source": "HippoRAG LLM cache rows (vLLM usage); cache hits not counted",
            },
            "check": {
                "diff_file": args.diff.name,
                "diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
                "diff_bytes": len(diff) if diff is not None else None,
                "filter_errors": counter.filter_errors,
                "dpr_fallbacks_logged": counter.dpr_fallbacks,
            },
            "gpu": gpu_name(),
            "cost_per_hr_usd": float(os.environ["POD_COST_PER_HR"])
            if os.environ.get("POD_COST_PER_HR")
            else None,
            "python": sys.version.split()[0],
            "started_utc": timing["index_attempts"][0]["started_utc"]
            if timing["index_attempts"]
            else None,
            "written_utc": utc(),
            "inputs_sha256": inputs,
            "outputs": {RANKINGS: rankings_sha256},
        }
        write_json(args.out / MANIFEST, manifest)
        print(
            f"DONE {len(questions)} questions; {RANKINGS} {rankings_sha256}",
            flush=True,
        )


if __name__ == "__main__":
    main()
