"""Phase 07 pod runners on QASPER (plan increment 5c): thin adapters that feed the split's
`units.jsonl` and `questions.jsonl` (as `qasper.write` wrote them; `gold.json` is never read)
to the Phase 02-06 pod modules in the shapes they already take.

- `g-l`: `colbert.load_model`, PyLate's PLAID with default settings and `colbert.spread_chunks`
  (the D17 build: a spread first chunk, then `update`), depth 100, D8 pooled query.
- `rerank`: J-strong (`rerank.load_strong`, `strong_scores`) or G-R2 (`rerank.QwenReranker`)
  over the top 100 of a first stage (G-L, or RRF3 / RRF4 fused here with `retrieval.rrf.rrf`
  as `fuse.py` and `pool.rrf4` do), D8 pooled query, order by `rerank.reorder` (ties by the
  first-stage rank, as `rivals.rank`).
- `within`: J-strong over every unit of the question's own paper, the question alone (D4, D8),
  ties by unit file order.
- `serve` and `ga1`: `colbert.serve` over the QASPER G-L index and `searchr1.main` unchanged,
  with `common.open_set`, `units_by_id`, `set_questions` and `PHASE_DIR` replaced for the run
  (the `ga1_subset` pattern); the question Search-R1 reads is the D8 pooled query.
Top-level imports are the standard library and `common`, so it runs in both pod environments.

    uv run --group pod python -m edge_rag.pod.exam_qasper rerank <split> <out> --first g-l=...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from edge_rag.artifacts import ArtifactError, sha256_file, write_json
from edge_rag.corpus import Question
from edge_rag.pod import common

DEPTH = 100
RRF_K = 60  # config.RRF_K, the terrain's k
Score = Callable[[Sequence[tuple[str, str]]], Any]  # floats aligned with the pairs


def read_units(split_dir: Path, limit: int | None = None) -> list[tuple[str, common.Unit]]:
    """(paper_id, unit) in file order; the unit's `text` is the harness `title. body`."""
    rows = (split_dir / "units.jsonl").read_text("utf-8").splitlines()[:limit]
    return [
        (r["paper_id"], common.Unit(r["unit_id"], r["title"], r["body"]))
        for r in map(json.loads, rows)
    ]


def read_questions(split_dir: Path, limit: int | None = None) -> list[dict[str, str]]:
    """qid, paper_id, question and the D8 `pooled_query`, in file order."""
    rows = (split_dir / "questions.jsonl").read_text("utf-8").splitlines()[:limit]
    return [json.loads(r) for r in rows]


def as_driver_questions(questions: Sequence[Mapping[str, str]]) -> list[Question]:
    """The `corpus.Question` shape the Phase 02 drivers read: the D8 query as the question,
    no gold (the exam's gold stays on the laptop)."""
    return [Question(q["qid"], q["pooled_query"], (), "qasper") for q in questions]


def fused_first_stage(kind: str, lists: Mapping[str, Mapping[str, Sequence[str]]]) -> dict:
    """RRF3 (Dense, BM25, G-L) or RRF4 (those, then the entity hop), top 100 per question."""
    from edge_rag.retrieval.rrf import rrf

    names = {"rrf3": ("dense", "bm25", "g-l"), "rrf4": ("dense", "bm25", "g-l", "hop")}[kind]
    qids = lists[names[0]].keys()
    if any(lists[n].keys() != qids for n in names):
        raise ArtifactError(f"{kind}: inputs do not rank the same questions")
    return {qid: rrf([lists[n][qid] for n in names], RRF_K, DEPTH) for qid in qids}


def rerank_lists(
    jobs: Sequence[tuple[str, str, Sequence[str]]],
    texts: Mapping[str, str],
    score: Score,
    shard: int = 250,
) -> dict[str, list[str]]:
    """jobs: (qid, query, candidates in first-stage order); each list reordered by score
    descending, ties by first-stage position (`rerank.reorder`)."""
    from edge_rag.pod.rerank import reorder

    out: dict[str, list[str]] = {}
    for start in range(0, len(jobs), shard):
        part = jobs[start : start + shard]
        scores = score([(query, texts[u]) for _qid, query, top in part for u in top])
        offset = 0
        for qid, _query, top in part:
            out[qid] = reorder(top, scores[offset : offset + len(top)])
            offset += len(top)
        print(f"[INFO] {start + len(part)} questions reranked", flush=True)
    return out


def pooled_jobs(
    questions: Sequence[Mapping[str, str]], tops: Mapping[str, Sequence[str]]
) -> list[tuple[str, str, Sequence[str]]]:
    missing = [q["qid"] for q in questions if q["qid"] not in tops]
    if missing:
        raise ArtifactError(f"first stage misses {len(missing)} questions")
    return [(q["qid"], q["pooled_query"], tops[q["qid"]][:DEPTH]) for q in questions]


def within_jobs(
    questions: Sequence[Mapping[str, str]], units: Sequence[tuple[str, common.Unit]]
) -> list[tuple[str, str, Sequence[str]]]:
    """Every unit of the own paper, in file order; the question alone (no title, D8)."""
    by_paper: dict[str, list[str]] = {}
    for paper_id, unit in units:
        by_paper.setdefault(paper_id, []).append(unit.unit_id)
    return [(q["qid"], q["question"], by_paper.get(q["paper_id"], [])) for q in questions]


def scorer(qwen: bool) -> tuple[Score, dict[str, Any]]:
    from edge_rag.pod import rerank

    if qwen:
        model = rerank.QwenReranker()
        return model.predict, {"model": rerank.QWEN_MODEL, "revision": rerank.QWEN_REVISION}
    strong = rerank.load_strong()
    return (
        lambda pairs: rerank.strong_scores(strong, pairs),
        {"model": rerank.MODEL, "revision": rerank.REVISION},
    )


def write_system(out: Path, name: str, rankings: Mapping[str, Sequence[str]], qids, body) -> None:
    path = out / f"{name}.jsonl.gz"
    sha = common.write_rankings(path, [(qid, rankings[qid]) for qid in qids])
    write_json(out / f"{name}.manifest.json", {"system": name, "sha256": sha, **body})
    print(f"[OK] {name}: {len(qids)} rankings", flush=True)


def run_rerank(args: argparse.Namespace) -> None:
    questions = read_questions(args.split_dir, args.limit_questions)
    qids = [q["qid"] for q in questions]
    lists = {name: dict(common.read_rankings(Path(p))) for name, p in args.first}
    units = read_units(args.split_dir)
    texts = {u.unit_id: u.text for _paper, u in units}
    started = time.perf_counter()
    score, model = scorer(args.qwen)
    load_seconds = time.perf_counter() - started
    for system in args.systems:
        started = time.perf_counter()
        if system == "within-j-strong":
            jobs = within_jobs(questions, units)
            source: Any = "own paper, all units"
        else:
            first = system.split("-", 1)[1] if system.startswith("j-") else "g-l"
            tops = lists["g-l"] if first == "g-l" else fused_first_stage(first, lists)
            jobs = pooled_jobs(questions, tops)
            source = {n: sha256_file(Path(p)) for n, p in args.first}
        rankings = rerank_lists(jobs, texts, score)
        body = {
            **model,
            "first_stage": source,
            "depth": DEPTH,
            "questions": len(qids),
            "pairs": sum(len(j[2]) for j in jobs),
            "load_seconds": round(load_seconds, 3),
            "seconds": round(time.perf_counter() - started, 3),
            "gpu": common.gpu_name(),
        }
        write_system(args.out, system, rankings, qids, body)


def plaid(folder: Path, *, override: bool) -> Any:
    from pylate import indexes

    from edge_rag.pod.colbert import SYSTEM

    return indexes.PLAID(index_folder=str(folder), index_name=SYSTEM, override=override)


def run_gl(args: argparse.Namespace) -> None:
    from pylate import retrieve

    from edge_rag.pod import colbert

    units = [u for _paper, u in read_units(args.split_dir, args.limit_units)]
    questions = read_questions(args.split_dir)
    started = time.perf_counter()
    model = colbert.load_model()
    index = plaid(args.index, override=True)
    stride = -(-len(units) // args.chunk_units)
    for chunk in colbert.spread_chunks(lambda: iter(units), stride, args.update_units):
        embeddings = [
            e.astype("float16")
            for e in model.encode(
                [u.text for u in chunk], batch_size=32, is_query=False, show_progress_bar=False
            )
        ]
        index.add_documents(
            documents_ids=[u.unit_id for u in chunk], documents_embeddings=embeddings
        )
    queries = model.encode(
        [q["pooled_query"] for q in questions],
        batch_size=32,
        is_query=True,
        show_progress_bar=False,
    )
    hits = retrieve.ColBERT(index=index).retrieve(queries_embeddings=queries, k=colbert.DEPTH)
    rankings = {q["qid"]: [str(h["id"]) for h in hs] for q, hs in zip(questions, hits, strict=True)}
    body = {
        "model": colbert.MODEL,
        "revision": colbert.REVISION,
        "index": "PLAID default",
        "first_chunk_stride": stride,
        "units": len(units),
        "query": "D8 pooled",
        "depth": colbert.DEPTH,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_system(args.out, colbert.SYSTEM, rankings, [q["qid"] for q in questions], body)


def patch_drivers(split_dir: Path, index: Path, phase_dir: Path) -> None:
    """Point the Phase 02 drivers at QASPER for this process only (`ga1_subset` pattern)."""
    from edge_rag.artifacts import Checks

    questions = as_driver_questions(read_questions(split_dir))
    units = {u.unit_id: u for _paper, u in read_units(split_dir)}
    common.config.SETS = {**common.config.SETS, "qasper": None}  # type: ignore[dict-item]
    common.PHASE_DIR = phase_dir
    common.open_set = lambda _name: (None, None, Checks())  # type: ignore[assignment,return-value]
    common.units_by_id = lambda _old, _spec, wanted: (  # type: ignore[assignment]
        units if wanted is None else {u: units[u] for u in wanted}
    )
    common.set_questions = lambda *_a: (questions, {q.qid: "" for q in questions})  # type: ignore[assignment]
    from edge_rag.pod import colbert  # PyLate is imported only when an index opens

    colbert.open_index = lambda set_name, tag, *, override: plaid(index, override=override)


def run_driver(args: argparse.Namespace) -> None:
    patch_drivers(args.split_dir, args.index, args.out)
    if args.command == "serve":
        from edge_rag.pod import colbert

        colbert.serve(argparse.Namespace(set="qasper", port=args.port))
        return
    from edge_rag.pod import searchr1

    if args.gpu_memory_utilization is not None:
        searchr1.GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization
    limit = [] if args.limit_questions is None else ["--limit", str(args.limit_questions)]
    sys.argv = [searchr1.__file__ or "searchr1", "--set", "qasper", *limit]
    searchr1.main()


def first_arg(text: str) -> tuple[str, str]:
    name, _, path = text.partition("=")
    return name, path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m edge_rag.pod.exam_qasper")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("g-l", "rerank", "serve", "ga1"):
        sub = commands.add_parser(name)
        sub.add_argument("split_dir", type=Path)
        sub.add_argument("out", type=Path)
        sub.add_argument("--index", type=Path, default=Path("/workspace/qasper-gl-index"))
        sub.add_argument("--limit-units", type=int, default=None)
        sub.add_argument("--limit-questions", type=int, default=None)
        sub.add_argument("--chunk-units", type=int, default=500_000)
        sub.add_argument("--update-units", type=int, default=100_000)
        sub.add_argument("--first", type=first_arg, action="append", default=[])
        sub.add_argument("--systems", nargs="+", default=[])
        sub.add_argument("--qwen", action="store_true")
        sub.add_argument("--port", type=int, default=8000)
        sub.add_argument("--gpu-memory-utilization", type=float, default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    {"g-l": run_gl, "rerank": run_rerank, "serve": run_driver, "ga1": run_driver}[args.command](
        args
    )


if __name__ == "__main__":
    main()
