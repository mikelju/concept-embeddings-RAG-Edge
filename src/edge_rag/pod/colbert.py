"""G-L: answerai-colbert-small-v1 with PyLate's PLAID index, depth-100 rankings (spec scope 3).

Runs in `src/edge_rag/pod/colbert_env` (PyLate pins its own torch):
    uv run --project src/edge_rag/pod/colbert_env python -m edge_rag.pod.colbert --set musique
`--serve` instead answers Search-R1 retrieval requests over a built index (G-A1).

The corpus is encoded and added to the index in chunks of `--chunk-units`; the first chunk
creates the index (its k-means centroids) and is a spread sample of the whole corpus (every
`stride`-th unit), later chunks go through fast-plaid's `update`, which keeps those centroids.
A corpus no larger than one chunk is indexed in one call, PyLate's default; the manifest
records the chunking. Each chunk is encoded `--encode-units` at a time and cast to float16,
which fast-plaid stores anyway, so a whole-corpus chunk fits in host RAM (plan D17).
"""

import argparse
import json
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import islice
from typing import Any

from edge_rag.pod import common

MODEL = "answerdotai/answerai-colbert-small-v1"
# HF Hub `sha` of the model, read 2026-10-02 (api/models/answerdotai/answerai-colbert-small-v1).
REVISION = "934fa8bb4ce2284f4c2baa232d81aca4d076fa5e"
DOCUMENT_LENGTH = 512
QUERY_LENGTH = 32
DEPTH = 100
SYSTEM = "g-l"
SERVE_BATCH = 32


def load_model() -> Any:
    from pylate import models

    return models.ColBERT(
        model_name_or_path=MODEL,
        revision=REVISION,
        document_length=DOCUMENT_LENGTH,
        query_length=QUERY_LENGTH,
    )


def open_index(set_name: str, tag: str | None, *, override: bool) -> Any:
    from pylate import indexes

    folder = common.PHASE_DIR / "index" / (set_name if tag is None else f"{set_name}-{tag}")
    # PyLate's PLAID with every setting at its default (spec scope 3).
    return indexes.PLAID(index_folder=str(folder), index_name=SYSTEM, override=override)


def spread_chunks(
    make_units: Callable[[], Iterator[Any]], stride: int, update_units: int
) -> Iterator[list[Any]]:
    """First chunk: every `stride`-th unit, so the centroids it sets span the whole corpus;
    then the remaining units in corpus order, `update_units` at a time (fast-plaid's update holds
    a whole chunk's embeddings on the GPU: 500,000 HotpotQA units ran out of 24 GB, plan D9).
    stride 1 is a single chunk, PyLate's default."""
    yield [u for i, u in enumerate(make_units()) if i % stride == 0]
    rest = (u for i, u in enumerate(make_units()) if i % stride != 0)
    while chunk := list(islice(rest, update_units)):
        yield chunk


def centroid_update_on_cpu() -> None:
    """Run fast-plaid's `update_centroids` through its own CPU branch (plan D12): on HotpotQA
    its GPU branch ran out of the 4090's 24 GB (F3). Only that step moves; the rest of the
    update and the search stay on the GPU."""
    from fast_plaid.search import update

    original = update.update_centroids

    def on_cpu(**kwargs: Any) -> None:
        return original(**{**kwargs, "device": "cpu"})

    update.update_centroids = on_cpu


def build(args: argparse.Namespace) -> None:
    tag = None if args.limit_units is None else f"units{args.limit_units}"
    out = common.rankings_path(args.set, SYSTEM, tag)
    manifest = common.manifest_path(out)
    if manifest.exists():
        print(f"[INFO] {manifest} exists, nothing to do", flush=True)
        return
    spec, old, checks = common.open_set(args.set)
    questions, _answers = common.set_questions(old, checks, spec)
    if args.limit_questions is not None:
        questions = questions[: args.limit_questions]
    if args.cpu_centroid_update:
        centroid_update_on_cpu()
    timer = common.Timer()
    started = time.perf_counter()
    model = load_model()
    timer.add("load_model", started)
    index = open_index(args.set, tag, override=True)
    n_units = sum(1 for _ in common.iter_units(old, spec, limit=args.limit_units))
    stride = -(-n_units // args.chunk_units)
    total, chunks = 0, 0
    for chunk in spread_chunks(
        lambda: common.iter_units(old, spec, limit=args.limit_units), stride, args.update_units
    ):
        started = time.perf_counter()
        embeddings = []
        for start in range(0, len(chunk), args.encode_units):
            embeddings += [
                e.astype("float16")
                for e in model.encode(
                    [u.text for u in chunk[start : start + args.encode_units]],
                    batch_size=args.batch_size,
                    is_query=False,
                    show_progress_bar=False,
                )
            ]
            print(f"[INFO] {args.set}: {total + len(embeddings)} units encoded", flush=True)
        timer.add("encode", started)
        started = time.perf_counter()
        index.add_documents(
            documents_ids=[u.unit_id for u in chunk], documents_embeddings=embeddings
        )
        timer.add("index", started)
        total += len(chunk)
        chunks += 1
        del embeddings
        print(f"[INFO] {args.set}: {total} units encoded and indexed", flush=True)
    from pylate import retrieve

    started = time.perf_counter()
    queries = model.encode(
        [q.question for q in questions],
        batch_size=args.batch_size,
        is_query=True,
        show_progress_bar=False,
    )
    results = retrieve.ColBERT(index=index).retrieve(queries_embeddings=queries, k=DEPTH)
    timer.add("search", started)
    rankings = [
        (q.qid, [str(hit["id"]) for hit in hits])
        for q, hits in zip(questions, results, strict=True)
    ]
    common.write_rankings(out, rankings)
    seconds = timer.seconds
    common.write_manifest(
        manifest,
        {
            "system": SYSTEM,
            "set": args.set,
            "model": MODEL,
            "revision": REVISION,
            "document_length": DOCUMENT_LENGTH,
            "query_length": QUERY_LENGTH,
            "index": "pylate.indexes.PLAID, default settings",
            "index_chunks": chunks,
            "chunk_units": args.chunk_units,
            "encode_units": args.encode_units,
            "embeddings_dtype": "float16",
            "update_units": args.update_units,
            "cpu_centroid_update": args.cpu_centroid_update,
            "first_chunk_stride": stride,
            "encode_batch_size": args.batch_size,
            "depth": DEPTH,
            "units": total,
            "questions": len(questions),
            "limit_units": args.limit_units,
            "seconds": seconds,
            "search_seconds_per_question": seconds["search"] / max(len(questions), 1),
            "checks": checks.records,
        },
        [out],
    )
    print(f"[INFO] wrote {out} and {manifest}", flush=True)


def serve(args: argparse.Namespace) -> None:
    """Search-R1's retrieval API (`POST /retrieve`) over a built G-L index: `contents` is
    `title + "\\n" + text`, which `infer.py` splits back into title and text."""
    spec, old, _checks = common.open_set(args.set)
    units = common.units_by_id(old, spec, None)
    model = load_model()
    index = open_index(args.set, None, override=False)
    from pylate import retrieve

    retriever = retrieve.ColBERT(index=index)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            self._reply({"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            queries = [str(q) for q in request["queries"]]
            hits = []
            # A G-A1 turn sends every active question's search at once; PLAID on the CPU over
            # 2,255 queries outgrew the pod's 62 GB and the server was killed (plan D14).
            for start in range(0, len(queries), SERVE_BATCH):
                embeddings = model.encode(
                    queries[start : start + SERVE_BATCH], is_query=True, show_progress_bar=False
                )
                hits += retriever.retrieve(queries_embeddings=embeddings, k=int(request["topk"]))
            result = [
                [
                    {
                        "document": {
                            "id": str(h["id"]),
                            "contents": f"{units[str(h['id'])].title}\n{units[str(h['id'])].body}",
                        },
                        "score": float(h["score"]),
                    }
                    for h in query_hits
                ]
                for query_hits in hits
            ]
            self._reply({"result": result})

        def _reply(self, body: dict) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            return

    print(f"[INFO] serving {args.set} on 127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, choices=sorted(common.config.SETS))
    parser.add_argument("--limit-units", type=int, default=None, help="probe: first N units")
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--chunk-units", type=int, default=500_000)
    parser.add_argument("--update-units", type=int, default=100_000)
    parser.add_argument("--encode-units", type=int, default=200_000)
    parser.add_argument(
        "--cpu-centroid-update", action="store_true", help="fast-plaid's CPU update_centroids"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="PyLate's encode default")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(args) if args.serve else build(args)


if __name__ == "__main__":
    main()
