"""Phase 07 laptop rankings on QASPER (increment 5, laptop half).

Pooled (every question against every unit of the split, query `<paper title>. <question>`,
spec D8): Dense, BM25, `p10-b`, the entity hop and `p14`, as `reproduce.py` and `pool.py`
build them on the terrain; within-paper control (the question alone, only its own paper's
units): Dense and BM25. RRF3's laptop inputs are the pooled Dense and BM25 lists; RRF4 needs
G-L from the pod and is fused by `rrf4_lists` once G-L is here. The hop reads GLiNER records
written by the copied `local_extraction.py` (plan D6). Ranking never reads `gold.json`.

Each system writes `<name>.jsonl.gz` once (`scoring.write_rankings`) and then its completion
marker `<name>.complete.json` (sha256, question count, seconds); a system without a marker,
or whose marker does not match its file, is not complete (plan D9).
"""

import hashlib
import json
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edge_rag import config, scoring
from edge_rag.artifacts import ArtifactError, write_json
from edge_rag.old_nodes import ExtractionRecord, build_node_index
from edge_rag.pool import rrf4
from edge_rag.qasper import QasperQuestion, Unit
from edge_rag.retrieval.dense_bm25 import BM25, Dense, Hit
from edge_rag.retrieval.fusion import fuse_lists
from edge_rag.retrieval.hop import Hops

POOLED = ("dense", "bm25", "p10-b", "hop", "p14")
ENTITY = ("hop", "p14")  # the systems that read GLiNER records
WITHIN = ("dense", "bm25")
MARKER = "{name}.complete.json"
Encode = Callable[[Sequence[str]], np.ndarray]


def read_split(directory: Path) -> tuple[list[Unit], list[QasperQuestion]]:
    """`units.jsonl` and `questions.jsonl` as written by `qasper.write`; never `gold.json`."""
    units = [
        Unit(r["unit_id"], r["paper_id"], r["title"], r["body"])
        for r in map(json.loads, (directory / "units.jsonl").read_text("utf-8").splitlines())
    ]
    questions = [
        QasperQuestion(r["qid"], r["paper_id"], r["question"], r["pooled_query"])
        for r in map(json.loads, (directory / "questions.jsonl").read_text("utf-8").splitlines())
    ]
    return units, questions


def bge_encoder(batch_size: int = 32) -> Encode:
    """BGE-small at the pinned revision, CLS pooling and L2 norm (as `components.py`)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION
    )
    model = AutoModel.from_pretrained(config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION)
    model.eval()

    def encode(texts: Sequence[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                list(texts[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.inference_mode():
                cls = model(**batch).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).numpy())
        return np.concatenate(out).astype(np.float32)

    return encode


def ids(hits: Sequence[Hit]) -> list[str]:
    return [unit_id for unit_id, _ in hits]


def pooled_rankings(
    units: Sequence[Unit],
    questions: Sequence[QasperQuestion],
    unit_vectors: np.ndarray,
    query_vectors: np.ndarray,
    records: Mapping[str, ExtractionRecord] | None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, float]]:
    """Every pooled laptop system, as `reproduce.py` (P10-B, P14) and `pool.py` (hop);
    without records the entity systems are left out."""
    depth = config.DEPTH
    unit_ids = [u.unit_id for u in units]
    started = time.perf_counter()
    dense = Dense(unit_vectors, unit_ids)
    bm25 = BM25([u.text for u in units], unit_ids)
    hops = None if records is None else Hops(build_node_index(records, unit_ids), unit_vectors)
    names = [name for name in POOLED if hops is not None or name not in ENTITY]
    seconds = {"index": time.perf_counter() - started, **dict.fromkeys(names, 0.0)}
    out: dict[str, dict[str, list[str]]] = {name: {} for name in names}
    for question, vector in zip(questions, query_vectors, strict=True):
        qid = question.qid
        tick = time.perf_counter()
        first = dense.retrieve(vector, depth)
        seconds["dense"] += time.perf_counter() - tick
        tick = time.perf_counter()
        second = bm25.retrieve(question.pooled_query, depth)
        seconds["bm25"] += time.perf_counter() - tick
        tick = time.perf_counter()
        fused = fuse_lists([first, second], config.WEIGHTS_P10B, top_k=depth)
        seconds["p10-b"] += time.perf_counter() - tick
        out["dense"][qid], out["bm25"][qid], out["p10-b"][qid] = ids(first), ids(second), ids(fused)
        if hops is None:
            continue
        tick = time.perf_counter()
        hop = hops.entity_hop(first, depth, seed="first-entity")  # spec pre-download change 3
        seconds["hop"] += time.perf_counter() - tick
        tick = time.perf_counter()
        relevance = hops.relevance_hop(first, vector, depth, config.P14_ALPHA)
        p14 = fuse_lists([first, second, relevance], config.WEIGHTS_TRIPLE, top_k=depth)
        seconds["p14"] += time.perf_counter() - tick
        out["hop"][qid], out["p14"][qid] = ids(hop), ids(p14)
    return out, seconds


def within_rankings(
    units: Sequence[Unit],
    questions: Sequence[QasperQuestion],
    unit_vectors: np.ndarray,
    question_vectors: np.ndarray,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, float]]:
    """Dense and BM25 over the question's own paper only, with the question alone."""
    depth = config.DEPTH
    rows: dict[str, list[int]] = {}
    for row, unit in enumerate(units):
        rows.setdefault(unit.paper_id, []).append(row)
    out: dict[str, dict[str, list[str]]] = {name: {} for name in WITHIN}
    seconds = {"index": 0.0, **dict.fromkeys(WITHIN, 0.0)}
    retrievers: dict[str, tuple[Dense, BM25]] = {}
    for question, vector in zip(questions, question_vectors, strict=True):
        paper = question.paper_id
        if paper not in retrievers:
            tick = time.perf_counter()
            mine = rows.get(paper, [])
            if not mine:
                raise ArtifactError(f"paper {paper} of question {question.qid} has no units")
            paper_ids = [units[r].unit_id for r in mine]
            retrievers[paper] = (
                Dense(unit_vectors[mine], paper_ids),
                BM25([units[r].text for r in mine], paper_ids),
            )
            seconds["index"] += time.perf_counter() - tick
        dense, bm25 = retrievers[paper]
        tick = time.perf_counter()
        out["dense"][question.qid] = ids(dense.retrieve(vector, depth))
        seconds["dense"] += time.perf_counter() - tick
        tick = time.perf_counter()
        out["bm25"][question.qid] = ids(bm25.retrieve(question.question, depth))
        seconds["bm25"] += time.perf_counter() - tick
    return out, seconds


def rrf4_lists(
    dense: Mapping[str, Sequence[str]],
    bm25: Mapping[str, Sequence[str]],
    gl: Mapping[str, Sequence[str]],
    hop: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """`pool.rrf4` per question, inputs in the terrain order Dense, BM25, G-L, then the hop."""
    if not (dense.keys() == bm25.keys() == gl.keys() == hop.keys()):
        raise ArtifactError("RRF4 inputs do not rank the same questions")
    return {qid: rrf4([dense[qid], bm25[qid], gl[qid]], hop[qid]) for qid in dense}


def write_system(
    directory: Path,
    name: str,
    rankings: Mapping[str, Sequence[str]],
    qids: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The ranking file once, then its completion marker; every qid must be ranked."""
    if set(rankings) != set(qids):
        raise ArtifactError(f"{name}: {len(rankings)} rankings for {len(qids)} questions")
    sha256 = scoring.write_rankings(
        directory / f"{name}.jsonl.gz", ((qid, rankings[qid]) for qid in qids)
    )
    marker = {
        "system": name,
        "questions": len(qids),
        "sha256": sha256,
        "hardware": f"laptop CPU ({platform.machine()}, {platform.system()})",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
    }
    write_json(directory / MARKER.format(name=name), marker)
    return marker


def is_complete(directory: Path, name: str, questions: int) -> bool:
    """A marker that names this file's sha256 and the full question count (plan D9)."""
    marker_path = directory / MARKER.format(name=name)
    ranking_path = directory / f"{name}.jsonl.gz"
    if not (marker_path.exists() and ranking_path.exists()):
        return False
    marker = json.loads(marker_path.read_text("utf-8"))
    sha256 = hashlib.sha256(ranking_path.read_bytes()).hexdigest()
    return marker.get("sha256") == sha256 and marker.get("questions") == questions


def run(
    split_dir: Path,
    out: Path,
    records: Mapping[str, ExtractionRecord] | None,
    encode: Encode,
    *,
    papers: Sequence[str] | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Every laptop system on one split; `papers` limits it to a smoke subset. Without GLiNER
    records (`None`) the entity systems (`ENTITY`) are not written, the others are."""
    units, questions = read_split(split_dir)
    if papers is not None:
        keep = set(papers)
        units = [u for u in units if u.paper_id in keep]
        questions = [q for q in questions if q.paper_id in keep]
    qids = [q.qid for q in questions]
    seconds: dict[str, float] = {}
    tick = time.perf_counter()
    unit_vectors = encode([u.text for u in units])
    seconds["encode_units"] = time.perf_counter() - tick
    tick = time.perf_counter()
    pooled_vectors = encode([q.pooled_query for q in questions])
    question_vectors = encode([q.question for q in questions])
    seconds["encode_queries"] = time.perf_counter() - tick
    say(f"[OK] encoded {len(units)} units and {2 * len(questions)} queries")
    pooled, pooled_seconds = pooled_rankings(
        units, questions, unit_vectors, pooled_vectors, records
    )
    within, within_seconds = within_rankings(units, questions, unit_vectors, question_vectors)
    for setting, lists, timing in (
        ("pooled", pooled, pooled_seconds),
        ("within", within, within_seconds),
    ):
        for name, rankings in lists.items():
            write_system(
                out / setting,
                name,
                rankings,
                qids,
                {
                    "online_seconds": round(timing[name], 3),
                    "index_seconds": round(timing["index"], 3),
                },
            )
            say(f"[OK] {setting}/{name}: {len(rankings)} rankings")
    return {"units": len(units), "questions": len(questions), "seconds": seconds}


def first_papers(split_dir: Path, n: int) -> list[str]:
    """The first `n` paper ids in unit order, for a smoke run."""
    units, _ = read_split(split_dir)
    return list(dict.fromkeys(u.paper_id for u in units))[:n]


def gliner_pass(
    split_dir: Path,
    out: Path,
    model_dir: Path,
    *,
    papers: Sequence[str] | None = None,
    limit_units: int | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The copied `local_extraction.py` over QASPER units (`title. body`, one sentence each);
    writes `records.jsonl.gz` and `extraction.json`, the archive `run` reads back.
    `limit_units` keeps the first units in file order (the pod probe, spec)."""
    from edge_rag import local_extraction
    from edge_rag.old_nodes import IndexingUnit

    units, _ = read_split(split_dir)
    if papers is not None:
        keep = set(papers)
        units = [u for u in units if u.paper_id in keep]
    if limit_units is not None:
        units = units[:limit_units]
    extractor, load_seconds = local_extraction.gliner_extractor(model_dir)
    pool = [IndexingUnit(u.unit_id, u.title, (u.body,)) for u in units]
    records, seconds = local_extraction.run_extraction(pool, extractor)
    manifest = local_extraction.build_manifest(
        records,
        extractor,
        seconds=seconds,
        hardware=local_extraction.hardware_block(extractor.hardware_device),
        model_load_seconds=load_seconds,
    )
    local_extraction.write_extraction(records, manifest, out)
    say(f"[OK] GLiNER {len(records)} units in {seconds:.1f} s (load {load_seconds:.1f} s)")
    return {"units": len(records), "seconds": seconds, "load_seconds": load_seconds}
