"""Phase 03 inputs: Dense and BM25 depth-100 lists with their laptop cost (spec C1, plan D1, D8).

The lists come from the Phase 01 loading path (`reproduce.load_inputs`). Every `dense` list is
compared with the frozen `p10-a` list, and the 0.5 / 0.5 fusion of the same hits with `p10-b`,
so the new BM25 lists are tied to the gated pipeline end to end.
"""

import hashlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edge_rag import config, scoring
from edge_rag.artifacts import ArtifactError, write_json
from edge_rag.corpus import Question
from edge_rag.embeddings import QueryTable
from edge_rag.pod.common import git_commit
from edge_rag.process import peak_rss_mb
from edge_rag.reproduce import Laps, Say, load_inputs
from edge_rag.retrieval.dense_bm25 import BM25, Dense
from edge_rag.retrieval.fusion import fuse_lists

MANIFEST = "components.manifest.json"
NAMES = ("dense", "bm25")


def phase02_list(set_name: str, name: str, sha256: str) -> dict[str, list[str]]:
    """A Phase 02 ranking file, read only after its bytes match the given sha256."""
    path = config.RANKINGS_DIR / set_name / f"{name}.jsonl.gz"
    measured = hashlib.sha256(path.read_bytes()).hexdigest()
    if measured != sha256:
        raise ArtifactError(f"{path}: sha256 {measured}, expected {sha256}")
    return scoring.read_rankings(path)


def retrieve(
    questions: Sequence[Question],
    dense: Dense,
    bm25: BM25,
    table: QueryTable,
    p10a: Mapping[str, Sequence[str]],
    p10b: Mapping[str, Sequence[str]],
    say: Say = print,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, float], dict[str, int]]:
    """Both lists per question, retrieval seconds per retriever and the two equality counts."""
    depth = config.DEPTH
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in NAMES}
    seconds = dict.fromkeys(NAMES, 0.0)
    equal = {"dense_equals_p10-a": 0, "fusion_equals_p10-b": 0}
    for number, question in enumerate(questions, start=1):
        vector = table.vector(question.qid)
        started = time.perf_counter()
        first = dense.retrieve(vector, depth)
        seconds["dense"] += time.perf_counter() - started
        started = time.perf_counter()
        second = bm25.retrieve(question.question, depth)
        seconds["bm25"] += time.perf_counter() - started
        rankings["dense"][question.qid] = [unit_id for unit_id, _ in first]
        rankings["bm25"][question.qid] = [unit_id for unit_id, _ in second]
        fused = fuse_lists([first, second], config.WEIGHTS_P10B, top_k=depth)
        equal["dense_equals_p10-a"] += rankings["dense"][question.qid] == list(p10a[question.qid])
        equal["fusion_equals_p10-b"] += [u for u, _ in fused] == list(p10b[question.qid])
        if number % 500 == 0:
            say(f"{number}/{len(questions)} questions")
    return rankings, seconds, equal


def time_question_encoding(texts: Sequence[str], cached: np.ndarray) -> dict[str, Any]:
    """BGE-small at the pinned revision, CLS pooling and L2 norm, batch size 1, laptop CPU;
    the cosine with the cached vector shows the timed encoder is the one the lists used."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION
    )
    model = AutoModel.from_pretrained(config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION)
    model.eval()

    def encode(text: str) -> np.ndarray:
        batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            cls = model(**batch).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1)[0].numpy()

    started = time.perf_counter()
    encode("warm-up")
    warm_up = time.perf_counter() - started
    cosines: list[float] = []
    started = time.perf_counter()
    vectors = [encode(text) for text in texts]
    seconds = time.perf_counter() - started
    for vector, reference in zip(vectors, cached, strict=True):
        cosines.append(float(vector @ reference / np.linalg.norm(reference)))
    return {
        "model": config.EMBEDDING_MODEL,
        "revision": config.EMBEDDING_REVISION,
        "sample": "the first questions of the set in question-file order",
        "questions": len(texts),
        "batch_size": 1,
        "torch_threads": torch.get_num_threads(),
        "warm_up_seconds_not_counted": round(warm_up, 4),
        "seconds": round(seconds, 4),
        "seconds_per_question": round(seconds / len(texts), 6),
        "min_cosine_to_cached_vector": round(min(cosines), 6),
    }


def manifest(
    *,
    set_name: str,
    questions: int,
    checks: list[dict[str, str]],
    inputs_sha256: Mapping[str, str],
    stage_seconds: Mapping[str, float],
    retrieval_seconds: Mapping[str, float],
    encoding: Mapping[str, Any],
    equal: Mapping[str, int],
    outputs: Mapping[str, str],
    peak_rss: float,
) -> dict[str, Any]:
    return {
        "set": set_name,
        "questions": questions,
        "depth": config.DEPTH,
        "hardware": f"laptop CPU ({platform.machine()}, {platform.system()})",
        "git_commit": git_commit(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digests_checked": checks,
        "inputs_sha256": dict(inputs_sha256),
        "offline_seconds": {"bm25_build": stage_seconds["BM25 rebuild"]},
        "stage_seconds": dict(stage_seconds),
        "retrieval_seconds": {name: round(value, 3) for name, value in retrieval_seconds.items()},
        "retrieval_seconds_per_question": {
            name: round(value / questions, 6) for name, value in retrieval_seconds.items()
        },
        "question_encoding": dict(encoding),
        "equality": {name: {"equal": count, "of": questions} for name, count in equal.items()},
        "outputs_sha256": dict(outputs),
        "peak_rss_mb": peak_rss,
    }


def run(set_name: str, say: Say = print) -> dict[str, Any]:
    lap = Laps(set_name, say)
    phase02 = json.loads((config.RANKINGS_DIR / set_name / scoring.MANIFEST).read_text("utf-8"))
    p10b_sha = phase02["p10-b.jsonl.gz"]
    inputs_sha256 = {"p10-a.jsonl.gz": config.P10A_SHA256[set_name], "p10-b.jsonl.gz": p10b_sha}
    p10a = phase02_list(set_name, "p10-a", inputs_sha256["p10-a.jsonl.gz"])
    p10b = phase02_list(set_name, "p10-b", p10b_sha)
    inputs = load_inputs(set_name, lap)
    questions = inputs.questions
    dense = Dense(inputs.vectors, inputs.unit_ids)
    rankings, seconds, equal = retrieve(
        questions,
        dense,
        inputs.bm25,
        inputs.table,
        p10a,
        p10b,
        say=lambda line: say(f"[{set_name}] {line}"),
    )
    lap("retrieval")
    sample = questions[: config.ENCODING_SAMPLE]
    cached = np.stack([inputs.table.vector(q.qid) for q in sample])
    encoding = time_question_encoding([q.question for q in sample], cached)
    lap("question encoding sample")
    directory = config.PHASE03_RANKINGS_DIR / set_name
    outputs = {
        f"{name}.jsonl.gz": scoring.write_rankings(
            directory / f"{name}.jsonl.gz",
            ((q.qid, rankings[name][q.qid]) for q in questions),
        )
        for name in NAMES
    }
    lap("ranking files")
    body = manifest(
        set_name=set_name,
        questions=len(questions),
        checks=inputs.checks.records,
        inputs_sha256=inputs_sha256,
        stage_seconds=lap.seconds,
        retrieval_seconds=seconds,
        encoding=encoding,
        equal=equal,
        outputs=outputs,
        peak_rss=peak_rss_mb(),
    )
    path: Path = write_json(directory / MANIFEST, body)
    n = len(questions)
    for name, count in equal.items():
        say(f"[{set_name}] {name}: {count} of {n}")
    for name in NAMES:
        say(f"[{set_name}] {name}: {body['retrieval_seconds_per_question'][name]} s per question")
    say(
        f"[{set_name}] question encoding: {encoding['seconds_per_question']} s per question, "
        f"min cosine to cached {encoding['min_cosine_to_cached_vector']}"
    )
    say(f"[{set_name}] peak RSS {body['peak_rss_mb']} MB; manifest {path}")
    return body
