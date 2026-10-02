"""The reproduction gate: the four inherited systems run live over one set's cached inputs."""

import gc
import gzip
import io
import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from edge_rag import config, metrics
from edge_rag.artifacts import ArtifactError, Checks, OldData, digest_of, sha256_file, write_json
from edge_rag.corpus import load_corpus, load_questions
from edge_rag.embeddings import (
    QueryTable,
    legacy_corpus_cache_key,
    legacy_question_cache_key,
    load_vectors,
    question_set_hash,
    vectors_digest,
)
from edge_rag.retrieval.dense_bm25 import BM25, Dense, Hit
from edge_rag.retrieval.fusion import fuse_lists
from edge_rag.retrieval.hop import Hops, load_node_index

Say = Callable[[str], None]


def _key_of(relative: str) -> str:
    return relative.rsplit("embeddings-", 1)[1].removesuffix(".npz")


def _token_counts(old: OldData, checks: Checks, spec: config.CorpusSet, unit_ids: list[str]):
    manifest = old.json("token-counts.json")
    if manifest["unit_set_hash"] != spec.unit_set_hash:
        raise ArtifactError("the token counts belong to another corpus")
    with old.path("token-counts.npz").open("rb") as handle, np.load(handle) as z:
        ids = [str(unit_id) for unit_id in z["unit_ids"]]
        counts = z["counts"].astype(np.int32)
    if ids != unit_ids:
        raise ArtifactError("the token counts are not keyed by the corpus units in order")
    checks.expect(
        "token-counts.json counts_digest", manifest["counts_digest"], spec.token_counts_digest
    )
    checks.expect("token counts digest", digest_of(counts), spec.token_counts_digest)
    return dict(zip(ids, (int(c) for c in counts), strict=True))


def _fits(checks: Checks) -> None:
    """The frozen weights are pinned in config; the fits they came from must be unchanged."""
    expected = {
        "phase10/fit.json": ("entity-hop", None),
        "phase14/fit.json": ("relevance-hop", 0.75),
    }
    for relative, digest in config.FIT_DIGESTS.items():
        directory, name = relative.split("/")
        fit = OldData(directory).json(name)
        checks.expect(relative, digest_of(json.dumps(dict(fit), sort_keys=True)), digest)
        hop, alpha = expected[relative]
        weights = (fit["weights"]["dense"], fit["weights"]["bm25"], fit["weights"][hop])
        if weights != config.WEIGHTS_TRIPLE or fit.get("alpha") != alpha:
            raise ArtifactError(f"{relative} does not hold the pinned weights")


def _stored(old: OldData, spec: config.CorpusSet) -> dict[str, dict[str, list[str]]]:
    stored: dict[str, dict[str, list[str]]] = {}
    for system, relative in spec.stored_rankings.items():
        lists: dict[str, list[str]] = {}
        with old.path(relative).open("rb") as raw, gzip.GzipFile(fileobj=raw) as gz:
            for line in io.TextIOWrapper(gz, encoding="utf-8"):
                record = json.loads(line)
                lists[str(record["qid"])] = [str(hit[0]) for hit in record["fused"]]
        stored[system] = lists
    return stored


def run(set_name: str, say: Say = print) -> dict[str, Any]:
    spec = config.SETS[set_name]
    old = OldData(spec.directory)
    checks = Checks()
    seconds: dict[str, float] = {}
    started = time.perf_counter()

    def lap(stage: str) -> None:
        seconds[stage] = round(time.perf_counter() - started - sum(seconds.values()), 3)
        say(f"[{set_name}] {stage} done in {seconds[stage]:.1f} s")

    say(f"[{set_name}] reading {old.base}")
    for relative, digest in spec.file_sha256.items():
        checks.expect(f"sha256 {relative}", sha256_file(old.path(relative)), digest)
    lap("sha256")
    corpus = load_corpus(
        old, checks, ordered_digest=spec.ordered_unit_digest, set_hash=spec.unit_set_hash
    )
    unit_ids = corpus.unit_ids
    questions, _body = load_questions(
        old,
        checks,
        corpus_set_hash=spec.unit_set_hash,
        question_digest_recorded=spec.question_digest,
        mapping_digest_recorded=spec.mapping_digest,
    )
    if len(questions) != spec.questions:
        raise ArtifactError(f"{len(questions)} questions, expected {spec.questions}")
    token_counts = _token_counts(old, checks, spec, unit_ids)
    _fits(checks)
    lap("corpus, questions, token counts")

    bm25 = BM25(corpus.texts, unit_ids, stopwords=config.BM25_STOPWORDS)
    del corpus
    gc.collect()
    bm25_manifest = old.json(spec.bm25_manifest)
    if bm25_manifest["library_version"] != config.BM25_VERSION:
        raise ArtifactError("the BM25 manifest names another bm25s version")
    checks.expect("bm25.json index_digest", bm25_manifest["index_digest"], spec.bm25_index_digest)
    checks.expect("rebuilt BM25 index digest", bm25.digest(), spec.bm25_index_digest)
    lap("BM25 rebuild")

    embedding = old.json("cache/embedding.json" if spec.directory != "phase9" else "embedding.json")
    corpus_key = legacy_corpus_cache_key(
        config.EMBEDDING_MODEL, config.EMBEDDING_REVISION, spec.unit_set_hash
    )
    checks.expect("corpus vector cache key", corpus_key, _key_of(spec.vectors_file))
    checks.expect("embedding.json vectors_digest", embedding["vectors_digest"], spec.vectors_digest)
    checks.expect(
        "embedding.json weights_sha256",
        embedding["weights_sha256"],
        config.EMBEDDING_WEIGHTS_SHA256,
    )
    vectors, vector_ids = load_vectors(old.path(spec.vectors_file))
    if vector_ids != unit_ids:
        raise ArtifactError("the passage vectors are not the corpus units in order")
    del vector_ids
    checks.expect("passage vectors digest", vectors_digest(vectors), spec.vectors_digest)
    question_vectors, qids = load_vectors(old.path(spec.question_vectors_file))
    question_key = legacy_question_cache_key(
        config.EMBEDDING_MODEL,
        config.EMBEDDING_REVISION,
        question_set_hash(qids),
        config.LEGACY_QUESTION_SPLIT,
    )
    checks.expect("question vector cache key", question_key, _key_of(spec.question_vectors_file))
    if qids[: len(questions)] != [q.qid for q in questions]:
        raise ArtifactError("the question vectors do not start with the questions in order")
    table = QueryTable(question_vectors, qids)
    lap("vectors")

    index = load_node_index(old, spec.nodes_dir, checks, recorded=spec.entity_index_digest)
    if list(index.unit_ids) != unit_ids:
        raise ArtifactError("the entity index rows are not the corpus units in order")
    dense = Dense(vectors, unit_ids)
    hops = Hops(index, vectors)
    lap("entity index")

    stored = _stored(old, spec)
    names = config.SYSTEMS
    outcomes: dict[str, list[int]] = {name: [] for name in names}
    recall = {name: {k: 0.0 for k in config.RECALL_KS} for name in names}
    ndcg = dict.fromkeys(names, 0.0)
    agree = dict.fromkeys(stored, 0)
    depth = config.DEPTH
    for number, question in enumerate(questions, start=1):
        qv = table.vector(question.qid)
        first = dense.retrieve(qv, depth)
        second = bm25.retrieve(question.question, depth)
        lists: dict[str, list[Hit]] = {
            config.P10A: first,
            config.P10B: fuse_lists([first, second], config.WEIGHTS_P10B, top_k=depth),
            config.P10C: fuse_lists(
                [first, second, hops.entity_hop(first, depth)], config.WEIGHTS_TRIPLE, top_k=depth
            ),
            config.P14: fuse_lists(
                [first, second, hops.relevance_hop(first, qv, depth, config.P14_ALPHA)],
                config.WEIGHTS_TRIPLE,
                top_k=depth,
            ),
        }
        gold = question.gold_unit_ids
        for name, hits in lists.items():
            ranked = [unit_id for unit_id, _ in hits]
            context = metrics.fill_context(ranked, token_counts, config.BUDGET)
            outcomes[name].append(metrics.full_support(context, gold))
            for k in config.RECALL_KS:
                recall[name][k] += metrics.recall_at_k(ranked, gold, k)
            ndcg[name] += metrics.ndcg_at_k(ranked, gold, config.NDCG_K)
            if name in stored and stored[name].get(question.qid) == ranked:
                agree[name] += 1
        if number % 500 == 0:
            progress = " ".join(f"{n}={sum(outcomes[n])}" for n in names)
            say(f"[{set_name}] {number}/{len(questions)} questions: {progress}")
    lap("retrieval")

    n = len(questions)
    counts = {name: sum(outcomes[name]) for name in names}
    gate = {
        name: {
            "recorded": spec.gate[name],
            "measured": counts[name],
            "match": counts[name] == spec.gate[name],
        }
        for name in names
    }
    pairs = [(config.P10A, config.P10B), (config.P10B, config.P10C), (config.P10C, config.P14)]
    result: dict[str, Any] = {
        "set": set_name,
        "questions": n,
        "budget": config.BUDGET,
        "depth": depth,
        "full_support": counts,
        "gate": gate,
        "gate_pass": all(entry["match"] for entry in gate.values()),
        "gliner_configuration_digest": config.GLINER_CONFIGURATION_DIGEST,
        "recall_at_k": {name: {str(k): v / n for k, v in recall[name].items()} for name in names},
        "ndcg_at_10": {name: ndcg[name] / n for name in names},
        "paired_full_support": {
            f"{b} vs {a}": metrics.paired(outcomes[a], outcomes[b]) for a, b in pairs
        },
        "stored_ranking_cross_check": {
            name: {"identical_lists": agree[name], "of": n} for name in stored
        },
        "outcomes_digest": {
            name: digest_of(np.array(outcomes[name], dtype=np.int8)) for name in names
        },
        "digests_checked": checks.records,
        "old_data_root": str(old.root),
        "seconds": seconds,
    }
    path = write_json(config.DATA_DIR / "results" / f"reproduce-{set_name}.json", result)
    for name in names:
        entry = gate[name]
        verdict = "match" if entry["match"] else "MISS"
        say(
            f"[{set_name}] {name}: Full Support {entry['measured']} of {n} "
            f"(recorded {entry['recorded']}) {verdict}"
        )
    for name, entry in result["stored_ranking_cross_check"].items():
        say(f"[{set_name}] {name}: {entry['identical_lists']} of {n} lists equal the stored ones")
    say(f"[{set_name}] {len(checks.records)} digests checked; result written to {path}")
    say(f"[{set_name}] GATE {'PASS' if result['gate_pass'] else 'FAIL'}")
    return result
