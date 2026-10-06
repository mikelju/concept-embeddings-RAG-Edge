"""`edge-rag converge`: the multi-seed convergent hop and its fusions per set (Phase 05 spec C1).

Seeds, `mch`, `rrf-prf` and `rrf-1s` read the stored Dense (`p10-a`) and BM25 (Phase 03 `bm25`)
lists and the Phase 04 single-seed `hop` list, each only after its sha256 matches (plan D7).
Dense and BM25 are also recomputed per question, only for the online timings and the equality
counts, and the per-seed hop from Dense's first unit with Dense's top 10 excluded is checked
against the Phase 04 `hop` list, which `Hops.entity_hop` wrote.
"""

import hashlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edge_rag import config, fuse, scoring
from edge_rag.artifacts import ArtifactError, keep_manifest, write_manifest
from edge_rag.pod.common import git_provenance
from edge_rag.pool import hop_only
from edge_rag.process import peak_rss_mb
from edge_rag.reproduce import Laps, Say, load_inputs
from edge_rag.retrieval.dense_bm25 import Dense
from edge_rag.retrieval.hop import Hops, load_node_index
from edge_rag.retrieval.rrf import rrf

MANIFEST = "converge.manifest.json"
OUTPUTS = ("hop-ms", "dense-prf", "mch", "rrf-prf", "rrf-1s")
FUSION_INPUTS = {
    "mch": ("p10-a", "bm25", "dense-prf", "hop-ms"),
    "rrf-prf": ("p10-a", "bm25", "dense-prf"),
    "rrf-1s": ("p10-a", "bm25", "dense-prf", "hop"),
}


def seeds(
    dense: Sequence[str], bm25: Sequence[str], per_list: int = config.SEEDS_PER_LIST
) -> list[str]:
    """The top `per_list` units of Dense, then those of BM25's top `per_list` not already in."""
    chosen = list(dict.fromkeys(dense[:per_list]))
    chosen += [unit_id for unit_id in dict.fromkeys(bm25[:per_list]) if unit_id not in chosen]
    return chosen


def refined_query(question: np.ndarray, seed_vectors: np.ndarray) -> np.ndarray:
    """Rocchio with equal weights, `(q + c) / ||q + c||` with `c` the seeds' mean vector,
    computed in float64 and returned in the unit vectors' dtype (plan D8)."""
    centroid = np.asarray(seed_vectors, dtype=np.float64).mean(axis=0)
    total = np.asarray(question, dtype=np.float64) + centroid
    return (total / np.linalg.norm(total)).astype(seed_vectors.dtype)


def hop_ms(per_seed: Sequence[Sequence[str]]) -> list[str]:
    """RRF (k = 60) over the per-seed hop lists, top 100; an empty list adds nothing."""
    return rrf(per_seed, depth=config.DEPTH)


def fusion(name: str, lists: Mapping[str, Sequence[str]]) -> list[str]:
    """`mch`, `rrf-prf` or `rrf-1s`: RRF (k = 60) over its FUSION_INPUTS lists, top 100."""
    return rrf([lists[source] for source in FUSION_INPUTS[name]], depth=config.DEPTH)


def read_single_hop(set_name: str) -> tuple[str, dict[str, list[str]]]:
    """Phase 04's `hop` list, read only after it matches the sha256 its pool manifest records."""
    directory = config.PHASE04_RANKINGS_DIR / set_name
    manifest = json.loads((directory / "pool.manifest.json").read_text("utf-8"))
    sha = manifest["outputs_sha256"]["hop.jsonl.gz"]
    path = directory / "hop.jsonl.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ArtifactError(f"{path} does not match its pool manifest")
    return sha, scoring.read_rankings(path)


def run(set_name: str, say: Say = print) -> dict[str, Any]:
    provenance = git_provenance()
    directory = config.PHASE05_RANKINGS_DIR / set_name
    keep_manifest(directory / MANIFEST)
    lap = Laps(set_name, say)
    inputs_sha256, (p10a, bm25, _gl) = fuse.read_inputs(set_name)
    del inputs_sha256["g-l.jsonl.gz"], _gl
    inputs_sha256["hop.jsonl.gz"], single = read_single_hop(set_name)
    inputs = load_inputs(set_name, lap)
    spec, questions = inputs.spec, inputs.questions
    started = time.perf_counter()
    index = load_node_index(
        inputs.old, spec.nodes_dir, inputs.checks, recorded=spec.entity_index_digest
    )
    if list(index.unit_ids) != inputs.unit_ids:
        raise ArtifactError("the entity index rows are not the corpus units in order")
    hops = Hops(index, inputs.vectors)
    hop_offline = time.perf_counter() - started
    lap("entity index")
    dense = Dense(inputs.vectors, inputs.unit_ids)
    rows = {unit_id: row for row, unit_id in enumerate(inputs.unit_ids)}

    ranked: dict[str, dict[str, list[str]]] = {name: {} for name in OUTPUTS}
    online = dict.fromkeys(
        (
            "dense",
            "bm25",
            "seed_hops",
            "refine_and_dense_prf",
            "rrf_hop_ms",
            "rrf_mch",
            "rrf_rrf_prf",
            "rrf_rrf_1s",
        ),
        0.0,
    )
    equal = {"dense_equals_p10-a": 0, "bm25_equals_bm25": 0, "seed_hop_equals_phase04_hop": 0}
    seed_counts: list[int] = []
    empty_seed_lists = only = 0
    depth = config.DEPTH
    for number, question in enumerate(questions, start=1):
        qid = question.qid
        vector = inputs.table.vector(qid)
        tick = time.perf_counter()
        first = dense.retrieve(vector, depth)
        online["dense"] += time.perf_counter() - tick
        tick = time.perf_counter()
        second = inputs.bm25.retrieve(question.question, depth)
        online["bm25"] += time.perf_counter() - tick
        equal["dense_equals_p10-a"] += [u for u, _ in first] == p10a[qid]
        equal["bm25_equals_bm25"] += [u for u, _ in second] == bm25[qid]
        read = [u for u, _ in first[: config.READ_DEPTH]]
        check = hops.seed_hop(read[0], read, depth)
        equal["seed_hop_equals_phase04_hop"] += [u for u, _ in check] == single[qid]

        chosen = seeds(p10a[qid], bm25[qid])
        seed_counts.append(len(chosen))
        tick = time.perf_counter()
        per_seed = [[u for u, _ in hops.seed_hop(seed, chosen, depth)] for seed in chosen]
        online["seed_hops"] += time.perf_counter() - tick
        empty_seed_lists += sum(not hits for hits in per_seed)
        tick = time.perf_counter()
        refined = refined_query(vector, inputs.vectors[[rows[seed] for seed in chosen]])
        prf = [u for u, _ in dense.retrieve(refined, depth)]
        online["refine_and_dense_prf"] += time.perf_counter() - tick
        tick = time.perf_counter()
        convergent = hop_ms(per_seed)
        online["rrf_hop_ms"] += time.perf_counter() - tick
        lists = {
            "p10-a": p10a[qid],
            "bm25": bm25[qid],
            "dense-prf": prf,
            "hop-ms": convergent,
            "hop": single[qid],
        }
        ranked["hop-ms"][qid], ranked["dense-prf"][qid] = convergent, prf
        for name in FUSION_INPUTS:
            tick = time.perf_counter()
            ranked[name][qid] = fusion(name, lists)
            online[f"rrf_{name.replace('-', '_')}"] += time.perf_counter() - tick
        others = [p10a[qid], bm25[qid], prf]
        only += hop_only(ranked["mch"][qid], others, convergent)
        if number % 500 == 0:
            say(f"[{set_name}] {number}/{len(questions)} questions")
    lap("seeds, hops and fusions")

    outputs = {
        f"{name}.jsonl.gz": scoring.write_rankings(
            directory / f"{name}.jsonl.gz", ((q.qid, ranked[name][q.qid]) for q in questions)
        )
        for name in OUTPUTS
    }
    lap("ranking files")
    n = len(questions)
    hop_ms_lists = ranked["hop-ms"].values()
    body: dict[str, Any] = {
        "set": set_name,
        "questions": n,
        "depth": depth,
        "fusion_input_order": {name: list(order) for name, order in FUSION_INPUTS.items()},
        "inputs_sha256": inputs_sha256,
        "entity_index_digest": index.digest,
        "gliner_configuration_digest": config.GLINER_CONFIGURATION_DIGEST,
        "constants": {
            "seeds_per_list": config.SEEDS_PER_LIST,
            "rrf_k": config.RRF_K,
            "depth": depth,
            "read_depth_of_equality_hop": config.READ_DEPTH,
        },
        "refined_query_dtype": str(inputs.vectors.dtype),
        "hardware": f"laptop CPU ({platform.machine()}, {platform.system()})",
        **provenance,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digests_checked": inputs.checks.records,
        "offline_seconds": {"hop_entity_index_and_weights": round(hop_offline, 3)},
        "stage_seconds": dict(lap.seconds),
        "online_seconds": {name: round(value, 3) for name, value in online.items()},
        "online_seconds_per_question": {
            name: round(value / n, 6) for name, value in online.items()
        },
        "equality": {name: {"equal": value, "of": n} for name, value in equal.items()},
        "seed_count": {
            "mean": round(sum(seed_counts) / n, 2),
            "min": min(seed_counts),
            "max": max(seed_counts),
        },
        "per_seed_lists_empty": empty_seed_lists,
        "hop_ms_list_lengths": {
            "short": sum(len(v) < depth for v in hop_ms_lists),
            "empty": sum(not v for v in hop_ms_lists),
        },
        "mch_units_only_in_hop_ms": {
            "label": "measured",
            "total": only,
            "per_question": round(only / n, 4),
        },
        "outputs_sha256": outputs,
        "peak_rss_mb": peak_rss_mb(),
    }
    path: Path = write_manifest(directory / MANIFEST, body)
    for name, entry in body["equality"].items():
        say(f"[{set_name}] {name}: {entry['equal']} of {entry['of']}")
    say(f"[{set_name}] seed count {body['seed_count']}")
    say(f"[{set_name}] hop-ms lists {body['hop_ms_list_lengths']}")
    say(f"[{set_name}] per-seed lists empty: {empty_seed_lists}")
    say(f"[{set_name}] mch units only in hop-ms: {body['mch_units_only_in_hop_ms']}")
    say(f"[{set_name}] online s per question {body['online_seconds_per_question']}")
    say(f"[{set_name}] outputs {outputs}")
    say(f"[{set_name}] peak RSS {body['peak_rss_mb']} MB; manifest {path}")
    return body
