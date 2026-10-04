"""`edge-rag pool`: the entity hop at depth 100 and RRF4 per set (Phase 04 spec C1, plan D2).

RRF4 fuses, in this order, the stored Dense (`p10-a`), BM25 (Phase 03 `bm25`) and G-L (`g-l`)
lists, each read only after its sha256 matches, with the hop computed here. The recomputed
Dense and BM25 hits serve only the hop and the equality count against `p10-c`, which ties the
hop to the gated Phase 01 pipeline.
"""

import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from edge_rag import components, config, fuse, scoring
from edge_rag.artifacts import ArtifactError, keep_manifest, write_manifest
from edge_rag.pod.common import git_provenance
from edge_rag.process import peak_rss_mb
from edge_rag.reproduce import Laps, Say, load_inputs
from edge_rag.retrieval.dense_bm25 import Dense
from edge_rag.retrieval.fusion import fuse_lists
from edge_rag.retrieval.hop import Hops, load_node_index
from edge_rag.retrieval.rrf import rrf

MANIFEST = "pool.manifest.json"
INPUTS = ("p10-a", "bm25", "g-l", "hop")


def rrf4(lists: Sequence[Sequence[str]], hop: Sequence[str]) -> list[str]:
    """RRF (k = 60) of Dense, BM25 and G-L, then the hop; a short hop list adds what it holds."""
    return rrf([*lists, hop], depth=config.DEPTH)


def hop_only(pool: Sequence[str], lists: Sequence[Sequence[str]], hop: Sequence[str]) -> int:
    """Units of the pool that only the hop list holds."""
    others = {unit_id for ranked in lists for unit_id in ranked}
    held = set(hop)
    return sum(unit_id in held and unit_id not in others for unit_id in pool)


def run(set_name: str, say: Say = print) -> dict[str, Any]:
    provenance = git_provenance()
    directory = config.PHASE04_RANKINGS_DIR / set_name
    keep_manifest(directory / MANIFEST)
    lap = Laps(set_name, say)
    inputs_sha256, stored = fuse.read_inputs(set_name)
    phase02 = json.loads((config.RANKINGS_DIR / set_name / scoring.MANIFEST).read_text("utf-8"))
    inputs_sha256["p10-c.jsonl.gz"] = phase02["p10-c.jsonl.gz"]
    p10c = components.phase02_list(set_name, "p10-c", inputs_sha256["p10-c.jsonl.gz"])
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

    hop_lists: dict[str, list[str]] = {}
    pools: dict[str, list[str]] = {}
    online = {"hop": 0.0, "rrf4": 0.0}
    equal = only = 0
    depth = config.DEPTH
    for number, question in enumerate(questions, start=1):
        qid = question.qid
        first = dense.retrieve(inputs.table.vector(qid), depth)
        second = inputs.bm25.retrieve(question.question, depth)
        tick = time.perf_counter()
        hits = hops.entity_hop(first, depth)
        online["hop"] += time.perf_counter() - tick
        fused = fuse_lists([first, second, hits], config.WEIGHTS_TRIPLE, top_k=depth)
        equal += [unit_id for unit_id, _ in fused] == p10c[qid]
        hop_lists[qid] = [unit_id for unit_id, _ in hits]
        lists = [ranked[qid] for ranked in stored]
        tick = time.perf_counter()
        pools[qid] = rrf4(lists, hop_lists[qid])
        online["rrf4"] += time.perf_counter() - tick
        only += hop_only(pools[qid], lists, hop_lists[qid])
        if number % 500 == 0:
            say(f"[{set_name}] {number}/{len(questions)} questions")
    lap("hop and RRF4")

    outputs = {
        f"{name}.jsonl.gz": scoring.write_rankings(
            directory / f"{name}.jsonl.gz", ((q.qid, ranked[q.qid]) for q in questions)
        )
        for name, ranked in (("hop", hop_lists), ("rrf4", pools))
    }
    lap("ranking files")
    n = len(questions)
    body: dict[str, Any] = {
        "set": set_name,
        "questions": n,
        "depth": depth,
        "fusion_input_order": list(INPUTS),
        "inputs_sha256": inputs_sha256,
        "entity_index_digest": index.digest,
        "constants": {"rrf_k": config.RRF_K, "p10c_weights": list(config.WEIGHTS_TRIPLE)},
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
        "equality": {"fusion_equals_p10-c": {"equal": equal, "of": n}},
        "hop_list_lengths": {
            "short": sum(len(v) < depth for v in hop_lists.values()),
            "empty": sum(not v for v in hop_lists.values()),
        },
        "rrf4_units_only_in_hop": {
            "label": "measured",
            "total": only,
            "per_question": round(only / n, 4),
        },
        "outputs_sha256": outputs,
        "peak_rss_mb": peak_rss_mb(),
    }
    path: Path = write_manifest(directory / MANIFEST, body)
    say(f"[{set_name}] fusion_equals_p10-c: {equal} of {n}")
    say(f"[{set_name}] RRF4 units only in hop: {only} ({body['rrf4_units_only_in_hop']})")
    say(f"[{set_name}] online s per question {body['online_seconds_per_question']}")
    say(f"[{set_name}] hop {outputs['hop.jsonl.gz']}; rrf4 {outputs['rrf4.jsonl.gz']}")
    say(f"[{set_name}] peak RSS {body['peak_rss_mb']} MB; manifest {path}")
    return body
