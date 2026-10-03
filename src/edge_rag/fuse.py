"""`edge-rag fuse`: RRF3 and F3 depth-100 lists per set, with cost and diagnostics (spec C3, C6).

Inputs, each read only after its sha256 matches: Dense (`p10-a`) and G-L (`g-l`) from Phase 02,
by the sha256 the spec froze, and BM25 from Phase 03, by its `components` manifest. They are
fused in that order. The gold-informed counts are exploratory diagnostics and decide nothing.
"""

import hashlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from edge_rag import components, config, metrics, scoring
from edge_rag.artifacts import ArtifactError, Checks, OldData, keep_manifest, write_manifest
from edge_rag.corpus import Question, load_corpus
from edge_rag.pod.common import git_provenance
from edge_rag.process import peak_rss_mb
from edge_rag.reproduce import Say
from edge_rag.retrieval.rrf import RULES, boilerplate_flags, diversify, rrf

MANIFEST = "fuse.manifest.json"
INPUTS = ("p10-a", "bm25", "g-l")


def read_inputs(set_name: str) -> tuple[dict[str, str], list[dict[str, list[str]]]]:
    directory = config.PHASE03_RANKINGS_DIR / set_name
    bm25_sha = json.loads((directory / components.MANIFEST).read_text("utf-8"))["outputs_sha256"][
        "bm25.jsonl.gz"
    ]
    path = directory / "bm25.jsonl.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != bm25_sha:
        raise ArtifactError(f"{path} does not match its components manifest")
    sha256 = {
        "p10-a.jsonl.gz": config.P10A_SHA256[set_name],
        "bm25.jsonl.gz": bm25_sha,
        "g-l.jsonl.gz": config.GL_SHA256[set_name],
    }
    lists = [
        components.phase02_list(set_name, "p10-a", sha256["p10-a.jsonl.gz"]),
        scoring.read_rankings(path),
        components.phase02_list(set_name, "g-l", sha256["g-l.jsonl.gz"]),
    ]
    return sha256, lists


def gold_demotions(
    questions: Sequence[Question],
    rrf3: Mapping[str, Sequence[str]],
    f3: Mapping[str, Sequence[str]],
    demoted: Mapping[str, Mapping[str, str]],
    counts: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Per rule, the questions where a gold unit that rule demoted was in RRF3's reading context
    at the budget but not in F3's, and the same out of the top 100."""
    found = {rule: {"in_context_at_budget": 0, "in_top_100": 0} for rule in RULES}
    for question in questions:
        qid = question.qid
        before = set(metrics.fill_context(rrf3[qid], counts, config.BUDGET))
        after = set(metrics.fill_context(f3[qid], counts, config.BUDGET))
        top_before, top_after = set(rrf3[qid]), set(f3[qid])
        for rule in RULES:
            gold = [g for g in question.gold_unit_ids if demoted[qid].get(g) == rule]
            if any(g in before and g not in after for g in gold):
                found[rule]["in_context_at_budget"] += 1
            if any(g in top_before and g not in top_after for g in gold):
                found[rule]["in_top_100"] += 1
    return found


def demotions_in(
    ranked: Mapping[str, list[str]], demoted: Mapping[str, Mapping[str, str]]
) -> dict[str, int]:
    """Demotions per rule counted only over the units each question's list holds."""
    return {
        rule: sum([demoted[qid].get(u) for u in units].count(rule) for qid, units in ranked.items())
        for rule in RULES
    }


def run(set_name: str, say: Say = print) -> dict[str, Any]:
    spec = config.SETS[set_name]
    provenance = git_provenance()
    keep_manifest(config.PHASE03_RANKINGS_DIR / set_name / MANIFEST)
    started = time.perf_counter()
    inputs_sha256, lists = read_inputs(set_name)
    checks = Checks()
    old = OldData(spec.directory, spec.file_sha256, checks)
    corpus = load_corpus(
        old, checks, ordered_digest=spec.ordered_unit_digest, set_hash=spec.unit_set_hash
    )
    questions = scoring.questions_of(old, checks, spec)
    counts = scoring.token_counts(old, checks, spec)
    loading = time.perf_counter() - started
    say(f"[{set_name}] inputs and corpus read in {loading:.1f} s")

    started = time.perf_counter()
    flags = boilerplate_flags(map(corpus.body, range(len(corpus.unit_ids))), corpus.titles)
    flag_seconds = time.perf_counter() - started
    row_of = {unit_id: row for row, unit_id in enumerate(corpus.unit_ids)}
    say(f"[{set_name}] {int(flags.sum())} boilerplate units, flagged in {flag_seconds:.1f} s")

    rrf3: dict[str, list[str]] = {}
    f3: dict[str, list[str]] = {}
    demoted: dict[str, dict[str, str]] = {}
    online = {"rrf": 0.0, "diversity": 0.0}
    for number, question in enumerate(questions, start=1):
        qid = question.qid
        tick = time.perf_counter()
        order = rrf([ranked[qid] for ranked in lists])
        online["rrf"] += time.perf_counter() - tick
        rrf3[qid] = order[: config.DEPTH]
        tick = time.perf_counter()
        rows = {unit_id: row_of[unit_id] for unit_id in order}
        f3[qid], demoted[qid] = diversify(
            order,
            {unit_id: corpus.titles[row] for unit_id, row in rows.items()},
            {unit_id: corpus.body(row) for unit_id, row in rows.items()},
            {unit_id: bool(flags[row]) for unit_id, row in rows.items()},
        )
        online["diversity"] += time.perf_counter() - tick
        if number % 1000 == 0:
            say(f"[{set_name}] {number}/{len(questions)} questions")

    directory = config.PHASE03_RANKINGS_DIR / set_name
    outputs = {
        f"{name}.jsonl.gz": scoring.write_rankings(
            directory / f"{name}.jsonl.gz", ((q.qid, ranked[q.qid]) for q in questions)
        )
        for name, ranked in (("rrf3", rrf3), ("f3", f3))
    }
    gold_units = {g for q in questions for g in q.gold_unit_ids}
    n = len(questions)
    body: dict[str, Any] = {
        "set": set_name,
        "questions": n,
        "depth": config.DEPTH,
        "fusion_input_order": list(INPUTS),
        "inputs_sha256": inputs_sha256,
        "constants": {
            "rrf_k": config.RRF_K,
            "source_cap": config.SOURCE_CAP,
            "shingle_size": config.SHINGLE_SIZE,
            "near_duplicate_jaccard": config.NEAR_DUPLICATE_JACCARD,
        },
        "hardware": f"laptop CPU ({platform.machine()}, {platform.system()})",
        **provenance,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "digests_checked": checks.records,
        "corpus_units": len(corpus.unit_ids),
        "boilerplate_units": int(flags.sum()),
        "offline_seconds": {"boilerplate_flags": round(flag_seconds, 3)},
        "loading_seconds": round(loading, 3),
        "online_seconds": {name: round(value, 3) for name, value in online.items()},
        "online_seconds_per_question": {
            name: round(value / n, 6) for name, value in online.items()
        },
        "demotions_per_rule": {
            rule: sum(list(d.values()).count(rule) for d in demoted.values()) for rule in RULES
        },
        "demotions_per_rule_scope": "RRF3's whole union, up to 300 units per question",
        "demotions_per_rule_in_rrf3_top_100": demotions_in(rrf3, demoted),
        "exploratory_gold_diagnostics": {
            "label": "exploratory",
            "questions_with_a_gold_unit_demoted": gold_demotions(
                questions, rrf3, f3, demoted, counts
            ),
            "gold_units_flagged_boilerplate": sum(
                bool(flags[row_of[g]]) for g in sorted(gold_units)
            ),
            "gold_units": len(gold_units),
        },
        "outputs_sha256": outputs,
        "peak_rss_mb": peak_rss_mb(),
    }
    path: Path = write_manifest(directory / MANIFEST, body)
    say(f"[{set_name}] rrf3 sha256 {outputs['rrf3.jsonl.gz']}")
    say(f"[{set_name}] f3 sha256 {outputs['f3.jsonl.gz']}")
    say(f"[{set_name}] demotions per rule {body['demotions_per_rule']}")
    say(f"[{set_name}] online s per question {body['online_seconds_per_question']}")
    say(f"[{set_name}] peak RSS {body['peak_rss_mb']} MB; manifest {path}")
    return body
