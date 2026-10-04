"""Phase 04 judge: J-strong over RRF3's and RRF4's top 100 (spec, frozen definition; plan D3).

A pair (question id, unit id) already scored in the old Phase 17 J-strong file takes that stored
score; every other pair is scored on the pod. Both judged systems read one score table per set,
so a pair both pools hold has one score. The judged order is score descending, ties by the
pool's own rank (`rerank.reorder`).
"""

import gzip
import hashlib
import json
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from edge_rag import config, fuse, pool, scoring
from edge_rag.artifacts import (
    ArtifactError,
    keep_manifest,
    sha256_file,
    write_bytes,
    write_manifest,
)
from edge_rag.pod.common import Unit, git_provenance, open_set, units_by_id
from edge_rag.pod.rerank import OLD17, reorder
from edge_rag.process import peak_rss_mb
from edge_rag.reproduce import Say

Pair = tuple[str, str]


def union(*pools: Sequence[str]) -> list[str]:
    """Every unit of the pools once, in the order first seen."""
    return list(dict.fromkeys(unit_id for ranked in pools for unit_id in ranked))


def uncached(qid: str, units: Sequence[str], stored: Mapping[Pair, float]) -> list[str]:
    """The units of one question whose pair has no stored score: the ones sent to the pod."""
    return [unit_id for unit_id in units if (qid, unit_id) not in stored]


def score_table(
    pairs: Iterable[Pair], stored: Mapping[Pair, float], fresh: Mapping[Pair, float]
) -> dict[Pair, float]:
    """One score per pair: the stored one when it exists, else the pod's. A pair scored by
    both, or by neither, refuses."""
    table: dict[Pair, float] = {}
    for pair in pairs:
        if pair in table:
            continue
        if pair in stored and pair in fresh:
            raise ArtifactError(f"{pair} has a stored score and a pod score")
        if pair in stored:
            table[pair] = float(stored[pair])
        elif pair in fresh:
            table[pair] = float(fresh[pair])
        else:
            raise ArtifactError(f"{pair} has no score")
    return table


def judged(qid: str, top: Sequence[str], table: Mapping[Pair, float]) -> list[str]:
    """A pool's top reordered by its scores, ties by pool rank."""
    return reorder(top, [table[(qid, unit_id)] for unit_id in top])


def pair_row(
    qid: str, question: str, unit_ids: Sequence[str], units: Mapping[str, Unit]
) -> dict[str, Any]:
    """One pairs-file line: the judge reads `(question, text)` with the harness text."""
    return {
        "qid": qid,
        "question": question,
        "unit_ids": list(unit_ids),
        "texts": [units[unit_id].text for unit_id in unit_ids],
    }


# --- `edge-rag judge-pairs` (spec C3, plan increment 3) ---------------------------------------
PAIRS_DIR = config.PHASE04_DIR / "pairs"
TIMING_QUESTIONS = 200
# The master plan's preregistered HotpotQA subsample (spec gate fallback, plan D8).
SUBSAMPLE_SET, SUBSAMPLE_SEED, SUBSAMPLE_SIZE = "hotpotqa-dev", 20261002, 1000
POD_SETUP_HOURS, POD_USD_PER_HOUR = 0.75, 0.74


def pinned_rankings(path: Path, sha256: str) -> dict[str, list[str]]:
    measured = sha256_file(path)
    if measured != sha256:
        raise ArtifactError(f"{path}: sha256 {measured}, expected {sha256}")
    return scoring.read_rankings(path)


def pools_of(set_name: str) -> tuple[dict[str, str], dict[str, dict[str, list[str]]]]:
    """`rrf3` by its Phase 03 manifest and `rrf4` by its Phase 04 manifest, top 100 each."""
    phase03 = config.PHASE03_RANKINGS_DIR / set_name
    phase04 = config.PHASE04_RANKINGS_DIR / set_name
    sha256 = {
        "rrf3.jsonl.gz": json.loads((phase03 / fuse.MANIFEST).read_text("utf-8"))["outputs_sha256"][
            "rrf3.jsonl.gz"
        ],
        "rrf4.jsonl.gz": json.loads((phase04 / pool.MANIFEST).read_text("utf-8"))["outputs_sha256"][
            "rrf4.jsonl.gz"
        ],
    }
    pools = {
        name: {
            qid: units[: config.DEPTH]
            for qid, units in pinned_rankings(
                directory / f"{name}.jsonl.gz", sha256[f"{name}.jsonl.gz"]
            ).items()
        }
        for name, directory in (("rrf3", phase03), ("rrf4", phase04))
    }
    return sha256, pools


def old17_files(set_name: str) -> dict[str, tuple[Path, str]]:
    old_name, pairs_sha, scores_sha = OLD17[set_name]
    directory = config.old_data_root() / config.OLD_REFERENCES_DIR
    return {
        "pairs": (directory / f"pairs-{old_name}.jsonl.gz", pairs_sha),
        "scores": (directory / f"scores-strong-{old_name}.jsonl.gz", scores_sha),
    }


def read_cache(
    set_name: str, wanted: Mapping[str, set[str]]
) -> tuple[dict[Pair, float], dict[str, str]]:
    """The stored J-strong scores of the wanted pairs and the old question text per qid, read
    in place after both old files match their pinned sha256, streamed line by line."""
    files = old17_files(set_name)
    for path, sha256 in files.values():
        measured = sha256_file(path)
        if measured != sha256:
            raise ArtifactError(f"{path}: sha256 {measured}, pinned {sha256}")
    stored: dict[Pair, float] = {}
    texts: dict[str, str] = {}
    with (
        gzip.open(files["pairs"][0], "rt", encoding="utf-8") as pairs,
        gzip.open(files["scores"][0], "rt", encoding="utf-8") as scores,
    ):
        for pair_line, score_line in zip(pairs, scores, strict=True):
            pair_row, score_row = json.loads(pair_line), json.loads(score_line)
            qid = str(pair_row["qid"])
            if qid != score_row["qid"] or pair_row["unit_ids"] != score_row["unit_ids"]:
                raise ArtifactError(f"old pairs and scores disagree at {qid}")
            texts[qid] = str(pair_row["question"])
            keep = wanted.get(qid, set())
            for unit_id, score in zip(score_row["unit_ids"], score_row["scores"], strict=True):
                if unit_id in keep:
                    stored[(qid, str(unit_id))] = float(score)
    return stored, texts


def projection(fresh_pairs: int, set_name: str) -> dict[str, Any]:
    """Scoring time and money of one set's fresh pairs at G-R's measured rate (spec gate)."""
    rate = config.GR_PAIRS_PER_SECOND[set_name]
    hours = fresh_pairs / rate / 3600
    return {
        "label": "projection",
        "fresh_pairs": fresh_pairs,
        "pairs_per_second": rate,
        "scoring_hours": round(hours, 4),
        "scoring_usd": round(hours * POD_USD_PER_HOUR, 4),
        "note": f"the phase adds {POD_SETUP_HOURS} h once, at {POD_USD_PER_HOUR} USD/h",
    }


def write_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    body = gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"), mtime=0)
    digest = hashlib.sha256(body).hexdigest()
    if path.exists() and sha256_file(path) != digest:
        raise ArtifactError(f"refusing to overwrite {path} with other content")
    write_bytes(path, body)
    return digest


def counted(pools: Mapping[str, Sequence[str]], stored: Mapping[Pair, float]) -> dict[str, Any]:
    total = sum(len(units) for units in pools.values())
    new = sum(len(uncached(qid, units, stored)) for qid, units in pools.items())
    return {
        "pairs": total,
        "cached": total - new,
        "uncached": new,
        "uncached_per_question": round(new / len(pools), 4),
    }


def run_pairs(set_name: str, say: Say = print) -> dict[str, Any]:
    provenance = git_provenance()
    manifest_path = PAIRS_DIR / f"{set_name}.manifest.json"
    keep_manifest(manifest_path)
    started = time.perf_counter()
    pool_sha256, pools = pools_of(set_name)
    spec, old, checks = open_set(set_name)
    questions = scoring.questions_of(old, checks, spec)
    qids = sorted(q.qid for q in questions)
    if qids != sorted(pools["rrf3"]) or qids != sorted(pools["rrf4"]):
        raise ArtifactError(f"{set_name}: the pools do not cover the set's questions")
    both = {q.qid: union(pools["rrf3"][q.qid], pools["rrf4"][q.qid]) for q in questions}
    stored, old_texts = read_cache(set_name, {qid: set(units) for qid, units in both.items()})
    say(f"[{set_name}] {len(stored)} stored scores kept from the old Phase 17 file")

    harness = {q.qid: q.question for q in questions}
    text_check = {"equal": 0, "of": 0, "old_qids_outside_set": 0}
    for qid, text in old_texts.items():
        if qid not in harness:
            text_check["old_qids_outside_set"] += 1
            continue
        text_check["of"] += 1
        text_check["equal"] += text == harness[qid]

    sent = {qid: uncached(qid, units, stored) for qid, units in both.items()}
    timing = questions[:TIMING_QUESTIONS]
    wanted = {u for units in sent.values() for u in units} | {
        u for q in timing for u in both[q.qid]
    }
    units = units_by_id(old, spec, wanted)
    pair_rows = [pair_row(q.qid, q.question, sent[q.qid], units) for q in questions if sent[q.qid]]
    timing_rows = [
        {
            **pair_row(q.qid, q.question, both[q.qid], units),
            "stored": [stored.get((q.qid, u)) for u in both[q.qid]],
        }
        for q in timing
    ]
    outputs = {
        f"{set_name}.jsonl.gz": write_once(PAIRS_DIR / f"{set_name}.jsonl.gz", pair_rows),
        f"{set_name}.timing.jsonl.gz": write_once(
            PAIRS_DIR / f"{set_name}.timing.jsonl.gz", timing_rows
        ),
    }
    seconds = time.perf_counter() - started

    counts = {name: counted(pools[name], stored) for name in ("rrf3", "rrf4")}
    counts["union"] = counted(both, stored)
    union_new = counts["union"]["uncached"]
    timing_pairs = sum(len(row["unit_ids"]) for row in timing_rows)
    projected = {"full": projection(union_new + timing_pairs, set_name)}
    if set_name == SUBSAMPLE_SET:
        chosen = set(random.Random(SUBSAMPLE_SEED).sample(qids, SUBSAMPLE_SIZE))  # noqa: S311
        sub_new = sum(len(units) for qid, units in sent.items() if qid in chosen)
        projected["subsample"] = {
            **projection(sub_new + timing_pairs, set_name),
            "questions": SUBSAMPLE_SIZE,
            "uncached_pairs": sub_new,
            "rule": f"random.Random({SUBSAMPLE_SEED}).sample over the sorted qids",
        }
    body: dict[str, Any] = {
        "set": set_name,
        "questions": len(questions),
        **provenance,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_sha256": {
            **pool_sha256,
            **{path.name: sha256 for path, sha256 in old17_files(set_name).values()},
        },
        "digests_checked": checks.records,
        "pair_counts": counts,
        "question_text_check": text_check,
        "timing_sample": {
            "questions": len(timing),
            "rule": "the first questions in question-file order, every unit of both pools",
            "pairs": timing_pairs,
            "with_stored_score": sum(s is not None for row in timing_rows for s in row["stored"]),
        },
        "projected_pod_cost": projected,
        "seconds": round(seconds, 3),
        "outputs_sha256": outputs,
        "peak_rss_mb": peak_rss_mb(),
    }
    write_manifest(manifest_path, body)
    for name, entry in counts.items():
        say(f"[{set_name}] {name}: {entry}")
    say(f"[{set_name}] question text check {text_check}")
    say(f"[{set_name}] timing sample {body['timing_sample']}")
    say(f"[{set_name}] projection {projected}")
    say(f"[{set_name}] outputs {outputs}; {seconds:.1f} s; peak RSS {body['peak_rss_mb']} MB")
    return body


# --- `edge-rag judge` (spec C6, plan increment 6) ---------------------------------------------
SCORES_DIR = config.PHASE04_DIR / "scores"
JUDGE_MANIFEST = "judge.manifest.json"
JUDGED = {"rrf3": "j-rrf3", "rrf4": "j-rrf4"}


def pod_scores(
    set_name: str, scores_dir: Path = SCORES_DIR, pairs_dir: Path = PAIRS_DIR
) -> tuple[dict[Pair, float], dict[str, str]]:
    """The pod's scores of one set, read only when its manifest says C4 passed, it scored the
    pairs file this laptop wrote, and the scores file matches the sha256 it recorded."""
    name = f"{set_name}.jsonl.gz"
    manifest = json.loads((scores_dir / f"{set_name}.manifest.json").read_text("utf-8"))
    pairs = json.loads((pairs_dir / f"{set_name}.manifest.json").read_text("utf-8"))
    if not manifest["check"]["pass"]:
        raise ArtifactError(f"{set_name}: the pod's C4 check did not pass")
    if manifest["inputs_sha256"] != pairs["outputs_sha256"]:
        raise ArtifactError(f"{set_name}: the pod scored other pairs files")
    sha256 = manifest["outputs"][name]
    measured = sha256_file(scores_dir / name)
    if measured != sha256:
        raise ArtifactError(f"{scores_dir / name}: sha256 {measured}, recorded {sha256}")
    fresh: dict[Pair, float] = {}
    with gzip.open(scores_dir / name, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for unit_id, score in zip(row["unit_ids"], row["scores"], strict=True):
                pair = (str(row["qid"]), str(unit_id))
                if pair in fresh:
                    raise ArtifactError(f"{pair} has two pod scores")
                fresh[pair] = float(score)
    return fresh, {
        name: sha256,
        f"{set_name}.manifest.json": sha256_file(scores_dir / f"{set_name}.manifest.json"),
    }


def assemble(
    pools: Mapping[str, Mapping[str, Sequence[str]]],
    stored: Mapping[Pair, float],
    fresh: Mapping[Pair, float],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Both judged systems from one score table over the union of the pools, with the
    permutation and one-score counts; a pod score no pool pair uses refuses."""
    qids = list(pools["rrf3"])
    pairs = [(qid, unit_id) for qid in qids for unit_id in union(*(p[qid] for p in pools.values()))]
    table = score_table(pairs, stored, fresh)
    unused = set(fresh) - set(table)
    if unused:
        raise ArtifactError(f"{len(unused)} pod scores belong to no pool pair")
    out = {name: {qid: judged(qid, p[qid], table) for qid in qids} for name, p in pools.items()}
    counts = {
        "questions": len(qids),
        "permutation_of_pool": {
            name: sum(sorted(out[name][qid]) == sorted(p[qid]) for qid in qids)
            for name, p in pools.items()
        },
        "pairs": len(pairs),
        "pairs_with_one_score": len(table),
        "from_cache": sum(pair in stored for pair in table),
        "from_pod": sum(pair in fresh for pair in table),
    }
    return out, counts


def run_judge(set_name: str, say: Say = print) -> dict[str, Any]:
    provenance = git_provenance()
    directory = config.PHASE04_RANKINGS_DIR / set_name
    keep_manifest(directory / JUDGE_MANIFEST)
    started = time.perf_counter()
    pool_sha256, pools = pools_of(set_name)
    spec, old, checks = open_set(set_name)
    order = [q.qid for q in scoring.questions_of(old, checks, spec)]
    if sorted(order) != sorted(pools["rrf3"]) or sorted(order) != sorted(pools["rrf4"]):
        raise ArtifactError(f"{set_name}: the pools do not cover the set's questions")
    pools = {name: {qid: p[qid] for qid in order} for name, p in pools.items()}
    wanted = {qid: set(pools["rrf3"][qid]) | set(pools["rrf4"][qid]) for qid in order}
    fresh, scores_sha256 = pod_scores(set_name)
    stored, _texts = read_cache(set_name, wanted)
    rankings, counts = assemble(pools, stored, fresh)
    outputs = {
        f"{JUDGED[name]}.jsonl.gz": scoring.write_rankings(
            directory / f"{JUDGED[name]}.jsonl.gz", ((qid, ranked[qid]) for qid in order)
        )
        for name, ranked in rankings.items()
    }
    body: dict[str, Any] = {
        "set": set_name,
        "questions": len(order),
        "depth": config.DEPTH,
        "order": "J-strong score descending, ties by the pool's rank",
        **provenance,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_sha256": {
            **pool_sha256,
            **{path.name: sha256 for path, sha256 in old17_files(set_name).values()},
            **scores_sha256,
        },
        "digests_checked": checks.records,
        "counts": counts,
        "seconds": round(time.perf_counter() - started, 3),
        "outputs_sha256": outputs,
        "peak_rss_mb": peak_rss_mb(),
    }
    write_manifest(directory / JUDGE_MANIFEST, body)
    say(f"[{set_name}] {counts}")
    say(f"[{set_name}] outputs {outputs}")
    return body
