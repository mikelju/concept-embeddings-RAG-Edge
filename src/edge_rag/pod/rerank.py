"""G-R: bge-reranker-v2-m3 over G-L's top 100, with the old J-strong settings (spec scope 4).

Settings copied from the old project (`evaluation/judge.py`, `config.PHASE_17_JUDGES["strong"]`):
`sentence_transformers.CrossEncoder` at the pinned revision, raw logit as the score
(`activation_fn=Identity`), float32 weights from `model.safetensors` only, matmul precision
`highest`, no `trust_remote_code`, the model's own maximum length (8,192), `predict` batch 32,
pairs `(question, f"{title}. {' '.join(sentences)}")`. The order is score descending, ties by
G-L's rank (old `phase17.reorder`).

`--check-old` is criterion C3: the first 100 pairs in file order of each old Phase 17 pairs
file, rescored and compared with the stored J-strong scores.
`--qwen` is G-R2 (optional): Qwen3-Reranker-0.6B with its model card's yes/no-logit prompt.
`--pairs <set>` is Phase 04's pod mode (spec C4, C5; plan D5): it reads the uploaded pairs and
timing files of `edge-rag judge-pairs` and never opens a corpus.

    uv run --group pod python -m edge_rag.pod.rerank --set musique
    uv run --group pod python -m edge_rag.pod.rerank --pairs musique
"""

import argparse
import gzip
import json
import time
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from edge_rag.artifacts import ArtifactError, sha256_file, write_bytes, write_json
from edge_rag.pod import common

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
WEIGHTS_FILE = "model.safetensors"
WEIGHTS_SHA256 = "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
SNAPSHOT_PATTERNS = ("*.json", WEIGHTS_FILE, "vocab.txt")
MAX_LENGTH = 8192
DTYPE = "float32"
BATCH_SIZE = 32
MATMUL_PRECISION = "highest"

QWEN_MODEL = "Qwen/Qwen3-Reranker-0.6B"
# HF Hub `sha`, read 2026-10-02 (api/models/Qwen/Qwen3-Reranker-0.6B).
QWEN_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
QWEN_MAX_LENGTH = 8192
QWEN_BATCH_SIZE = 16
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

DEPTH = 100
SHARD_QUESTIONS = 250  # old PHASE_17_SHARD_QUESTIONS: questions per `predict` call
CHECK_PAIRS = 100
CHECK_TOLERANCE = 1e-3
# Old Phase 17 file name of each set, and the sha256 of its pairs and J-strong score files
# (the scores' digests are the ones recorded in old data/phase17/scoring-strong.json).
OLD17 = {
    "hotpotqa-dev": (
        "hotpotqa",
        "a82185054cadb4ace32d757a7e55a9cf398a029242a115c8e4d206f54368dff3",
        "62e892bfd9b3444bd26ea4b1cd534aaba4c85ae5a8514cb284a4bcd6c4da5f7b",
    ),
    "musique": (
        "musique",
        "fef878aeba159afd159305804675a144d111459b22c8b186a523d0a4e79863a1",
        "a9e744f973a29445ca7a10e495d4e60851cf82cffece7cd6cca5e79badaef0f9",
    ),
    "multihop-rag": (
        "multihop-rag",
        "939132e93e0da3a604caf9569bdd94d1e8e7584f32c9d6f7335dae62baf3db71",
        "d5a7af5b2922543f98cc96aaaaa77e052922c83df6c23bbe634c2d32c3873cb0",
    ),
}


def load_strong() -> Any:
    """The J-strong CrossEncoder, refused unless the served snapshot is the pinned one."""
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder

    snapshot = Path(
        snapshot_download(MODEL, revision=REVISION, allow_patterns=list(SNAPSHOT_PATTERNS))
    )
    if snapshot.name != REVISION:
        raise ArtifactError(f"{MODEL}: the Hub served revision {snapshot.name}")
    digest = sha256_file(snapshot / WEIGHTS_FILE)
    if digest != WEIGHTS_SHA256:
        raise ArtifactError(f"{MODEL}: {WEIGHTS_FILE} has sha256 {digest}")
    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    model = CrossEncoder(
        MODEL,
        revision=REVISION,
        activation_fn=torch.nn.Identity(),
        model_kwargs={"dtype": getattr(torch, DTYPE), "use_safetensors": True},
    )
    if int(model.max_length) != MAX_LENGTH:
        raise ArtifactError(f"{MODEL} resolved a maximum length of {model.max_length}")
    return model


def strong_scores(model: Any, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
    """Old `judge.score`: raw logits, float32, aligned with `pairs`; non-finite refuses."""
    if not pairs:
        return np.zeros(0, dtype=np.float32)
    raw = model.predict(
        list(pairs), batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True
    )
    scores = np.asarray(raw, dtype=np.float32).reshape(-1)
    if scores.shape[0] != len(pairs) or not np.isfinite(scores).all():
        raise ArtifactError("the reranker returned a missing or non-finite score")
    return scores


class QwenReranker:
    """Qwen3-Reranker-0.6B as its model card scores a pair: the last position's `yes` against
    `no` logit. The score kept is log P(yes), monotone with the card's P(yes) and free of the
    ties its float32 saturation would make."""

    def __init__(self, dtype: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, padding_side="left"
        )
        kwargs = {} if dtype is None else {"dtype": getattr(torch, dtype)}
        model: Any = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, **kwargs
        )
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = self.tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)

    def predict(self, pairs: Sequence[tuple[str, str]], **_: Any) -> np.ndarray:
        out: list[float] = []
        for start in range(0, len(pairs), QWEN_BATCH_SIZE):
            texts = [
                f"<Instruct>: {QWEN_INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"
                for q, d in pairs[start : start + QWEN_BATCH_SIZE]
            ]
            inputs = self.tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=QWEN_MAX_LENGTH - len(self.prefix) - len(self.suffix),
            )
            inputs["input_ids"] = [self.prefix + ids + self.suffix for ids in inputs["input_ids"]]
            batch = self.tokenizer.pad(
                inputs, padding=True, return_tensors="pt", max_length=QWEN_MAX_LENGTH
            ).to(self.model.device)
            # Memory only (deviation 06.1): lm_head on the last position, no KV cache, and no
            # tensor of this batch carried into the next forward.
            with self.torch.no_grad():
                logits = self.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            pair = self.torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1)
            out.extend(self.torch.nn.functional.log_softmax(pair.float(), dim=1)[:, 1].tolist())
            del batch, logits, pair
        return np.asarray(out, dtype=np.float32)


def reorder(top: Sequence[str], scores: Sequence[float] | np.ndarray) -> list[str]:
    """Old `phase17.reorder`: score descending, ties by the first-stage rank."""
    entries = [
        (unit_id, float(s), rank) for rank, (unit_id, s) in enumerate(zip(top, scores, strict=True))
    ]
    return [unit_id for unit_id, _s, _r in sorted(entries, key=lambda e: (-e[1], e[2]))]


def rerank(args: argparse.Namespace) -> None:
    system = "g-r2" if args.qwen else "g-r"
    out = common.rankings_path(args.set, system)
    manifest = common.manifest_path(out)
    if manifest.exists():
        print(f"[INFO] {manifest} exists, nothing to do", flush=True)
        return
    first = common.rankings_path(args.set, "g-l")
    first_sha = sha256_file(first)
    spec, old, checks = common.open_set(args.set)
    questions, _answers = common.set_questions(old, checks, spec)
    tops = dict(common.read_rankings(first))
    if sorted(tops) != sorted(q.qid for q in questions):
        raise ArtifactError(f"{first} does not cover the set's questions")
    timer = common.Timer()
    started = time.perf_counter()
    wanted = {u for ranked in tops.values() for u in ranked[:DEPTH]}
    texts = {uid: unit.text for uid, unit in common.units_by_id(old, spec, wanted).items()}
    timer.add("read_texts", started)
    started = time.perf_counter()
    model = QwenReranker() if args.qwen else load_strong()
    timer.add("load_model", started)
    rankings: list[tuple[str, list[str]]] = []
    started = time.perf_counter()
    stream = iter(questions)
    while shard := list(islice(stream, SHARD_QUESTIONS)):
        pairs = [(q.question, texts[u]) for q in shard for u in tops[q.qid][:DEPTH]]
        scores = model.predict(pairs) if args.qwen else strong_scores(model, pairs)
        offset = 0
        for q in shard:
            top = tops[q.qid][:DEPTH]
            rankings.append((q.qid, reorder(top, scores[offset : offset + len(top)])))
            offset += len(top)
        print(f"[INFO] {args.set}: {len(rankings)} questions reranked", flush=True)
    timer.add("rerank", started)
    common.write_rankings(out, rankings)
    seconds = timer.seconds
    common.write_manifest(
        manifest,
        {
            "system": system,
            "set": args.set,
            "model": QWEN_MODEL if args.qwen else MODEL,
            "revision": QWEN_REVISION if args.qwen else REVISION,
            "settings": (
                {
                    "max_length": QWEN_MAX_LENGTH,
                    "batch_size": QWEN_BATCH_SIZE,
                    "instruction": QWEN_INSTRUCTION,
                    "score": "log P(yes)",
                }
                if args.qwen
                else {
                    "max_length": MAX_LENGTH,
                    "dtype": DTYPE,
                    "batch_size": BATCH_SIZE,
                    "matmul_precision": MATMUL_PRECISION,
                    "weights_sha256": WEIGHTS_SHA256,
                    "score": "raw logit",
                }
            ),
            "first_stage": {"file": str(first.name), "sha256": first_sha},
            "depth": DEPTH,
            "questions": len(questions),
            "pairs": sum(min(len(r), DEPTH) for r in tops.values()),
            "seconds": seconds,
            "rerank_seconds_per_question": seconds["rerank"] / max(len(questions), 1),
            "checks": checks.records,
        },
        [out],
    )
    print(f"[INFO] wrote {out} and {manifest}", flush=True)


def first_pairs(pairs_rows: Sequence[dict], scores_rows: Sequence[dict], n: int) -> list[dict]:
    """The first `n` pairs in file order, each with its stored score."""
    picked: list[dict] = []
    for pair_row, score_row in zip(pairs_rows, scores_rows, strict=True):
        if pair_row["qid"] != score_row["qid"] or pair_row["unit_ids"] != score_row["unit_ids"]:
            raise ArtifactError(f"pairs and scores disagree at {pair_row['qid']}")
        for unit_id, text, stored in zip(
            pair_row["unit_ids"], pair_row["texts"], score_row["scores"], strict=True
        ):
            picked.append(
                {
                    "qid": pair_row["qid"],
                    "unit_id": unit_id,
                    "question": pair_row["question"],
                    "text": text,
                    "stored": float(stored),
                }
            )
            if len(picked) == n:
                return picked
    raise ArtifactError(f"fewer than {n} pairs")


def read_pinned(path: Path, sha256: str, rows: int | None = None) -> list[dict]:
    """The first `rows` JSON lines of a gz file whose bytes match `sha256`."""
    measured = sha256_file(path)
    if measured != sha256:
        raise ArtifactError(f"{path}: sha256 {measured}, pinned {sha256}")
    out: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            out.append(json.loads(line))
            if rows is not None and len(out) == rows:
                break
    return out


def check_old(args: argparse.Namespace) -> None:
    out = common.PHASE_DIR / "checks" / "c3-strong-rescore.json"
    model = load_strong()
    report: dict[str, Any] = {"model": MODEL, "revision": REVISION, "tolerance": CHECK_TOLERANCE}
    worst = 0.0
    for set_name, (old_name, pairs_sha, scores_sha) in OLD17.items():
        old_dir = Path(args.old17_dir)
        # 100 pairs never need more than 100 rows; each row holds at least one pair.
        pairs_rows = read_pinned(old_dir / f"pairs-{old_name}.jsonl.gz", pairs_sha, CHECK_PAIRS)
        scores_rows = read_pinned(
            old_dir / f"scores-strong-{old_name}.jsonl.gz", scores_sha, CHECK_PAIRS
        )
        picked = first_pairs(pairs_rows, scores_rows, CHECK_PAIRS)
        fresh = strong_scores(model, [(p["question"], p["text"]) for p in picked])
        diffs = np.abs(fresh - np.asarray([p["stored"] for p in picked], dtype=np.float32))
        worst = max(worst, float(diffs.max()))
        report[set_name] = {
            "pairs": len(picked),
            "pairs_sha256": pairs_sha,
            "scores_sha256": scores_sha,
            "max_abs_diff": float(diffs.max()),
            "mean_abs_diff": float(diffs.mean()),
        }
    report["max_abs_diff"] = worst
    report["pass"] = worst <= CHECK_TOLERANCE
    report["git_commit"] = common.git_commit()
    report["gpu"] = common.gpu_name()
    write_json(out, report)
    print(f"[INFO] C3 max abs diff {worst:.3g}, pass {report['pass']}; wrote {out}", flush=True)


# --- Phase 04 `--pairs` mode (spec C4, C5; plan increment 5) ---------------------------------
PAIRS_DIR = common.config.PHASE04_PAIRS_DIR
SCORES_DIR = common.config.PHASE04_SCORES_DIR
CGROUP_DIR = Path("/sys/fs/cgroup")


def cgroup_memory(directory: Path = CGROUP_DIR) -> dict[str, int | None]:
    """cgroup v2 `memory.current` and `memory.peak` in bytes; None where the file is absent."""
    out: dict[str, int | None] = {}
    for name in ("memory.current", "memory.peak"):
        try:
            out[name] = int((directory / name).read_text("utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            out[name] = None
    return out


def pinned_pairs(set_name: str, pairs_dir: Path) -> tuple[dict[str, str], list[dict], list[dict]]:
    """The pairs and timing rows, each read only after its sha256 equals the one the laptop's
    `judge-pairs` manifest recorded."""
    manifest = json.loads((pairs_dir / f"{set_name}.manifest.json").read_text("utf-8"))
    pins = {
        name: manifest["outputs_sha256"][name]
        for name in (f"{set_name}.jsonl.gz", f"{set_name}.timing.jsonl.gz")
    }
    rows = [read_pinned(pairs_dir / name, sha256) for name, sha256 in pins.items()]
    return pins, rows[0], rows[1]


def timing_check(model: Any, rows: Sequence[dict]) -> dict[str, Any]:
    """Every pair of the timing sample scored fresh in one call (timed, as G-R was), then
    compared with the stored score wherever one exists (C4)."""
    pairs = [(row["question"], text) for row in rows for text in row["texts"]]
    started = time.perf_counter()
    fresh = strong_scores(model, pairs)
    seconds = time.perf_counter() - started
    stored = [s for row in rows for s in row["stored"]]
    if len(stored) != len(pairs):
        raise ArtifactError("timing rows: stored scores and texts disagree in length")
    kept = [(float(f), float(s)) for f, s in zip(fresh, stored, strict=True) if s is not None]
    diffs = [abs(f - s) for f, s in kept]
    worst = max(diffs, default=0.0)
    return {
        "tolerance": CHECK_TOLERANCE,
        "pairs_compared": len(kept),
        "max_abs_diff": worst,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "pass": bool(kept) and worst <= CHECK_TOLERANCE,
        "timing": {
            "questions": len(rows),
            "pairs": len(pairs),
            "seconds": seconds,
            "seconds_per_question": seconds / max(len(rows), 1),
            "seconds_per_100_pairs": seconds / max(len(pairs), 1) * 100,
        },
    }


def score_rows(model: Any, rows: Sequence[dict], say: Any = print) -> list[dict]:
    """`{"qid", "unit_ids", "scores"}` per pairs row, scored in shards of questions."""
    out: list[dict] = []
    stream = iter(rows)
    while shard := list(islice(stream, SHARD_QUESTIONS)):
        for row in shard:
            if len(row["texts"]) != len(row["unit_ids"]):
                raise ArtifactError(f"{row['qid']}: unit ids and texts disagree in length")
        scores = strong_scores(model, [(r["question"], t) for r in shard for t in r["texts"]])
        offset = 0
        for row in shard:
            size = len(row["unit_ids"])
            values = [float(v) for v in scores[offset : offset + size]]
            out.append({"qid": row["qid"], "unit_ids": list(row["unit_ids"]), "scores": values})
            offset += size
        say(f"[INFO] {len(out)} of {len(rows)} questions scored")
    return out


def score_pairs(
    set_name: str,
    *,
    load: Any = load_strong,
    pairs_dir: Path = PAIRS_DIR,
    scores_dir: Path = SCORES_DIR,
    say: Any = print,
) -> bool:
    """C4 then C5 for one set; False, with nothing scored, when the check fails. Write-once:
    an existing scores manifest means nothing to do; an existing check file is reused only when
    its inputs, model revision and weights, tolerance and commit match this run and no source
    change is uncommitted."""
    out = scores_dir / f"{set_name}.jsonl.gz"
    manifest = common.manifest_path(out)
    check_path = scores_dir / f"{set_name}.check.json"
    if manifest.exists():
        say(f"[INFO] {manifest} exists, nothing to do")
        return True
    if out.exists():
        raise ArtifactError(f"{out} exists without its manifest")
    provenance = common.git_provenance()
    timer = common.Timer()
    started = time.perf_counter()
    pins, rows, timing_rows = pinned_pairs(set_name, pairs_dir)
    timer.add("read", started)
    started = time.perf_counter()
    model = load()
    timer.add("load_model", started)
    same = {
        "inputs_sha256": pins,
        "model": MODEL,
        "revision": REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "tolerance": CHECK_TOLERANCE,
        "git_commit": provenance["git_commit"],
        "git_src_changes": provenance["git_src_changes"],
    }
    check = json.loads(check_path.read_text("utf-8")) if check_path.exists() else {}
    if provenance["git_src_changes"] or any(check.get(k) != v for k, v in same.items()):
        check = {"set": set_name, **timing_check(model, timing_rows), **same}
        check.update({**provenance, "gpu": common.gpu_name()})
        write_json(check_path, check)
    say(
        f"[INFO] {set_name} C4: {check['pairs_compared']} pairs, max abs diff "
        f"{check['max_abs_diff']:.3g}, pass {check['pass']}; {check_path}"
    )
    if not check["pass"]:
        return False
    started = time.perf_counter()
    scored = score_rows(model, rows, say)
    timer.add("score", started)
    write_bytes(out, common.gz_jsonl(scored))
    pairs = sum(len(r["unit_ids"]) for r in scored)
    common.write_manifest(
        manifest,
        {
            "set": set_name,
            "model": MODEL,
            "revision": REVISION,
            "settings": {
                "max_length": MAX_LENGTH,
                "dtype": DTYPE,
                "batch_size": BATCH_SIZE,
                "matmul_precision": MATMUL_PRECISION,
                "weights_sha256": WEIGHTS_SHA256,
                "score": "raw logit",
                "shard_questions": SHARD_QUESTIONS,
            },
            "git_src_changes": provenance["git_src_changes"],
            "inputs_sha256": pins,
            "check": {k: check[k] for k in ("pairs_compared", "max_abs_diff", "pass")},
            "questions": len(scored),
            "pairs": pairs,
            "seconds": timer.seconds,
            "score_seconds_per_100_pairs": timer.seconds["score"] / max(pairs, 1) * 100,
            "timing_sample": check["timing"],
            "cgroup_memory": cgroup_memory(),
        },
        [out, check_path],
    )
    say(f"[INFO] wrote {out} and {manifest}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=sorted(common.config.SETS))
    parser.add_argument("--qwen", action="store_true", help="G-R2: Qwen3-Reranker-0.6B")
    parser.add_argument("--check-old", action="store_true", help="criterion C3")
    parser.add_argument("--old17-dir", default="/workspace/old17")
    parser.add_argument("--pairs", choices=sorted(common.config.SETS), help="Phase 04 pod mode")
    args = parser.parse_args()
    if args.pairs is not None:
        if not score_pairs(args.pairs, say=lambda line: print(line, flush=True)):
            raise SystemExit(f"{args.pairs}: the C4 check failed; nothing scored")
    elif args.check_old:
        check_old(args)
    elif args.set is None:
        parser.error("--set is required")
    else:
        rerank(args)


if __name__ == "__main__":
    main()
