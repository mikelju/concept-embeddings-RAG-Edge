"""Phase 06 G-R2 on the pod: Qwen3-Reranker-0.6B over the uploaded G-L top-100 pairs.

The scorer is `rerank.QwenReranker` (model card prompt, default instruction, last-position
`yes` against `no` logit, kept as log P(yes)) in float32, batch 16, pairs in file order.
`--check` is spec C3, run once before any scoring, on the first 100 pairs in pair-file order
of each of the three sets (300 pairs):
- fidelity: the model card's reference code, verbatim except loaded in float32 (D7), one
  pair per call (no padding), against the batched scorer;
- determinism: the same 300 pairs scored in reversed order (other batches, other padding).
Both compare the card's score, P(yes), within 1e-3 absolute; log P(yes) differences are
recorded too. `--score <set>` refuses unless that check passed on the same pairs files and
commit. No corpus is read.

    uv run --group pod python -m edge_rag.pod.qwen_rerank --check
    uv run --group pod python -m edge_rag.pod.qwen_rerank --score musique
"""

import argparse
import json
import time
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from edge_rag.artifacts import ArtifactError, write_bytes, write_json
from edge_rag.pod import common
from edge_rag.pod.rerank import (
    CHECK_PAIRS,
    CHECK_TOLERANCE,
    QWEN_BATCH_SIZE,
    QWEN_INSTRUCTION,
    QWEN_MAX_LENGTH,
    QWEN_MODEL,
    QWEN_PREFIX,
    QWEN_REVISION,
    QWEN_SUFFIX,
    SHARD_QUESTIONS,
    QwenReranker,
    cgroup_memory,
    read_pinned,
)

config = common.config
DTYPE = "float32"
SETS = ("musique", "multihop-rag", "hotpotqa-dev")
PAIRS_DIR = config.PHASE06_PAIRS_DIR
SCORES_DIR = config.PHASE06_SCORES_DIR
CHECK_PATH = config.PHASE06_CHECKS_DIR / "c3-g-r2.json"
SETTINGS = {
    "dtype": DTYPE,
    "batch_size": QWEN_BATCH_SIZE,
    "max_length": QWEN_MAX_LENGTH,
    "instruction": QWEN_INSTRUCTION,
    "score": "log P(yes)",
    "shard_questions": SHARD_QUESTIONS,
}


def pins(set_name: str, pairs_dir: Path = PAIRS_DIR) -> dict[str, str]:
    manifest = json.loads((pairs_dir / f"{set_name}.manifest.json").read_text("utf-8"))
    return dict(manifest["outputs_sha256"])


def first_pairs(rows: Sequence[dict], n: int) -> list[tuple[str, str]]:
    """The first `n` (question, text) pairs in pair-file order."""
    out = [(row["question"], text) for row in rows for text in row["texts"]][:n]
    if len(out) < n:
        raise ArtifactError(f"fewer than {n} pairs")
    return out


class CardReference:
    """The model card's reference code, verbatim except the float32 load (D7):
    `format_instruction`, `process_inputs`, `compute_logits`; called with one pair at a time."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, padding_side="left"
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL, revision=QWEN_REVISION, dtype=getattr(torch, DTYPE)
        )
        self.model = model.cuda().eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)

    def format_instruction(self, instruction: str | None, query: str, doc: str) -> str:
        if instruction is None:
            instruction = QWEN_INSTRUCTION
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs: list[str]) -> Any:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=QWEN_MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=QWEN_MAX_LENGTH
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def compute_logits(self, inputs: Any) -> list[float]:
        with self.torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = self.torch.stack([false_vector, true_vector], dim=1)
            batch_scores = self.torch.nn.functional.log_softmax(batch_scores, dim=1)
            return batch_scores[:, 1].exp().tolist()

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        out = [
            self.compute_logits(self.process_inputs([self.format_instruction(None, q, d)]))[0]
            for q, d in pairs
        ]
        return np.asarray(out, dtype=np.float64)


def compare(batched: np.ndarray, reversed_: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """C3 on P(yes): batched against the card's reference and against its reversed rescoring;
    `batched` and `reversed_` are log P(yes), aligned; `reference` is P(yes)."""
    p_batched, p_reversed = np.exp(batched.astype(np.float64)), np.exp(reversed_.astype(np.float64))
    fidelity = np.abs(p_batched - reference)
    determinism = np.abs(p_batched - p_reversed)
    log_det = np.abs(batched.astype(np.float64) - reversed_.astype(np.float64))
    return {
        "pairs": int(len(batched)),
        "tolerance": CHECK_TOLERANCE,
        "space": "P(yes), the model card's score",
        "fidelity_max_abs_diff": float(fidelity.max()),
        "fidelity_mean_abs_diff": float(fidelity.mean()),
        "determinism_max_abs_diff": float(determinism.max()),
        "determinism_log_p_max_abs_diff": float(log_det.max()),
        "fidelity_pass": bool(fidelity.max() <= CHECK_TOLERANCE),
        "determinism_pass": bool(determinism.max() <= CHECK_TOLERANCE),
        "pass": bool(fidelity.max() <= CHECK_TOLERANCE and determinism.max() <= CHECK_TOLERANCE),
    }


def run_check(pairs_dir: Path = PAIRS_DIR, out: Path = CHECK_PATH) -> dict[str, Any]:
    provenance = common.git_provenance()
    pinned: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for set_name in SETS:
        set_pins = pins(set_name, pairs_dir)
        pinned.update(set_pins)
        name = f"{set_name}.jsonl.gz"
        rows = read_pinned(pairs_dir / name, set_pins[name], CHECK_PAIRS)
        pairs.extend(first_pairs(rows, CHECK_PAIRS))
    scorer = QwenReranker(DTYPE)
    started = time.perf_counter()
    batched = scorer.predict(pairs)
    reversed_ = scorer.predict(pairs[::-1])[::-1]
    reference_model = CardReference()
    reference = reference_model.score(pairs)
    report = {
        "model": QWEN_MODEL,
        "revision": QWEN_REVISION,
        "settings": SETTINGS,
        "reference_dtype": str(reference_model.model.dtype),
        "rule": f"first {CHECK_PAIRS} pairs in pair-file order of each of {list(SETS)}",
        "inputs_sha256": pinned,
        **compare(batched, reversed_, reference),
        "seconds": time.perf_counter() - started,
        **provenance,
        "gpu": common.gpu_name(),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out, report)
    print(
        f"[INFO] C3 fidelity {report['fidelity_max_abs_diff']:.3g}, determinism "
        f"{report['determinism_max_abs_diff']:.3g}, pass {report['pass']}; wrote {out}",
        flush=True,
    )
    return report


def score_set(
    set_name: str,
    *,
    load: Any = lambda: QwenReranker(DTYPE),
    pairs_dir: Path = PAIRS_DIR,
    scores_dir: Path = SCORES_DIR,
    check_path: Path = CHECK_PATH,
    say: Any = print,
) -> None:
    """Score one set's pairs file once; refuses without a passed C3 check on the same pairs
    files at the same commit with no uncommitted source change."""
    out = scores_dir / f"{set_name}.jsonl.gz"
    manifest = common.manifest_path(out)
    if manifest.exists():
        say(f"[INFO] {manifest} exists, nothing to do")
        return
    if out.exists():
        raise ArtifactError(f"{out} exists without its manifest")
    provenance = common.git_provenance()
    check = json.loads(check_path.read_text("utf-8"))
    set_pins = pins(set_name, pairs_dir)
    if not check["pass"]:
        raise ArtifactError("the C3 check did not pass; nothing is scored")
    if any(check["inputs_sha256"].get(k) != v for k, v in set_pins.items()):
        raise ArtifactError(f"{set_name}: the C3 check read other pairs files")
    if provenance["git_src_changes"] or check["git_commit"] != provenance["git_commit"]:
        raise ArtifactError("the C3 check ran at another commit or with uncommitted changes")
    timer = common.Timer()
    started = time.perf_counter()
    name = f"{set_name}.jsonl.gz"
    rows = read_pinned(pairs_dir / name, set_pins[name])
    timer.add("read", started)
    started = time.perf_counter()
    model = load()
    timer.add("load_model", started)
    started = time.perf_counter()
    scored: list[dict] = []
    stream = iter(rows)
    while shard := list(islice(stream, SHARD_QUESTIONS)):
        scores = model.predict([(r["question"], t) for r in shard for t in r["texts"]])
        if len(scores) != sum(len(r["texts"]) for r in shard) or not np.isfinite(scores).all():
            raise ArtifactError("the reranker returned a missing or non-finite score")
        offset = 0
        for row in shard:
            size = len(row["unit_ids"])
            values = [float(v) for v in scores[offset : offset + size]]
            scored.append({"qid": row["qid"], "unit_ids": list(row["unit_ids"]), "scores": values})
            offset += size
        say(f"[INFO] {set_name}: {len(scored)} of {len(rows)} questions scored")
    timer.add("score", started)
    write_bytes(out, common.gz_jsonl(scored))
    pairs = sum(len(r["unit_ids"]) for r in scored)
    common.write_manifest(
        manifest,
        {
            "set": set_name,
            "system": "g-r2",
            "model": QWEN_MODEL,
            "revision": QWEN_REVISION,
            "settings": SETTINGS,
            "git_src_changes": provenance["git_src_changes"],
            "inputs_sha256": set_pins,
            "check": {
                k: check[k] for k in ("fidelity_max_abs_diff", "determinism_max_abs_diff", "pass")
            },
            "questions": len(scored),
            "pairs": pairs,
            "seconds": timer.seconds,
            "score_seconds_per_question": timer.seconds["score"] / max(len(scored), 1),
            "cgroup_memory": cgroup_memory(),
        },
        [out, check_path],
    )
    say(f"[INFO] wrote {out} and {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="spec C3, before any scoring")
    parser.add_argument("--score", choices=SETS)
    args = parser.parse_args()
    if args.check:
        if not run_check()["pass"]:
            raise SystemExit("C3 failed; G-R2 is aborted (spec), nothing scored")
    elif args.score:
        score_set(args.score, say=lambda line: print(line, flush=True))
    else:
        parser.error("--check or --score is required")


if __name__ == "__main__":
    main()
