"""G-A1: Search-R1 (base 7B PPO) driven by batched vLLM over G-L's index (spec scope 5).

Prompt, stop sequences, EOS ids, query extraction, the `<information>` turn template and the
passage formatting are copied from Search-R1's `infer.py` (main branch, read 2026-10-02), as is
the chat-template use. Changes, all fixed by the spec: greedy decoding (plan D3), at most
`MAX_SEARCHES` retrievals per question, top-3 from G-L's PLAID index through the retrieval
server `python -m edge_rag.pod.colbert --serve` (same API as Search-R1's), and every active
question batched per turn.

    uv run --group pod python -m edge_rag.pod.searchr1 --set musique
    uv run --group pod python -m edge_rag.pod.searchr1 --set musique --compare-infer 50

`--compare-infer N` is criterion C4: `infer.py`'s own loop (transformers `generate`, patched
only to greedy and the same retriever) and the vLLM driver on the first N questions, each in
its own process, then first-search and normalized-answer agreement.
"""

import argparse
import json
import re
import string
import subprocess  # noqa: S404 - re-invokes this module with fixed arguments
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edge_rag.artifacts import sha256_file, write_bytes, write_json
from edge_rag.pod import common

MODEL = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
# HF Hub `sha` of the model, read 2026-10-02 (api/models/<MODEL>).
REVISION = "713cbe32f48a45a855da9cd09a0d980c2e3166e6"
SYSTEM = "g-a1"
TOPK = 3
MAX_NEW_TOKENS = 1024
MAX_SEARCHES = 8
MAX_MODEL_LEN = 16384
GPU_MEMORY_UTILIZATION = 0.80  # leaves room for the G-L retrieval server on the same card

# --- Copied from infer.py ------------------------------------------------------------------
CURR_EOS = [151645, 151643]  # for Qwen2.5 series models
CURR_SEARCH_TEMPLATE = "\n\n{output_text}<information>{search_results}</information>\n\n"
PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new "
    "information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    "<search> query </search> and it will return the top searched results between "
    "<information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: {question}\n"
)
TARGET_SEQUENCES = [
    "</search>",
    " </search>",
    "</search>\n",
    " </search>\n",
    "</search>\n\n",
    " </search>\n\n",
]


def get_query(text: str) -> str | None:
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    return None


def passages_to_string(retrieval_result: Sequence[dict]) -> str:
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return format_reference


def user_prompt(question: str) -> str:
    question = question.strip()
    if question[-1] != "?":
        question += "?"
    return PROMPT.format(question=question)


# --- Answers (Search-R1 `qa_em.py` normalization) -------------------------------------------
def extract_answer(generated: str) -> str | None:
    """The last `<answer>` span of the model's own text (the prompt's example is not in it)."""
    matches = re.findall(r"<answer>(.*?)</answer>", generated, re.DOTALL)
    return matches[-1].strip() if matches else None


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def dedup_in_order(unit_ids: Sequence[str]) -> list[str]:
    """The evidence ranking: retrieved units in retrieval order, later duplicates dropped."""
    return list(dict.fromkeys(unit_ids))


# --- Retrieval ------------------------------------------------------------------------------
def http_retriever(url: str) -> Callable[[list[str]], list[list[dict]]]:
    def retrieve(queries: list[str]) -> list[list[dict]]:
        payload = json.dumps({"queries": queries, "topk": TOPK, "return_scores": True})
        request = urllib.request.Request(  # noqa: S310 - local retrieval server
            url, data=payload.encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310
            return json.loads(response.read())["result"]

    return retrieve


@dataclass
class Trace:
    qid: str
    prompt: str  # grows as infer.py's does: chat prompt, then each turn and its information
    start: int = -1  # length of the chat prompt, set on creation
    searches: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    pending: str = ""  # this turn's text, until its search is absorbed
    final: str = ""  # the text of the last turn (ended by EOS or by the search cap)
    tokens: int = 0
    turns: int = 0
    status: str = "active"
    seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0:
            self.start = len(self.prompt)

    @property
    def generated(self) -> str:
        """Everything after the chat prompt: turns, information blocks and the last turn."""
        return self.prompt[self.start :] + self.final

    def to_row(self, gold: str) -> dict[str, Any]:
        answer = extract_answer(self.final) if self.status == "answered" else None
        em = None
        if gold:
            em = answer is not None and normalize_answer(answer) == normalize_answer(gold)
        return {
            "qid": self.qid,
            "searches": self.searches,
            "retrieved": self.retrieved,
            "generated": self.generated,
            "answer": answer,
            "gold": gold,
            "em": em,
            "tokens": self.tokens,
            "turns": self.turns,
            "status": self.status,
            "seconds": self.seconds,
        }


def step(trace: Trace, output_text: str, finished: bool, n_tokens: int) -> str | None:
    """One `infer.py` turn after generation: the query to search ("" when the turn holds
    none: infer.py then appends empty information), or None when the question is done.
    A turn that would make the (MAX_SEARCHES + 1)-th search ends the question."""
    trace.turns += 1
    trace.tokens += n_tokens
    if finished:
        trace.final = output_text
        trace.status = "answered"
        return None
    if len(trace.searches) >= MAX_SEARCHES:
        trace.final = output_text
        trace.status = "search_cap"
        return None
    trace.pending = output_text
    # infer.py: `get_query(tokenizer.decode(outputs[0]))`, the prompt plus this turn's text.
    return get_query(trace.prompt + output_text) or ""


def absorb(trace: Trace, query: str, hits: list[dict] | None) -> None:
    """infer.py: append the turn and its `<information>` block to the prompt."""
    if hits is not None:
        trace.searches.append(query)
        trace.retrieved.extend(str(h["document"]["id"]) for h in hits)
    results = passages_to_string(hits) if hits is not None else ""
    trace.prompt += CURR_SEARCH_TEMPLATE.format(output_text=trace.pending, search_results=results)
    trace.pending = ""


def run_turns(
    traces: list[Trace],
    generate: Callable[[list[Trace]], list[tuple[str, bool, int]]],
    retrieve: Callable[[list[str]], list[list[dict]]],
    prompt_tokens: Callable[[str], int],
) -> None:
    """The batched loop: every active question generates once per turn, then all their
    searches go to the retriever in one request."""
    started = time.perf_counter()
    while active := [t for t in traces if t.status == "active"]:
        for trace in active:
            if prompt_tokens(trace.prompt) + MAX_NEW_TOKENS > MAX_MODEL_LEN:
                trace.status = "context_cap"
                trace.seconds = time.perf_counter() - started
        active = [t for t in active if t.status == "active"]
        if not active:
            break
        searching: list[tuple[Trace, str]] = []
        for trace, (text, finished, n_tokens) in zip(active, generate(active), strict=True):
            query = step(trace, text, finished, n_tokens)
            if query is None:
                trace.seconds = time.perf_counter() - started
            else:
                searching.append((trace, query))
        with_query = [(t, q) for t, q in searching if q]
        hits = retrieve([q for _t, q in with_query]) if with_query else []
        by_trace = {id(t): h for (t, _q), h in zip(with_query, hits, strict=True)}
        for trace, query in searching:
            absorb(trace, query, by_trace.get(id(trace)))


# --- Engines --------------------------------------------------------------------------------
def chat_prompt(tokenizer: Any, question: str) -> str:
    prompt = user_prompt(question)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
    return prompt


def vllm_engine() -> tuple[Any, Callable[[list[Trace]], list[tuple[str, bool, int]]]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        revision=REVISION,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        stop=TARGET_SEQUENCES,
        include_stop_str_in_output=True,
    )

    def generate(active: list[Trace]) -> list[tuple[str, bool, int]]:
        outputs = llm.generate([t.prompt for t in active], params, use_tqdm=False)
        results = []
        for output in outputs:
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            finished = (bool(ids) and ids[-1] in CURR_EOS) or (
                completion.finish_reason == "stop" and completion.stop_reason is None
            )
            # infer.py decodes the generated ids, not a stop-truncated string.
            results.append((tokenizer.decode(ids, skip_special_tokens=True), finished, len(ids)))
        return results

    return tokenizer, generate


def hf_engine() -> tuple[Any, Callable[[list[Trace]], list[tuple[str, bool, int]]]]:
    """infer.py's model, stopping criterion and `generate` call, with `do_sample=False`."""
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model: Any = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # infer.py's device_map="auto" needs accelerate; on one GPU it is this move.
    model = model.to(device)

    class StopOnSequence(transformers.StoppingCriteria):
        def __init__(self, target_sequences: list[str], tokenizer: Any) -> None:
            self.target_ids = [
                tokenizer.encode(s, add_special_tokens=False) for s in target_sequences
            ]
            self.target_lengths = [len(t) for t in self.target_ids]

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            targets = [torch.as_tensor(t, device=input_ids.device) for t in self.target_ids]
            if input_ids.shape[1] < min(self.target_lengths):
                return False
            for i, target in enumerate(targets):
                if torch.equal(input_ids[0, -self.target_lengths[i] :], target):
                    return True
            return False

    stopping = transformers.StoppingCriteriaList([StopOnSequence(TARGET_SEQUENCES, tokenizer)])

    def generate(active: list[Trace]) -> list[tuple[str, bool, int]]:
        results = []
        for trace in active:  # infer.py runs one question at a time
            input_ids = tokenizer.encode(trace.prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                stopping_criteria=stopping,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            generated = outputs[0][input_ids.shape[1] :]
            finished = outputs[0][-1].item() in CURR_EOS
            text = str(tokenizer.decode(generated, skip_special_tokens=True))
            results.append((text, finished, int(generated.shape[0])))
        return results

    return tokenizer, generate


# --- Runs -----------------------------------------------------------------------------------
def output_dir(set_name: str, tag: str) -> Path:
    return common.PHASE_DIR / "searchr1" / set_name / tag


def run(args: argparse.Namespace) -> None:
    tag = args.tag or SYSTEM
    out_dir = output_dir(args.set, tag)
    manifest = out_dir / "manifest.json"
    if manifest.exists():
        print(f"[INFO] {manifest} exists, nothing to do", flush=True)
        return
    spec, old, checks = common.open_set(args.set)
    questions, answers = common.set_questions(old, checks, spec)
    if args.limit is not None:
        questions = questions[: args.limit]
    timer = common.Timer()
    started = time.perf_counter()
    tokenizer, generate = hf_engine() if args.engine == "hf" else vllm_engine()
    timer.add("load_model", started)
    traces = [Trace(qid=q.qid, prompt=chat_prompt(tokenizer, q.question)) for q in questions]
    started = time.perf_counter()
    run_turns(
        traces,
        generate,
        http_retriever(args.retriever),
        lambda text: len(tokenizer.encode(text)),
    )
    timer.add("run", started)
    rows = [t.to_row(answers[t.qid]) for t in traces]
    trace_file = out_dir / "trace.jsonl.gz"
    write_bytes(trace_file, common.gz_jsonl(rows))
    outputs = [trace_file]
    if args.limit is None and args.engine == "vllm":
        ranking = common.rankings_path(args.set, SYSTEM)
        common.write_rankings(ranking, [(t.qid, dedup_in_order(t.retrieved)) for t in traces])
        outputs.append(ranking)
    graded = [r["em"] for r in rows if r["em"] is not None]
    seconds = timer.seconds
    body = {
        "system": SYSTEM,
        "set": args.set,
        "engine": args.engine,
        "model": MODEL,
        "revision": REVISION,
        "settings": {
            "decoding": "greedy",
            "max_new_tokens_per_turn": MAX_NEW_TOKENS,
            "max_searches": MAX_SEARCHES,
            "topk": TOPK,
            "max_model_len": MAX_MODEL_LEN,
            "retriever": "G-L PLAID index via edge_rag.pod.colbert --serve",
        },
        "questions": len(traces),
        "status_counts": {
            s: sum(t.status == s for t in traces) for s in sorted({t.status for t in traces})
        },
        "searches_total": sum(len(t.searches) for t in traces),
        "tokens_generated": sum(t.tokens for t in traces),
        "answer_em": {"graded": len(graded), "exact": sum(graded), "label": "context only"},
        "seconds": seconds,
        "seconds_per_question": seconds["run"] / max(len(traces), 1),
        "checks": checks.records,
    }
    common.write_manifest(manifest, body, outputs)
    if len(outputs) == 2:
        common.write_manifest(common.manifest_path(outputs[1]), body, outputs)
    print(f"[INFO] wrote {out_dir}", flush=True)


def agreement(a_rows: Sequence[dict], b_rows: Sequence[dict]) -> dict[str, Any]:
    """C4 counts: same first search query (stripped) and same normalized answer."""
    first = answer = 0
    for a, b in zip(a_rows, b_rows, strict=True):
        if a["qid"] != b["qid"]:
            raise ValueError(f"question order differs at {a['qid']}")
        a_first = a["searches"][0].strip() if a["searches"] else None
        b_first = b["searches"][0].strip() if b["searches"] else None
        first += a_first == b_first
        a_ans = None if a["answer"] is None else normalize_answer(a["answer"])
        b_ans = None if b["answer"] is None else normalize_answer(b["answer"])
        answer += a_ans == b_ans
    return {"questions": len(a_rows), "same_first_search": first, "same_answer": answer}


def compare_infer(args: argparse.Namespace) -> None:
    out = common.PHASE_DIR / "checks" / f"c4-infer-vs-vllm-{args.set}.json"
    files = {}
    for engine in ("hf", "vllm"):
        tag = f"compare-{engine}-{args.compare_infer}"
        subprocess.run(  # noqa: S603 - this module, fixed arguments
            [
                sys.executable,
                "-m",
                "edge_rag.pod.searchr1",
                "--set",
                args.set,
                "--engine",
                engine,
                "--limit",
                str(args.compare_infer),
                "--tag",
                tag,
                "--retriever",
                args.retriever,
            ],
            check=True,
        )
        files[engine] = output_dir(args.set, tag) / "trace.jsonl.gz"
    counts = agreement(common.read_jsonl_gz(files["hf"]), common.read_jsonl_gz(files["vllm"]))
    counts.update(
        {
            "set": args.set,
            "thresholds": {"same_first_search": 45, "same_answer": 40, "of": 50},
            "pass": counts["same_first_search"] >= 45 and counts["same_answer"] >= 40,
            "traces": {
                e: {"file": str(p.relative_to(common.PHASE_DIR)), "sha256": sha256_file(p)}
                for e, p in files.items()
            },
            "git_commit": common.git_commit(),
        }
    )
    write_json(out, counts)
    print(f"[INFO] C4 {counts}; wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, choices=sorted(common.config.SETS))
    parser.add_argument("--engine", choices=("vllm", "hf"), default="vllm")
    parser.add_argument("--limit", type=int, default=None, help="first N questions in file order")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--retriever", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--compare-infer", type=int, default=None, metavar="N")
    args = parser.parse_args()
    if args.compare_infer is not None:
        compare_infer(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
