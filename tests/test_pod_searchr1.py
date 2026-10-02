"""The pure-Python parts of the G-A1 driver: infer.py's text handling and the batched loop."""

from edge_rag.pod import searchr1 as s


def hit(unit_id: str, title: str, text: str) -> dict:
    return {"document": {"id": unit_id, "contents": f"{title}\n{text}"}, "score": 1.0}


def test_get_query_takes_the_last_search():
    assert s.get_query("a <search> one </search> b <search>two</search>") == "two"
    assert s.get_query("no search here") is None
    assert s.get_query("<search>multi\nline</search>") == "multi\nline"


def test_passages_format_is_infer_py():
    out = s.passages_to_string(
        [hit("u1", "Paris", "Capital of France."), hit("u2", "Lyon", "City.")]
    )
    assert out == "Doc 1(Title: Paris) Capital of France.\nDoc 2(Title: Lyon) City.\n"


def test_user_prompt_adds_question_mark():
    assert s.user_prompt("  who is it ").endswith("Question: who is it?\n")
    assert s.user_prompt("who?").endswith("Question: who?\n")
    assert "<search> query </search>" in s.user_prompt("x?")


def test_answer_extraction_and_normalization():
    assert s.extract_answer("<think>x</think><answer> The Beatles </answer>") == "The Beatles"
    assert s.extract_answer("<answer>a</answer> then <answer>b</answer>") == "b"
    assert s.extract_answer("nothing") is None
    assert s.normalize_answer("The  Beatles!") == "beatles"


def test_dedup_keeps_first_retrieval_order():
    assert s.dedup_in_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_batched_loop_searches_absorbs_and_caps():
    corpus = {"q-a": [hit("u1", "T1", "x"), hit("u2", "T2", "y"), hit("u1", "T1", "x")]}
    calls: list[list[str]] = []

    def retrieve(queries):
        calls.append(list(queries))
        return [corpus.get(q, [hit("u9", "T9", "z")]) for q in queries]

    def generate(active):
        out = []
        for trace in active:
            if trace.qid == "answers":
                if trace.searches:
                    out.append(("<answer> Paris </answer>", True, 4))
                else:
                    out.append(("<think>t</think><search>q-a</search>", False, 6))
            else:  # never answers: searches forever
                out.append((f"<search>loop {trace.turns}</search>", False, 3))
        return out

    traces = [s.Trace("answers", "PROMPT1 "), s.Trace("loops", "PROMPT2 ")]
    s.run_turns(traces, generate, retrieve, prompt_tokens=lambda text: 0)
    done, loops = traces
    assert done.status == "answered"
    assert done.searches == ["q-a"]
    assert done.retrieved == ["u1", "u2", "u1"]
    assert s.dedup_in_order(done.retrieved) == ["u1", "u2"]
    assert "<information>Doc 1(Title: T1) x\n" in done.prompt
    assert done.generated.startswith("\n\n<think>t</think><search>q-a</search><information>")
    assert done.to_row("paris")["em"] is True
    assert done.tokens == 10
    assert loops.status == "search_cap"
    assert len(loops.searches) == s.MAX_SEARCHES
    assert loops.turns == s.MAX_SEARCHES + 1
    assert loops.to_row("")["em"] is None
    # the first turn batches both questions' searches into one request
    assert calls[0] == ["q-a", "loop 0"]


def test_turn_without_query_gets_empty_information():
    def generate(active):
        trace = active[0]
        if trace.turns == 0:
            return [("thinking only", False, 2)]
        return [("<answer>x</answer>", True, 2)]

    trace = s.Trace("q", "PROMPT")
    s.run_turns([trace], generate, lambda qs: [], prompt_tokens=lambda text: 0)
    # infer.py's get_query sees the prompt, whose instructions hold no <search> tag here
    assert trace.prompt.endswith("\n\nthinking only<information></information>\n\n")
    assert trace.searches == []
    assert trace.status == "answered"


def test_context_cap_stops_a_question():
    trace = s.Trace("q", "P")
    s.run_turns([trace], lambda a: [], lambda q: [], prompt_tokens=lambda t: s.MAX_MODEL_LEN)
    assert trace.status == "context_cap"


def test_agreement_counts():
    a = [
        {"qid": "1", "searches": [" x "], "answer": "The A"},
        {"qid": "2", "searches": [], "answer": None},
    ]
    b = [
        {"qid": "1", "searches": ["x"], "answer": "a"},
        {"qid": "2", "searches": ["y"], "answer": None},
    ]
    assert s.agreement(a, b) == {"questions": 2, "same_first_search": 1, "same_answer": 2}
