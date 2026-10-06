"""Deviation 06.3 patch of HippoRAG 2bfd831 for the Phase 06 G-A2 session.

Upstream `OpenIE.batch_openie` (information_extraction/openie_openai.py) raises when any chunk's
NER or triple answer fails to parse. Llama-3.1-8B-Instruct answers about 2.6% of the probe chunks
in prose, so the run could not finish. This patch treats such a chunk as one with no entities and
no triples: no guided decoding, no retries, and the triple step is skipped for a NER fallback
chunk, so every other chunk's LLM call and output are unchanged.

Each fallback chunk is appended as one JSON line to `$EDGE_FALLBACK_LOG` (stage, chunk id, error,
response head) and counted in a `[WARN] OPENIE_FALLBACK` line. Guard: more than 5% of the chunks
of one step falling back raises `OPENIE_FALLBACK_GUARD` and stops the run.

Standard library only, idempotent, run by `scripts/pod_ga2.sh` after the C3 check:

    python3 src/edge_rag/pod/ga2_openie_patch.py /workspace/HippoRAG/src/hipporag/information_extraction/openie_openai.py
"""

import sys
from pathlib import Path

MARKER = "# edge-rag deviation 06.3"

HELPER = '''

def _edge_fallback(stage, failed_ids, results, total):  # edge-rag deviation 06.3
    import os
    if not failed_ids:
        return
    path = os.environ.get("EDGE_FALLBACK_LOG", "openie_fallback.jsonl")
    failed = set(failed_ids)
    with open(path, "a", encoding="utf-8") as handle:
        for result in results:
            if result.chunk_id in failed:
                handle.write(json.dumps({
                    "stage": stage,
                    "chunk_id": result.chunk_id,
                    "error": str(result.metadata.get("error", ""))[:300],
                    "response_head": (result.response or "")[:300],
                }) + "\\n")
    share = len(failed_ids) / max(total, 1)
    print(f"[WARN] OPENIE_FALLBACK stage {stage} chunks {len(failed_ids)} of {total} share {share:.4f}", flush=True)
    logger.warning(f"OpenIE fallback (deviation 06.3): {stage} {len(failed_ids)} of {total}: {failed_ids}")
    if share > 0.05:
        raise RuntimeError(f"OPENIE_FALLBACK_GUARD {stage} {len(failed_ids)} of {total} chunks exceed 5%")
'''

REPLACEMENTS = [
    (
        '''        failed_ner_chunk_ids = [result.chunk_id for result in ner_results_list if result.metadata.get("error")]
        if failed_ner_chunk_ids:
            raise RuntimeError(f"NER failed for {len(failed_ner_chunk_ids)} chunk(s): {failed_ner_chunk_ids}")
''',
        '''        failed_ner_chunk_ids = [result.chunk_id for result in ner_results_list if result.metadata.get("error")]
        _edge_fallback("ner", failed_ner_chunk_ids, ner_results_list, len(chunk_passages))  # edge-rag deviation 06.3
''',
    ),
    (
        '''                for ner_result in ner_results_list
            }
''',
        '''                for ner_result in ner_results_list
                if ner_result.chunk_id not in failed_ner_chunk_ids  # edge-rag deviation 06.3
            }
''',
    ),
    (
        '''        failed_triple_chunk_ids = [result.chunk_id for result in triple_results_list if result.metadata.get("error")]
        if failed_triple_chunk_ids:
            raise RuntimeError(f"Triple extraction failed for {len(failed_triple_chunk_ids)} chunk(s): {failed_triple_chunk_ids}")
''',
        '''        failed_triple_chunk_ids = [result.chunk_id for result in triple_results_list if result.metadata.get("error")]
        _edge_fallback("triple", failed_triple_chunk_ids, triple_results_list, len(chunk_passages))  # edge-rag deviation 06.3
        triple_results_list.extend(  # edge-rag deviation 06.3: a NER fallback chunk has no triples
            TripleRawOutput(chunk_id=chunk_id, response="", triples=[],
                            metadata={"error": "NER fallback, triple step skipped (deviation 06.3)"})
            for chunk_id in failed_ner_chunk_ids
        )
''',
    ),
]

ANCHOR = "logger = get_logger(__name__)\n"


def patch(text: str) -> str:
    if MARKER in text:
        return text
    if text.count(ANCHOR) != 1:
        raise SystemExit("[ERROR] logger anchor not found exactly once")
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise SystemExit(f"[ERROR] block not found exactly once: {old.strip()[:80]}")
        text = text.replace(old, new)
    return text.replace(ANCHOR, ANCHOR + HELPER)


def main(argv: list[str]) -> int:
    path = Path(argv[1])
    before = path.read_text(encoding="utf-8")
    after = patch(before)
    if after != before:
        path.write_text(after, encoding="utf-8")
        print(f"[OK] patched {path}")
    else:
        print(f"[OK] already patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
