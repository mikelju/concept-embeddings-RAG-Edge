"""Deviation 06.3 patch of HippoRAG's OpenIE step: applied to a fixture string, idempotent,
and its 5% fallback guard fires."""

import json
import logging
import types

import pytest

from edge_rag.pod import ga2_openie_patch as p

FIXTURE = (
    "import json\n"
    + p.ANCHOR
    + "\n\nclass OpenIE:\n    def batch_openie(self, chunk_passages):\n"
    + "".join(old for old, _ in p.REPLACEMENTS)
)


def _fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("EDGE_FALLBACK_LOG", str(tmp_path / "fallback.jsonl"))
    namespace = {"json": json, "logger": logging.getLogger("test")}
    exec(p.HELPER, namespace)  # noqa: S102 - the patch's own helper source
    return namespace["_edge_fallback"]


def _result(chunk_id):
    return types.SimpleNamespace(chunk_id=chunk_id, response="prose", metadata={"error": "bad"})


def test_patch_replaces_every_block_and_is_idempotent():
    patched = p.patch(FIXTURE)
    assert p.MARKER in patched
    assert "raise RuntimeError(f\"NER failed" not in patched
    assert "raise RuntimeError(f\"Triple extraction failed" not in patched
    assert patched.count("def _edge_fallback(") == 1
    assert p.patch(patched) == patched


def test_patch_refuses_a_missing_block():
    with pytest.raises(SystemExit):
        p.patch(FIXTURE.replace(p.REPLACEMENTS[0][0], ""))


def test_guard_fires_only_above_five_percent(tmp_path, monkeypatch):
    fallback = _fallback(tmp_path, monkeypatch)
    results = [_result(f"c{i}") for i in range(6)]
    fallback("ner", ["c0", "c1", "c2", "c3", "c4"], results, 100)  # 5%: logged, no stop
    lines = (tmp_path / "fallback.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["chunk_id"] for line in lines] == ["c0", "c1", "c2", "c3", "c4"]
    with pytest.raises(RuntimeError, match="OPENIE_FALLBACK_GUARD"):
        fallback("ner", [r.chunk_id for r in results], results, 100)  # 6%
