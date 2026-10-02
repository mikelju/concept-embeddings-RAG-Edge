"""Every deciding constant and every pin, in one place.

The values are copied from the old project (`concept_embeddings_rag/config.py`, the phase
manifests and `docs/refs/successor_project_handover.md` sections 5.2-5.4). An artifact read
from the old data root is accepted only if it matches the digest recorded here.
"""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# This repository's own data directory (git-ignored): results and any cache it writes.
DATA_DIR = REPO_ROOT / "data"

OLD_DATA_ROOT_ENV = "OLD_DATA_ROOT"
OLD_DATA_ROOT_DEFAULT = Path("C:/Python Projects/concept-embeddings-RAG/data")


def old_data_root() -> Path:
    """The old project's `data/`, read in place and never written."""
    return Path(os.environ.get(OLD_DATA_ROOT_ENV, str(OLD_DATA_ROOT_DEFAULT))).resolve()


# --- Retrieval pins (handover 5.3) ------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBEDDING_WEIGHTS_SHA256 = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
# The budget ruler is the BGE-small tokenizer at the same revision (the Phase 9 counter).
BUDGET_TOKENIZER_ID = EMBEDDING_MODEL
BUDGET_TOKENIZER_REVISION = EMBEDDING_REVISION
BM25_STOPWORDS = "en"
BM25_VERSION = "0.3.11"
GLINER_CONFIGURATION_DIGEST = "2f7864661b8ce7ff"
# The split string every inherited question cache key carries.
LEGACY_QUESTION_SPLIT = "standard-dev"

DEPTH = 100
BUDGET = 2048
READ_DEPTH = 10
ENTITY_TYPES: tuple[str, ...] = ("entity",)
RECALL_KS: tuple[int, ...] = (2, 5, 10, 20, 100)
NDCG_K = 10

# --- The four inherited systems and their frozen fusion (handover 5.3) -----------------------
P10A, P10B, P10C, P14 = "P10-A", "P10-B", "P10-C", "P14"
SYSTEMS: tuple[str, ...] = (P10A, P10B, P10C, P14)
WEIGHTS_P10B: tuple[float, float] = (0.5, 0.5)  # dense, bm25
WEIGHTS_TRIPLE: tuple[float, float, float] = (0.5, 0.3, 0.2)  # dense, bm25, hop
P14_ALPHA = 0.75
# `digest_of(json.dumps(fit, sort_keys=True))` of the two fits (phase15/integrity.json).
FIT_DIGESTS = {
    "phase10/fit.json": "66dfcec1c2b831ac580ee2257ce346b150a9ba5d9af4fc0955b4683bd46bdd96",
    "phase14/fit.json": "7ce07a171def50d85ca480383a978a2233cfa491ae048b9314a45850dabfe224",
}


@dataclass(frozen=True)
class CorpusSet:
    """One question set over one corpus, with every recorded digest its inputs must match."""

    name: str
    directory: str
    questions: int
    unit_set_hash: str
    ordered_unit_digest: str
    vectors_file: str
    vectors_digest: str
    question_vectors_file: str
    bm25_manifest: str
    bm25_index_digest: str
    token_counts_digest: str
    nodes_dir: str
    entity_index_digest: str
    question_digest: str
    mapping_digest: str
    # Relative path -> SHA-256 of the file bytes, from the old `phase*-sha256.txt` lists.
    file_sha256: dict[str, str]
    gate: dict[str, int]
    # Stored rankings used only as a cross-check of the live lists.
    stored_rankings: dict[str, str]


SETS: dict[str, CorpusSet] = {
    "hotpotqa-dev": CorpusSet(
        name="hotpotqa-dev",
        directory="phase9",
        questions=7405,
        unit_set_hash="ae23aef9a64a63c8",
        ordered_unit_digest="940932531ce629ecda945ae65511349e217e0971e9cf49272cea3cf32d139864",
        vectors_file="cache/embeddings-6179a736d88880f8.npz",
        vectors_digest="5c9bf6e3d57ed79aa4b8294bd3b99329a5a4d0a70707db208e72f5d90d80dc3a",
        question_vectors_file="cache/questions/embeddings-4894aceb0406c63a.npz",
        bm25_manifest="bm25.json",
        bm25_index_digest="51530fa6c429a3f6b655d56a969139cf142c346ea4f56358d50a72a2d7464815",
        token_counts_digest="83c4695b393ff1a060a2cc40875e62b75b82f9e09e84b75484fbd8230eedb36a",
        nodes_dir="gliner",
        entity_index_digest="0cf28b3f58950af3d9825e9d0e76a86e51c722689a5d03f6bde931697f3ebaaa",
        question_digest="aaa0bc26a482d91e34f001dc41d530ffa72116e4da27472769d6207ddf53b706",
        mapping_digest="e893fb6de67c0854fefa7bd0806355d4dd19dc2469392a195b175c0e53e6cee6",
        file_sha256={
            "cache/embeddings-6179a736d88880f8.npz": (
                "2136eca0b4685f41b63137234c75f5eae426954c449f93bd5c3119bb91e590a9"
            ),
            "cache/questions/embeddings-4894aceb0406c63a.npz": (
                "6b9b4408d5c1e3e692a6bc5174e307e62e81d884c4c120c6af5842ca41b1dc1c"
            ),
        },
        gate={P10A: 4125, P10B: 4536, P10C: 4801, P14: 5224},
        stored_rankings={},
    ),
    "musique": CorpusSet(
        name="musique",
        directory="phase15",
        questions=2417,
        unit_set_hash="15e2770c8ceb70f7",
        ordered_unit_digest="91fccb2429d62a574a24129efdcd4d0e3ebe74a322b90a840ed47d3ec0a4857a",
        vectors_file="cache/embeddings-5315e217da2c8cc3.npz",
        vectors_digest="d42b925877a9a58e7dae65b3105eb90119ff5841cd836e6603f5a5bf18aeb86d",
        question_vectors_file="cache/questions/embeddings-aeaa435a8f27cac1.npz",
        bm25_manifest="bm25/bm25.json",
        bm25_index_digest="62679ba9ca09b2781a7cbe4d43f4af8164ab91d46527eebb93935d105a506f32",
        token_counts_digest="4820063c77101c8e18be3c3f5a8944c1ae1b1849745df71f50ceec597e3e191d",
        nodes_dir="nodes",
        entity_index_digest="12daa8b5d11a66006baec3e1e88a1202604a23c4464e6936e0d38dd267af5f53",
        question_digest="33db3ec4f563eb6e97170a701fba62984477dd92007c2eb7c5c11e6405e8b2cc",
        mapping_digest="c51f364f3e1c25f37362f7efc2d279aaea5fb5515f18a90fd46d8fde355763a5",
        file_sha256={
            "cache/embeddings-5315e217da2c8cc3.npz": (
                "54713b5ccdefba21d34184b61fb43c2b42ad03a2f0a9dc07c69b631ffe9bb6e0"
            ),
            "cache/questions/embeddings-aeaa435a8f27cac1.npz": (
                "7f75e72d40123be5056d7aff84ef929d518e974fc97d3ddb607c528344f04de7"
            ),
            "nodes/extraction.json": (
                "42052941f7a809d504ec895dac691fd69e7f3deab8dd22ea77d278ad9bb1c551"
            ),
        },
        gate={P10A: 436, P10B: 524, P10C: 669, P14: 761},
        stored_rankings={
            P10A: "rankings-dense.jsonl.gz",
            P10B: "rankings-hybrid-bm25.jsonl.gz",
            P10C: "rankings-hybrid-bm25-entity-hop.jsonl.gz",
            P14: "rankings-hybrid-bm25-seeded-hop.jsonl.gz",
        },
    ),
    "multihop-rag": CorpusSet(
        name="multihop-rag",
        directory="phase16",
        questions=2255,
        unit_set_hash="4aa1f9aab57fd101",
        ordered_unit_digest="1520cc88fa86b217e8ae12c47f7b358bedbac2f47d22b1ae4961c34bf6fb1040",
        vectors_file="cache/embeddings-b66f546ea3d5b580.npz",
        vectors_digest="daf68122393b1cb6898836a515f6c12e0f94526dda11db633c2a8f759cfa3b44",
        question_vectors_file="cache/questions/embeddings-48566a479a335d6e.npz",
        bm25_manifest="bm25/bm25.json",
        bm25_index_digest="ba90bbe8f2140db19795a0f1ab8965d87eaf39e2c086d7a55846b73f453b09e4",
        token_counts_digest="c9cc71153182e0b026047fec6ab636cc4122ee435826948ed563e9bdf076fdc3",
        nodes_dir="nodes",
        entity_index_digest="e0b0ab2502eab3764fbff44dc1dbc90d2d993e6998eccf248aa6b1101e6ca43e",
        question_digest="b97acfd62b1ed2af8f15eebb3cd56d5c932de8ec43a2db01bcfb940523cb35ce",
        mapping_digest="a251643534964e3c0ea3ac88f226955bc1733cfc974a6654796f5b1d8e2af05b",
        file_sha256={
            "cache/embeddings-b66f546ea3d5b580.npz": (
                "6eb25b42396334ef42c3b35098497a02015cfa4abbf8a870c51979f4374c47f7"
            ),
            "cache/questions/embeddings-48566a479a335d6e.npz": (
                "84b6e00e8d674ca62c9f83d3467a69b3767c7d9aa7f4c3f981aadfcc6e9b801f"
            ),
            "nodes/extraction.json": (
                "de940266f4d19e52d07aa091a4a61904118ac1686c7ddc6cf0fffa6c5cfd0a8e"
            ),
        },
        gate={P10A: 338, P10B: 587, P10C: 522, P14: 513},
        stored_rankings={
            P10A: "rankings-dense.jsonl.gz",
            P10B: "rankings-hybrid-bm25.jsonl.gz",
            P10C: "rankings-hybrid-bm25-entity-hop.jsonl.gz",
            P14: "rankings-hybrid-bm25-seeded-hop.jsonl.gz",
        },
    ),
}
