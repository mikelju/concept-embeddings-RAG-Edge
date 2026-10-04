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
# The old repository sits beside this one (set OLD_DATA_ROOT when it does not, e.g. in a worktree).
OLD_DATA_ROOT_DEFAULT = REPO_ROOT.parent / "concept-embeddings-RAG" / "data"


def old_data_root() -> Path:
    """The old project's `data/`, read in place and never written."""
    return Path(os.environ.get(OLD_DATA_ROOT_ENV, str(OLD_DATA_ROOT_DEFAULT))).resolve()


# --- Retrieval pins (handover 5.3) ------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EMBEDDING_WEIGHTS_SHA256 = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
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


# Byte digests of the two fit files; the old project recorded none, so they are pinned here.
# pinned 2026-10-02 from the file in place, no recorded digest
FIT_FILE_SHA256 = {
    "phase10/fit.json": "7ed67df1e02867e06cdaa04a4739f4cd16d5a0fd1890b3370327fb0a5bfb5f33",
    "phase14/fit.json": "292e287f31250b3054aa7ea50d672f0c0912151c74181359ba8a3714e1cac421",
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
            # pinned 2026-10-02 from the file in place, no recorded digest
            "corpus.json": ("d6987a19baccae0a11d18dd0f2481a5d34cd3fde3465e7c66be54b90c3b85553"),
            "questions.json": ("f3f8b4f0df32be5578588aeb079656c1a1fbe943ef6709f5d3420ae62b09ef0f"),
            "token-counts.json": (
                "06a0b0da46566880362449eef728456b73d06f74ddf273adbf922b4ae9968413"
            ),
            "token-counts.npz": (
                "a1cee0afb461677ff5c73259e4f08fc4e0b8315051aac9e724685ede51fac277"
            ),
            "bm25.json": ("d93e3200c79036f091d9a73dd89bb232c1bdc68d1006047e4f374371e04bd8c5"),
            "embedding.json": ("cd6c427c872b32ab808bf2f389660a70316edc3fcc09bec83b45ec8d920062ef"),
            "gliner/extraction.json": (
                "847ba4abb2684b604d43834ef39ce58f798679aa2e98a396ff655e9836f8174f"
            ),
            "corpus.jsonl.gz": ("d982847985bcc5d9b26a5e2ce379e2e04d59fb9ead6b81e21ab0f1c168e93b0b"),
            "gliner/nodes-ecef84bdef739874.json": (
                "d3ed23c73f74b1da2af4616f347cc69b958b43053350ac6ed9e74b48ef8e81a1"
            ),
            "gliner/nodes-ecef84bdef739874.npz": (
                "4bd56af1082fc8813daeebd54442336df61391f7aaf79f6e88f55bcaab30bf82"
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
            # pinned 2026-10-02 from the file in place, no recorded digest
            "corpus.json": ("8b78ac1b72d45d7b5b960070c9df546d5311d58621d8f2f224b3bb70b2007dfc"),
            "questions.json": ("727027edca8fa6ea7155408bfd0c2eeb85aedeaeb5abefb184b1f2bc4f7ae54b"),
            "token-counts.json": (
                "9d997a112683fb9d3fce49029abb9a2861e2401bb3d1a9157f4b16624c4bb4f5"
            ),
            "token-counts.npz": (
                "cbd5e71bd531ca7787ecc482522c9ab9f27d784e7e9b8ca064de9d773d058847"
            ),
            "bm25/bm25.json": ("492296737f1add8b0a981fb1b5026c0c070ff33d82c40421e1087dc23cb79d26"),
            "cache/embedding.json": (
                "d5e159382d5051e857b145e1ad5fbb1555d5177d155ef2cc45ab856bd93cee40"
            ),
            "corpus.jsonl.gz": ("edb6c959331d314871a41571f37834a16ae5f6389d80819e5b5e3d993846bbca"),
            "nodes/nodes-c2257a622f49a8aa.json": (
                "600becc2079bc7f775111910a7202880a3107d6e53a1670233fd771829cf69d9"
            ),
            "nodes/nodes-c2257a622f49a8aa.npz": (
                "3b91ac4f0051cf16bc7ef2417ce0a3bfd968511223a174e4da4b31b0148e0d8f"
            ),
            "rankings-dense.jsonl.gz": (
                "7dc9cf9e1b4a8db4b02831a245ce617871c8474f0f367809b2dad01d82a47be7"
            ),
            "rankings-hybrid-bm25.jsonl.gz": (
                "6756c2bc991ff0b34b6923266f6d830d8f4fc44971c3dd12911cf8facd1af006"
            ),
            "rankings-hybrid-bm25-entity-hop.jsonl.gz": (
                "7e362cba717ac802d899063748743543fcbff9d17c8eaf78cb3b3f8df6850809"
            ),
            "rankings-hybrid-bm25-seeded-hop.jsonl.gz": (
                "ff327370b3966617997e301ea3ccef9838fd28e49a0ae2f772a8dcbfe1ec412b"
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
            # pinned 2026-10-02 from the file in place, no recorded digest
            "corpus.json": ("2bcc3a1f26a9941efef8e96fe17f48cd789d1cefdb138fa646f7f2f1053b3282"),
            "questions.json": ("7edc3fcfb8971562b9d1ceeed7ef4eb5ac6d396dd35bd0d10319375fd1f742d8"),
            "token-counts.json": (
                "23472538006cbc93278117ffdd9a8da18d27838bcd516c515b39ee81772e3426"
            ),
            "token-counts.npz": (
                "fe0f40a0ba7ea381f1b491afac928663d6e954c9a9f0845e42fd02b6d079ca90"
            ),
            "bm25/bm25.json": ("6b71237b5ddc7f6f2a4ae6b3ac86bb687ff34a4914ab79996599d4f34aa32633"),
            "cache/embedding.json": (
                "da70c08af453a3a9ecd492af7fbea6ce3ed1aea6cabc234145bb0ab289109eb9"
            ),
            "corpus.jsonl.gz": ("8ec12623a16fe4e7b2d083b1989a9bb811b1f198aca775506896e55ed22ea811"),
            "nodes/nodes-18dcb6f700ce0de5.json": (
                "9e636fa9443879e9389bd50c958fffeec59bad32908d648e540ae200838b6a58"
            ),
            "nodes/nodes-18dcb6f700ce0de5.npz": (
                "ceba92d69c2ec4c6cc452fe6de77033751eecb8e4a866385e4a005c8c5fd0dd5"
            ),
            "rankings-dense.jsonl.gz": (
                "afe403e31c8500e5ce3fd24241af0bfcb910d7d84ace5fb058935fc8e149768a"
            ),
            "rankings-hybrid-bm25.jsonl.gz": (
                "b2e2a79d641fc67eacdc789951ddd8d2ac7b22af2a698e0941d24f28680914d8"
            ),
            "rankings-hybrid-bm25-entity-hop.jsonl.gz": (
                "dc8eb3915ad02fd45a8624421f5809d8b361ea959ab488380180939730c2d197"
            ),
            "rankings-hybrid-bm25-seeded-hop.jsonl.gz": (
                "17a329d7516561619e75bc8abc6f3b0325dbdfeeb77a2b8a29ab8923bad3f9ad"
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

# --- Phase 02: scoring of ranking files ------------------------------------------------------
FS_BUDGETS: tuple[int, ...] = (1024, 2048, 4096)
FS_UNIT_KS: tuple[int, ...] = (2, 5, 20)
GOLD_SHARE_K = 5
RANKINGS_DIR = DATA_DIR / "phase02" / "rankings"

# Old Phase 17 J-strong lists: J(P10-B) and the union pool. The sha256 values are the ones
# recorded in phase17/outcome.json (provenance.reordered_files), matched in place 2026-10-02.
OLD_REFERENCES_DIR = "phase17"
OLD_REFERENCES: dict[str, dict[str, tuple[str, str]]] = {
    "hotpotqa-dev": {
        "j-p10b": (
            "reordered-strong-hotpotqa-hybrid-bm25.jsonl.gz",
            "8f200803e3f9af60daa3b1988e0a608c99acdd4a85d0597af3376aad0ebf166a",
        ),
        "j-union": (
            "reordered-strong-hotpotqa-union.jsonl.gz",
            "e52ed3824497613f1b3e98ab8c5a198e03b962acf54e8938f884a0988d76e60d",
        ),
    },
    "musique": {
        "j-p10b": (
            "reordered-strong-musique-hybrid-bm25.jsonl.gz",
            "77bb51bc615dfde79abd259527ff878b14d4d46e8e4899326144a909b3502edb",
        ),
        "j-union": (
            "reordered-strong-musique-union.jsonl.gz",
            "ebd5b33f5dc9e7fc87c72077ac2e2443b383561c533505882dddcbec2462515c",
        ),
    },
    "multihop-rag": {
        "j-p10b": (
            "reordered-strong-multihop-rag-hybrid-bm25.jsonl.gz",
            "90878a8f5ff27315ae978ef8526a5a24f6e62ec346fa09c3efe8268a015c5aaa",
        ),
        "j-union": (
            "reordered-strong-multihop-rag-union.jsonl.gz",
            "ff2fdb455efed234b475af99a58c2a128dcacd530a683cf7c4f66de588303fd1",
        ),
    },
}

# --- Phase 03: weight-free fusion with source diversity (spec, frozen 2026-10-03) ------------
PHASE03_DIR = DATA_DIR / "phase03"
PHASE03_RANKINGS_DIR = PHASE03_DIR / "rankings"
# The Phase 02 input lists the fusion reads, by the sha256 the spec froze.
P10A_SHA256 = {
    "hotpotqa-dev": "0b498fd1ec26359a4bb1f18eede3e3ecad4babaa8ecfffafffaa830aa657a330",
    "musique": "4810eb4e62e95233df523cf35f289e92df2f9276e01fc3e001ac2f2e29db7ad0",
    "multihop-rag": "bffefb1cb8bea53ee44c3b0acb22ffdec0f123f2ecd7d0cbbc863f145c9a2b0b",
}
GL_SHA256 = {
    "hotpotqa-dev": "b5b348ee15dba2017bfde6ae6c25e06047ab1f2b1ffd8feb038c16561795a68b",
    "musique": "8d7807345acf1eea111c943460c6a53c0e6dd167525c3cb53ebb3e8778f3b150",
    "multihop-rag": "791ed3653e1f90aa49d240df7c51767f7e846dbf5996930dbe6e1f6a72201c52",
}
# BGE-small question encoding is timed on the first questions of each set, one at a time.
ENCODING_SAMPLE = 200
RRF_K = 60
SOURCE_CAP = 2
SHINGLE_SIZE = 5
NEAR_DUPLICATE_JACCARD = 0.8

# --- Phase 04: judge over a pooled bag with the hop (spec, frozen 2026-10-04) -----------------
PHASE04_DIR = DATA_DIR / "phase04"
PHASE04_RANKINGS_DIR = PHASE04_DIR / "rankings"
PHASE04_PAIRS_DIR = PHASE04_DIR / "pairs"
PHASE04_SCORES_DIR = PHASE04_DIR / "scores"
# G-R's measured J-strong pair rates (Phase 02 manifests), the pod gate's rate per set.
GR_PAIRS_PER_SECOND = {"hotpotqa-dev": 151.0, "musique": 240.0, "multihop-rag": 268.0}
# The RTX 4090 Secure Cloud rate the pod gate projects with (USD per hour).
POD_USD_PER_HOUR = 0.74

# --- Phase 05: multi-seed convergent hop with a refined query (spec, frozen 2026-10-04) -------
PHASE05_DIR = DATA_DIR / "phase05"
PHASE05_RANKINGS_DIR = PHASE05_DIR / "rankings"
# Seeds per question: the top SEEDS_PER_LIST units of Dense, then of BM25, deduplicated.
SEEDS_PER_LIST = 5
