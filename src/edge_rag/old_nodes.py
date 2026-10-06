"""What the copied `local_extraction.py` (plan D6) imported from the old package, re-declared.

Read in place in the old project (`config.py`, `artifacts.py`, `corpus/pool.py`,
`nodes/extraction.py`, `nodes/normalization.py`, `nodes/index.build_node_index`) and copied
here with the same values, so the copy's only edit is its four import lines. The node index
is built into this repo's `retrieval.hop.NodeIndex`, the shape the hops already read.
"""

import json
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from edge_rag.artifacts import digest_of
from edge_rag.retrieval.hop import NodeIndex

# Old `config.py`, the values the copy reads (unchanged).
DEFAULT_SEED = 42
MAX_NODES_PER_TYPE = 50
MAX_NODE_CHARS = 120
NODE_TYPES: tuple[str, ...] = ("entity", "concept")
GLINER_EXTRACTOR = "gliner"
SPACY_EXTRACTOR = "spacy"
PHASE_7_EXTRACTORS: tuple[str, ...] = (GLINER_EXTRACTOR, SPACY_EXTRACTOR)
GLINER_MODEL = "urchade/gliner_medium-v2.1"
GLINER_REVISION = "40ec419335d09393f298636f471328b722c6da9e"
GLINER_TOKENIZER_MODEL = "microsoft/deberta-v3-base"
GLINER_TOKENIZER_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
GLINER_ALLOW_PATTERNS: tuple[str, ...] = ("gliner_config.json", "model.safetensors")
GLINER_TOKENIZER_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "spm.model",
    "tokenizer_config.json",
)
GLINER_LABELS: tuple[str, ...] = ("person", "organization", "location", "work of art", "event")
GLINER_THRESHOLD = 0.5
GLINER_FLAT_NER = True
GLINER_MULTI_LABEL = False
GLINER_BATCH_SIZE = 8
GLINER_MAX_LEN = 384
GLINER_MAX_WIDTH = 12
PROJECTION_PARAGRAPHS = 5_000_000
PHASE_9_EXTRACTION_SHARD_UNITS = 100_000

# spaCy candidate of the old Phase 7 (unused here, kept so the copy type-checks).
SPACY_MODEL = "en_core_web_sm"
SPACY_MODEL_VERSION = "3.8.0"
SPACY_MODEL_WHEEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
SPACY_EXCLUDE = (
    "tagger",
    "parser",
    "attribute_ruler",
    "lemmatizer",
    "senter",
)
SPACY_BATCH_SIZE = 64
SPACY_PROCESSES = 1
SPACY_LABELS = (
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "NORP",
    "ORG",
    "PERSON",
    "PRODUCT",
    "WORK_OF_ART",
)
SPACY_DROPPED_LABELS = (
    "CARDINAL",
    "DATE",
    "MONEY",
    "ORDINAL",
    "PERCENT",
    "QUANTITY",
    "TIME",
)


# Old `nodes/extraction.py`.
OK = "ok"
FAILED = "failed"


@dataclass(frozen=True)
class ExtractionRecord:
    unit_id: str
    model: str
    prompt_digest: str
    status: str
    entities: tuple[str, ...]
    concepts: tuple[str, ...]
    failure: str | None
    input_tokens: int
    output_tokens: int


# Old `corpus/pool.IndexingUnit`; a QASPER unit is one "sentence", its body.
@dataclass(frozen=True)
class IndexingUnit:
    unit_id: str
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def indexable_text(self) -> str:
        return f"{self.title}. {self.text}"


def write_text_atomic(path: Path | str, text: str) -> None:
    """Old `artifacts.write_text_atomic`: a temporary sibling, then an atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


# Old `nodes/normalization.py` (normalization-v1), escapes as in the original.
_QUOTES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "\xab": '"',
        "\xbb": '"',
    }
)
_SEPARATORS = re.compile("[-_/‐‑‒–—―−﹘﹣－]")
_EDGE_PUNCTUATION = re.compile(r"^[\W_]+|[\W_]+$")
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an) (?=\S)")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def normalize(form: str) -> str:
    text = unicodedata.normalize("NFKC", form)
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    text = text.casefold()
    text = text.translate(_QUOTES)
    text = _SEPARATORS.sub(" ", text)
    text = _EDGE_PUNCTUATION.sub("", text)
    text = _collapse(text)
    text = _LEADING_ARTICLE.sub("", text, count=1)
    return _collapse(text)


def build_node_index(records: Mapping[str, ExtractionRecord], unit_ids: Sequence[str]) -> NodeIndex:
    """Old `build_node_index`: normalized typed nodes, rows in unit order, node ids sorted by
    (type order, form); every unit needs a record, ok or failed (a failed row is empty)."""
    missing = [unit_id for unit_id in unit_ids if unit_id not in records]
    if missing:
        raise ValueError(f"the extraction does not cover {len(missing)} unit(s)")
    rows: list[set[tuple[str, str]]] = []
    for unit_id in unit_ids:
        record = records[unit_id]
        keys: set[tuple[str, str]] = set()
        if record.status == OK:
            for node_type, raw_forms in (("entity", record.entities), ("concept", record.concepts)):
                keys.update((node_type, form) for form in map(normalize, raw_forms) if form)
        rows.append(keys)
    order = {node_type: position for position, node_type in enumerate(NODE_TYPES)}
    vocabulary = sorted({key for keys in rows for key in keys}, key=lambda k: (order[k[0]], k[1]))
    node_id = {key: position for position, key in enumerate(vocabulary)}
    indptr = [0]
    indices: list[int] = []
    for keys in rows:
        indices.extend(sorted(node_id[key] for key in keys))
        indptr.append(len(indices))
    indptr_array = np.array(indptr, dtype=np.int64)
    indices_array = np.array(indices, dtype=np.int64)
    incidence = sparse.csr_matrix(
        (np.ones(len(indices)), indices_array, indptr_array),
        shape=(len(unit_ids), len(vocabulary)),
    )
    digest = digest_of(
        indptr_array, indices_array, json.dumps([list(unit_ids), vocabulary], ensure_ascii=True)
    )
    return NodeIndex(tuple(unit_ids), tuple(key[0] for key in vocabulary), incidence, digest)
