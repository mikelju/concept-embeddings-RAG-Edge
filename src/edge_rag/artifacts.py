"""Digests, the read-only old data root, and the only write path of the harness."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from edge_rag import config


class ArtifactError(Exception):
    """An artifact is missing, modified, or the write would land in the old data root."""


def digest_of(*parts: Any) -> str:
    """A sha256 over an ordered list of arrays, strings and numbers (old `artifacts.digest_of`)."""
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            hasher.update(np.ascontiguousarray(part).tobytes())
        elif isinstance(part, str):
            hasher.update(part.encode("utf-8"))
        else:
            hasher.update(repr(part).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            hasher.update(block)
    return hasher.hexdigest()


def is_under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def guard_write(path: Path) -> Path:
    """Refuse any write under the old data root; return the resolved target otherwise."""
    target = Path(path).resolve()
    if is_under(target, config.old_data_root()):
        raise ArtifactError(f"refusing to write under the old data root: {target}")
    return target


def write_json(path: Path, body: Any) -> Path:
    target = guard_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


class Checks:
    """The digests a run verified, in order, so the result says what it read."""

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def expect(self, what: str, measured: str, recorded: str) -> None:
        if measured != recorded:
            raise ArtifactError(f"{what}: measured {measured}, recorded {recorded}")
        self.records.append({"artifact": what, "digest": recorded})


class OldData:
    """One old set directory, opened read-only under the configured root."""

    def __init__(self, directory: str, root: Path | None = None) -> None:
        self.root = (root or config.old_data_root()).resolve()
        self.base = (self.root / directory).resolve()
        if not self.base.is_relative_to(self.root):
            raise ArtifactError(f"{directory} escapes the old data root")

    def path(self, relative: str) -> Path:
        target = (self.base / relative).resolve()
        if not target.is_relative_to(self.base):
            raise ArtifactError(f"{relative} escapes {self.base}")
        if not target.is_file():
            raise ArtifactError(f"missing old artifact: {target}")
        return target

    def json(self, relative: str) -> Any:
        with self.path(relative).open("rb") as handle:
            return json.loads(handle.read().decode("utf-8"))
