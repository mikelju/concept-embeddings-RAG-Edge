"""Digests, the read-only old data root, and the only write path of the harness."""

import hashlib
import io
import json
from collections.abc import Mapping
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


def write_bytes(path: Path, body: bytes) -> Path:
    """The only write of the package: target and temporary file are both guarded."""
    target = guard_write(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = guard_write(target.with_name(target.name + ".tmp"))
    temporary.write_bytes(body)
    temporary.replace(target)
    return target


def write_json(path: Path, body: Any) -> Path:
    return write_bytes(path, json.dumps(body, indent=2, sort_keys=True).encode("utf-8"))


def keep_manifest(path: Path) -> None:
    """Keep an existing run manifest as `<stem>.<its written_utc>.json` beside it. A run calls
    this before it writes any output, so a collision stops the run with nothing overwritten."""
    if not path.exists():
        return
    old = path.read_bytes()
    stamp = str(json.loads(old).get("written_utc", "unstamped")).replace(":", "")
    kept = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    if kept.exists() and kept.read_bytes() != old:
        raise ArtifactError(f"{kept} exists with other content")
    write_bytes(kept, old)


def write_manifest(path: Path, body: Mapping[str, Any]) -> Path:
    """A run manifest is never lost: an existing one with other content is first kept."""
    if path.exists() and path.read_bytes() != json.dumps(body, indent=2, sort_keys=True).encode(
        "utf-8"
    ):
        keep_manifest(path)
    return write_json(path, body)


class Checks:
    """The digests a run verified, in order, so the result says what it read."""

    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def expect(self, what: str, measured: str, recorded: str) -> None:
        if measured != recorded:
            raise ArtifactError(f"{what}: measured {measured}, recorded {recorded}")
        self.records.append({"artifact": what, "digest": recorded})


class OldData:
    """One old set directory, opened read-only under the configured root.

    Every file read needs a pinned sha256 in `pins` (relative path -> digest) and is checked
    byte for byte before any parser sees it.
    """

    # Above this size a file is hashed by streaming and then opened, not held in memory.
    IN_MEMORY_LIMIT = 64 << 20

    def __init__(
        self,
        directory: str,
        pins: Mapping[str, str],
        checks: "Checks | None" = None,
        root: Path | None = None,
    ) -> None:
        self.root = (root or config.old_data_root()).resolve()
        self.base = (self.root / directory).resolve()
        if not self.base.is_relative_to(self.root):
            raise ArtifactError(f"{directory} escapes the old data root")
        self.pins = dict(pins)
        self.checks = checks
        self._verified: set[str] = set()

    def path(self, relative: str) -> Path:
        target = (self.base / relative).resolve()
        if not target.is_relative_to(self.base):
            raise ArtifactError(f"{relative} escapes {self.base}")
        if not target.is_file():
            raise ArtifactError(f"missing old artifact: {target}")
        return target

    def _expect(self, relative: str, measured: str) -> None:
        recorded = self.pins.get(relative)
        if recorded is None:
            raise ArtifactError(f"no pinned sha256 for old artifact {relative}")
        if measured != recorded:
            raise ArtifactError(f"sha256 {relative}: measured {measured}, pinned {recorded}")
        if relative not in self._verified:
            self._verified.add(relative)
            if self.checks is not None:
                self.checks.records.append({"artifact": f"sha256 {relative}", "digest": recorded})

    def verified_path(self, relative: str) -> Path:
        """The file's path once its bytes match the pin (hashed once per run); for large files."""
        target = self.path(relative)
        if relative not in self._verified:
            self._expect(relative, sha256_file(target))
        return target

    def read(self, relative: str) -> bytes:
        """The file's bytes, checked against the pin; the caller parses these same bytes."""
        target = self.path(relative)
        body = target.read_bytes()
        self._expect(relative, hashlib.sha256(body).hexdigest())
        return body

    def json(self, relative: str) -> Any:
        target = self.path(relative)
        if target.stat().st_size > self.IN_MEMORY_LIMIT:
            with self.verified_path(relative).open("rb") as handle:
                return json.load(handle)
        return json.loads(self.read(relative).decode("utf-8"))

    def arrays(self, relative: str, *names: str) -> list[np.ndarray]:
        """Named arrays of an `.npz` (no pickle), read from the bytes that were checked."""
        with np.load(io.BytesIO(self.read(relative)), allow_pickle=False) as payload:
            return [payload[name] for name in names]
