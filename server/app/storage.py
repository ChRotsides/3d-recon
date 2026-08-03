"""Blob storage and signed URLs (PLAN.md §2, §4/Phase 2).

The plan names S3 for blobs. Everything here goes through a `BlobStore`
protocol with a filesystem implementation, so the API and the worker can be
tested end to end without an object store — and so a self-hosted deployment
stays possible without rewriting the service.
"""

from __future__ import annotations

import hashlib
import hmac
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    def put(self, key: str, data: bytes | BinaryIO) -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def size(self, key: str) -> int: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def local_path(self, key: str) -> Path | None: ...


class LocalBlobStore:
    """Filesystem-backed blobs.

    `local_path` exists because the reconstruction worker wants a directory it
    can hand to the existing `BundleDir` loader. An S3 implementation returns
    None there and the worker stages to a temp dir instead — the difference is
    confined to one method rather than smeared through the pipeline.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are server-generated, but a traversal here would let a client
        # write anywhere on the worker, so it is checked rather than assumed.
        p = (self.root / key).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"blob key escapes the store root: {key!r}")
        return p

    def put(self, key: str, data: bytes | BinaryIO) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes | bytearray):
            p.write_bytes(bytes(data))
        else:
            with p.open("wb") as fh:
                shutil.copyfileobj(data, fh)
        return key

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def delete_prefix(self, prefix: str) -> int:
        base = self._path(prefix)
        if not base.exists():
            return 0
        n = sum(1 for p in base.rglob("*") if p.is_file())
        shutil.rmtree(base)
        return n

    def local_path(self, key: str) -> Path | None:
        return self._path(key)


@dataclass
class SignedUrl:
    url: str
    expires_at: float


class UrlSigner:
    """HMAC-signed, expiring download URLs (PLAN.md §4/Phase 2).

    Artifacts are a customer's property scan — floor plans and photoreal
    geometry of where they live. Serving them from a guessable path would make
    every scan world-readable to anyone who learns a job id, so links are signed
    and time-limited.
    """

    def __init__(self, secret: str, default_ttl_s: int = 3600) -> None:
        if not secret or secret == "change-me":
            raise ValueError("refusing to sign artifact URLs with a default secret")
        self._secret = secret.encode()
        self.default_ttl_s = default_ttl_s

    def _mac(self, key: str, expires: int) -> str:
        msg = f"{key}\n{expires}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()[:32]

    def sign(self, key: str, ttl_s: int | None = None) -> SignedUrl:
        expires = int(time.time()) + int(ttl_s or self.default_ttl_s)
        sig = self._mac(key, expires)
        return SignedUrl(
            url=f"/v1/artifacts/{key}?expires={expires}&sig={sig}", expires_at=expires
        )

    def verify(self, key: str, expires: int, sig: str) -> bool:
        if expires < time.time():
            return False
        # Constant-time: a timing oracle on the signature would let it be forged
        # byte by byte.
        return hmac.compare_digest(self._mac(key, expires), sig)
