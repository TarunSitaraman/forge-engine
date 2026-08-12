"""Identity generation.

Forge uses **deterministic identity** for anything derived from the corpus.
Re-indexing an unchanged vault must produce a byte-identical index, which is
impossible with random ids. So:

* ``Source``   -> derived from vault-relative path (stable across content edits)
* ``Document`` -> derived from (source_id, content_hash) — a new parse of new
  content is a new document; re-parsing identical content is the same document
* ``Span``     -> derived from (document_id, ordinal, locator)
* ``Concept``  -> derived from canonical name

Random, time-ordered ids (:func:`new_id`) are used only for genuinely
append-only records — revisions and run ids — where no natural key exists.
"""

from __future__ import annotations

import hashlib
import os
import time

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def deterministic_id(namespace: str, *parts: str) -> str:
    """Stable 26-char id derived from ``namespace`` and ``parts``.

    Uses BLAKE2b with the namespace as a personalization key so that the same
    parts under different namespaces never collide.
    """
    h = hashlib.blake2b(digest_size=16, person=namespace.encode("utf-8")[:16])
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")  # unambiguous separator: ("a","bc") != ("ab","c")
    return _b32(h.digest())[:26]


def new_id() -> str:
    """Time-ordered, non-deterministic id for append-only records."""
    ts = int(time.time() * 1000).to_bytes(6, "big")
    return _b32(ts + os.urandom(10))[:26]


def content_hash(data: bytes) -> str:
    """Canonical content hash. sha256 hex — the change-detection key."""
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    """Content hash of text, normalized to LF so CRLF checkouts hash equal.

    The corpus was authored on Windows; without newline normalization the same
    logical content would hash differently depending on the checkout, and
    change detection would report spurious modifications.
    """
    return content_hash(text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))


def _b32(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    if n == 0:
        return _ALPHABET[0]
    out: list[str] = []
    while n:
        n, rem = divmod(n, 32)
        out.append(_ALPHABET[rem])
    return "".join(reversed(out))
