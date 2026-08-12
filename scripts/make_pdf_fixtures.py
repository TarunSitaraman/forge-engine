#!/usr/bin/env python3
"""Generate the small PDF fixtures used by the test suite.

Writes raw PDF syntax rather than depending on a PDF-authoring library. The
fixtures are committed, so this script exists to document exactly how they were
made and to allow regeneration — the tests do not run it.

Deliberately minimal and deterministic: byte-identical output on every run, so
a fixture's content hash is stable and can be asserted.

    python3 scripts/make_pdf_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "pdf"

#: 1x1 black PNG-ish raw image data (uncompressed grayscale), enough to make a
#: page that renders something but contains no extractable text.
_IMAGE_BYTES = bytes([0x00])


def _pdf(objects: list[bytes], root: int = 1) -> bytes:
    """Assemble numbered objects into a valid PDF with a correct xref table."""
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    n = len(objects) + 1
    out += f"xref\n0 {n}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {n} /Root {root} 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def _content_stream(ops: str) -> bytes:
    data = ops.encode("latin-1")
    return b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream"


def _text_ops(lines: list[tuple[str, int]], start_y: int = 740) -> str:
    """Build a content stream placing (text, font_size) lines down the page.

    Font size is what makes deterministic heading detection possible: headings
    are set larger than body text, exactly as in a real document.
    """
    ops = ["BT"]
    y = start_y
    for text, size in lines:
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"/F1 {size} Tf")
        ops.append(f"1 0 0 1 72 {y} Tm")
        ops.append(f"({escaped}) Tj")
        y -= int(size * 1.6) + 4
    ops.append("ET")
    return "\n".join(ops)


def _simple_doc(pages: list[list[tuple[str, int]]], title: str | None = None) -> bytes:
    """Build a PDF with one content stream per page and a shared Helvetica font."""
    n_pages = len(pages)
    font_obj = 3 + 2 * n_pages
    info_obj = font_obj + 1 if title else None

    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode(),
    ]
    for i, lines in enumerate(pages):
        content_obj = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_obj} 0 R "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> >>".encode()
        )
        objects.append(_content_stream(_text_ops(lines)))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    if title:
        objects.append(
            f"<< /Title ({title}) /Author (Forge Test Suite) "
            f"/Creator (make_pdf_fixtures.py) >>".encode()
        )

    body = _pdf(objects)
    if info_obj:
        body = body.replace(
            b"/Root 1 0 R >>", f"/Root 1 0 R /Info {info_obj} 0 R >>".encode()
        )
    return body


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Normal single-page text PDF with a heading and body text.
    (OUT / "simple.pdf").write_bytes(
        _simple_doc(
            [
                [
                    ("Attention Mechanisms", 20),
                    ("Self-attention lets a model weigh all positions at once.", 11),
                    ("It replaces recurrence with direct pairwise interaction.", 11),
                ]
            ],
            title="Attention Mechanisms",
        )
    )

    # 2. Multi-page document with headings on separate pages.
    (OUT / "multipage.pdf").write_bytes(
        _simple_doc(
            [
                [
                    ("Retrieval Augmented Generation", 20),
                    ("RAG grounds generation in retrieved passages.", 11),
                    ("This reduces hallucination on open-domain questions.", 11),
                ],
                [
                    ("Chunking Strategy", 18),
                    ("Chunk size materially affects retrieval quality.", 11),
                    ("Small chunks lose context; large chunks dilute embeddings.", 11),
                ],
                [
                    ("Hybrid Search", 18),
                    ("Combining BM25 with dense vectors outperforms either alone.", 11),
                ],
            ],
            title="Retrieval Augmented Generation",
        )
    )

    # 3. Document where one page has no text at all.
    (OUT / "empty-page.pdf").write_bytes(
        _simple_doc(
            [
                [("Vector Databases", 20), ("Indexes embeddings for similarity search.", 11)],
                [],  # deliberately blank
                [("Graph Traversal", 18), ("Visits every reachable vertex once.", 11)],
            ]
        )
    )

    # 4. Overlapping document — shares concepts with multipage.pdf, for testing
    #    concept candidate matching across sources.
    (OUT / "overlapping.pdf").write_bytes(
        _simple_doc(
            [
                [
                    ("Retrieval Augmented Generation Revisited", 20),
                    ("RAG systems depend heavily on the retrieval step.", 11),
                    ("Hybrid Search improves recall over dense-only retrieval.", 11),
                ],
                [
                    ("Heap", 18),
                    ("A heap maintains the smallest element at its root.", 11),
                ],
            ],
            title="RAG Revisited",
        )
    )

    # 5. Image-only page: valid PDF, renders content, contains zero text.
    #    Must be reported as OCR_REQUIRED rather than as a successful empty
    #    extraction.
    img = (
        b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length 1 >>\n"
        b"stream\n" + _IMAGE_BYTES + b"\nendstream"
    )
    (OUT / "image-only.pdf").write_bytes(
        _pdf(
            [
                b"<< /Type /Catalog /Pages 2 0 R >>",
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
                b"/Resources << /XObject << /Im1 5 0 R >> >> >>",
                _content_stream("q 200 0 0 200 100 500 cm /Im1 Do Q"),
                img,
            ]
        )
    )

    # 6. Malformed: truncated mid-object, no xref, no EOF marker.
    (OUT / "malformed.pdf").write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Cou"
    )

    # 7. Not a PDF at all, despite the extension.
    (OUT / "not-a-pdf.pdf").write_bytes(b"This is plain text pretending to be a PDF.\n")

    for p in sorted(OUT.iterdir()):
        print(f"  {p.stat().st_size:>7} bytes  {p.name}")


if __name__ == "__main__":
    main()
