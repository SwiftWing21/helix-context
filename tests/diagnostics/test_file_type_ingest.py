"""File-type ingest diagnostic — sniff the failure modes Helix has around
binary / rich / structured file types.

Known from code audit (2026-04-19):

    - ``/ingest`` accepts only ``content: str``; no file or bytes path.
    - ``CodonChunker`` only knows ``text`` / ``code`` / ``conversation``.
      Everything else becomes ``text`` chunking.
    - ``provenance.EXT_TO_KIND`` covers code / config / doc / log / db
      (inc. .csv, .tsv, .parquet, .sqlite). **No** image, video, audio,
      pdf, html, docx, xlsx entries — all fall through to ``"doc"``.

This script feeds one representative of each broken category to the
live Helix instance and records what happened. Output is a single table
you can triage from, plus a JSON artifact for the bug report.

Per-type expectations (BEFORE running):

    | type           | expected failure mode                              |
    |----------------|---------------------------------------------------|
    | .png / .jpg    | binary bytes → UTF-8 decode at HTTP boundary     |
    | .mp4 / .webm   | same                                              |
    | .mp3 / .wav    | same                                              |
    | .pdf           | same (binary header)                              |
    | .csv           | accepts but text-chunks rows (no column awareness)|
    | .html (bare)   | accepts as doc; tags tokenized                    |
    | .html (media)  | <img src>/<video src> just more text; no asset extract |
    | .xlsx / .docx  | binary zip → UTF-8 decode                        |
    | .svg (xml)     | accepts as doc; useful? markup noise              |

Usage:
    python tests/diagnostics/test_file_type_ingest.py
"""

from __future__ import annotations

import base64
import io
import json
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402


HELIX_URL = "http://127.0.0.1:11437"


# ── Fixture generators (tiny, self-contained, no network/disk deps) ─


def fixture_png_bytes() -> bytes:
    """A 1x1 red PNG (67 bytes). Valid, decodable by any image library."""
    # PNG signature + IHDR + IDAT + IEND for a 1x1 red pixel
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    ihdr_chunk = b"IHDR" + ihdr
    ihdr_full = struct.pack(">I", len(ihdr)) + ihdr_chunk + struct.pack(
        ">I", zlib.crc32(ihdr_chunk)
    )
    raw = b"\x00\xff\x00\x00"  # scanline filter byte + RGB
    idat_data = zlib.compress(raw)
    idat_chunk = b"IDAT" + idat_data
    idat_full = struct.pack(">I", len(idat_data)) + idat_chunk + struct.pack(
        ">I", zlib.crc32(idat_chunk)
    )
    iend_chunk = b"IEND"
    iend_full = struct.pack(">I", 0) + iend_chunk + struct.pack(">I", zlib.crc32(iend_chunk))
    return sig + ihdr_full + idat_full + iend_full


def fixture_jpg_bytes() -> bytes:
    """Minimal JPEG SOI + APP0 + EOI — not a real image, but valid magic number."""
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"
    )


def fixture_mp4_bytes() -> bytes:
    """Minimal MP4 ftyp box. Enough header to be recognized as MP4."""
    ftyp = b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    return ftyp


def fixture_wav_bytes() -> bytes:
    """44-byte RIFF WAV header, 0 samples. Valid WAV."""
    return (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )


def fixture_pdf_bytes() -> bytes:
    """Minimal PDF — header + EOF marker. Most PDF libs accept it as empty."""
    return (
        b"%PDF-1.4\n1 0 obj\n<< >>\nendobj\n"
        b"xref\n0 1\n0000000000 65535 f \n"
        b"trailer\n<< /Size 1 >>\nstartxref\n0\n%%EOF\n"
    )


def fixture_csv_text() -> str:
    """Realistic multi-row CSV — 5 columns, 12 rows, mixed types."""
    lines = ["id,name,email,role,active"]
    for i in range(12):
        lines.append(f"{i},user_{i},user_{i}@example.com,{'admin' if i%3==0 else 'user'},{i%2==0}")
    return "\n".join(lines)


def fixture_html_bare() -> str:
    return (
        "<!DOCTYPE html><html><head><title>Test Doc</title></head>"
        "<body><h1>Header</h1><p>First paragraph with <b>bold</b>.</p>"
        "<ul><li>one</li><li>two</li></ul>"
        "<p>Final paragraph.</p></body></html>"
    )


def fixture_html_with_media() -> str:
    return (
        "<!DOCTYPE html><html><body>"
        "<h2>Product page</h2>"
        "<img src=\"https://cdn.example.com/hero.png\" alt=\"Hero image\">"
        "<video src=\"https://cdn.example.com/demo.mp4\" controls></video>"
        "<picture><source srcset=\"lg.webp\" type=\"image/webp\">"
        "<img src=\"lg.jpg\" alt=\"Large display\"></picture>"
        "<iframe src=\"https://youtube.com/embed/abc\"></iframe>"
        "<p>Buy now for $99</p></body></html>"
    )


def fixture_svg_text() -> str:
    return (
        "<?xml version=\"1.0\"?>"
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\">"
        "<circle cx=\"50\" cy=\"50\" r=\"40\" fill=\"red\"/>"
        "<text x=\"50\" y=\"55\" text-anchor=\"middle\">HELLO</text>"
        "</svg>"
    )


def fixture_xlsx_bytes() -> bytes:
    """First 8 bytes of a real XLSX (PK zip signature + content type marker).

    Not a valid XLSX, but enough to test the "zip signature hits UTF-8 decode"
    failure mode without shipping a real xlsx fixture.
    """
    return b"PK\x03\x04\x14\x00\x06\x00"


# ── Test cases ─────────────────────────────────────────────────────


@dataclass
class IngestCase:
    name: str
    fixture: Callable[[], Any]
    content_type: str  # what we pass to /ingest
    source_ext: str    # what's in metadata.source_id
    sending: str       # how we serialize for transport: "raw_text", "utf8_ignore", "base64"
    content_probe: str = ""  # query token(s) that SHOULD appear in stored content
    note: str = ""


@dataclass
class CaseResult:
    name: str
    sending: str
    status: int = 0
    error: str = ""
    gene_count: int = 0
    retrievable: Optional[bool] = None
    retrieval_top_sources: list[str] = field(default_factory=list)
    payload_bytes: int = 0
    stored_bytes: Optional[int] = None     # length of gene.content in DB
    content_integrity: Optional[bool] = None  # all probe tokens present in stored content
    source_kind: str = ""
    notes: str = ""


CASES = [
    IngestCase("png_1x1",       fixture_png_bytes, "text", ".png", "utf8_ignore",
               content_probe="PNG IHDR IDAT",
               note="binary image, lossy utf-8 decode"),
    IngestCase("png_base64",    fixture_png_bytes, "text", ".png", "base64",
               content_probe="iVBORw0KGgo AAAANSUhEUgAAAAE",
               note="binary image wrapped in base64"),
    IngestCase("jpg_min",       fixture_jpg_bytes, "text", ".jpg", "utf8_ignore",
               content_probe="JFIF"),
    IngestCase("mp4_ftyp",      fixture_mp4_bytes, "text", ".mp4", "utf8_ignore",
               content_probe="ftyp isom mp41"),
    IngestCase("wav_empty",     fixture_wav_bytes, "text", ".wav", "utf8_ignore",
               content_probe="RIFF WAVE fmt"),
    IngestCase("pdf_minimal",   fixture_pdf_bytes, "text", ".pdf", "utf8_ignore",
               content_probe="PDF obj endobj xref"),
    IngestCase("csv_12rows",    fixture_csv_text,  "text", ".csv", "raw_text",
               content_probe="user_0 example admin",
               note="csv should produce row-aware claims but gets text-chunked"),
    IngestCase("html_bare",     fixture_html_bare, "text", ".html", "raw_text",
               content_probe="First paragraph bold Header",
               note="html → doc kind, tags tokenized"),
    IngestCase("html_media",    fixture_html_with_media, "text", ".html", "raw_text",
               content_probe="Product page hero Buy",
               note="html with img/video/iframe — media src dropped to noise"),
    IngestCase("svg_circle",    fixture_svg_text,  "text", ".svg", "raw_text",
               content_probe="circle viewBox HELLO",
               note="svg xml → doc kind, markup tokens"),
    IngestCase("xlsx_min_zip",  fixture_xlsx_bytes, "text", ".xlsx", "utf8_ignore",
               content_probe="PK zip xlsx",
               note="zip signature immediately truncates at NULL"),
]


# ── Runner ─────────────────────────────────────────────────────────


def _encode(data: Any, mode: str) -> str:
    if isinstance(data, str):
        return data if mode == "raw_text" else data
    # bytes
    if mode == "utf8_ignore":
        return data.decode("utf-8", errors="replace")
    if mode == "base64":
        return base64.b64encode(data).decode("ascii")
    return data.decode("utf-8", errors="replace")


def run_case(client: httpx.Client, case: IngestCase) -> CaseResult:
    res = CaseResult(name=case.name, sending=case.sending)
    raw = case.fixture()
    try:
        content = _encode(raw, case.sending)
    except Exception as e:
        res.error = f"encode failed: {e}"
        return res
    res.payload_bytes = len(content.encode("utf-8", errors="replace"))

    source_id = f"F:/Projects/helix-context/tests/diagnostics/fixtures/{case.name}{case.source_ext}"
    # NOTE: Helix only honors metadata["path"] (not metadata["source_id"]).
    # See context_manager.py:389 — metadata.source_id is silently dropped.
    # We populate BOTH here so when the bug is fixed the test doesn't silently
    # swap meaning; the case.name being unique in both keys makes that visible.
    payload = {
        "content": content,
        "content_type": case.content_type,
        "metadata": {"path": source_id, "source_id": source_id},
    }
    try:
        r = client.post(f"{HELIX_URL}/ingest", json=payload, timeout=90)
        res.status = r.status_code
        if r.status_code == 200:
            data = r.json()
            res.gene_count = data.get("count", 0) or len(data.get("gene_ids", []) or [])
            res.notes = (data.get("note") or "")[:120]
        else:
            res.error = r.text[:200]
            return res
    except Exception as e:
        res.error = f"ingest exception: {type(e).__name__}: {e}"
        return res

    # Probe retrievability — two axes:
    #   1. content_probe: tokens expected IN the stored content. If these
    #      don't match, it's a content-integrity failure (e.g. binary NULL
    #      truncation) even if the gene exists.
    #   2. path probe: the fixture name in the source_id. Matches if the
    #      gene exists with proper metadata.path regardless of content.
    try:
        rr = client.post(
            f"{HELIX_URL}/context/packet",
            json={"query": case.content_probe or case.name,
                  "task_type": "explain", "top_k": 10},
            timeout=30,
        ).json()
        sources: list[str] = []
        for bucket in ("verified", "stale_risk", "contradictions"):
            for it in rr.get(bucket) or []:
                sid = it.get("source_id") or ""
                if sid and sid not in sources:
                    sources.append(sid)
        res.retrieval_top_sources = sources[:3]
        res.retrievable = any(case.name in s for s in sources)
    except Exception as e:
        res.notes = (res.notes + f" retrieve_err={e}")[:160]

    # Direct SQLite probe — did the stored gene preserve the payload?
    try:
        import sqlite3
        db = sqlite3.connect(
            "F:/Projects/helix-context/genomes/main/genome.db",
            timeout=5,
        )
        row = db.execute(
            "SELECT source_kind, length(content), content "
            "FROM genes WHERE source_id = ? ORDER BY observed_at DESC LIMIT 1",
            (source_id,),
        ).fetchone()
        db.close()
        if row:
            res.source_kind = row[0] or ""
            res.stored_bytes = row[1]
            stored_content = (row[2] or "").lower()
            if case.content_probe:
                # All-tokens present? Loose whitespace split, per-token substring.
                tokens = [t.lower() for t in case.content_probe.split() if t]
                res.content_integrity = all(t in stored_content for t in tokens)
    except Exception as e:
        res.notes = (res.notes + f" sqlite_err={e}")[:160]
    return res


def main():
    print(f"File-type ingest diagnostic - {HELIX_URL}")
    client = httpx.Client(timeout=120)
    try:
        stats = client.get(f"{HELIX_URL}/stats", timeout=5).json()
        print(f"Genome start: {stats.get('total_genes')} genes")
    except Exception as e:
        print(f"Cannot reach Helix: {e}")
        return 1

    results: list[CaseResult] = []
    for case in CASES:
        print(f"  {case.name:18s} ({case.sending}) ...", end="", flush=True)
        r = run_case(client, case)
        results.append(r)
        status = "OK" if r.status == 200 else f"FAIL {r.status}"
        retr = "retr=Y" if r.retrievable else ("retr=N" if r.retrievable is False else "retr=?")
        print(f" {status:10s} genes={r.gene_count:3d} {retr} bytes={r.payload_bytes}")

    print()
    print("-- Summary -----------------------------------------------------------")
    print(f"{'case':20s}{'send':12s}{'in_B':6s}{'stored_B':9s}"
          f"{'kind':7s}{'retr':6s}{'content_ok':12s}notes")
    for r in results:
        retr = "Y" if r.retrievable else ("N" if r.retrievable is False else "?")
        ci = "Y" if r.content_integrity else ("N" if r.content_integrity is False else "?")
        stored = f"{r.stored_bytes}" if r.stored_bytes is not None else "?"
        kind = r.source_kind or "?"
        notes = (r.error or r.notes or "")[:45]
        print(f"{r.name:20s}{r.sending:12s}{r.payload_bytes:<6d}{stored:<9s}"
              f"{kind:7s}{retr:6s}{ci:12s}{notes}")

    try:
        stats_end = client.get(f"{HELIX_URL}/stats", timeout=5).json()
        delta = (stats_end.get("total_genes") or 0) - (stats.get("total_genes") or 0)
        print(f"\nGenome delta: +{delta} genes from {len(CASES)} fixtures")
    except Exception:
        pass

    out_path = REPO_ROOT / "tests" / "diagnostics" / f"file_type_ingest_{time.strftime('%Y-%m-%d')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "helix_url": HELIX_URL,
        "genome_before": stats.get("total_genes"),
        "results": [
            {
                "name": r.name,
                "sending": r.sending,
                "status": r.status,
                "error": r.error,
                "gene_count": r.gene_count,
                "retrievable": r.retrievable,
                "retrieval_top_sources": r.retrieval_top_sources,
                "payload_bytes": r.payload_bytes,
                "notes": r.notes,
            } for r in results
        ],
    }, indent=2))
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
