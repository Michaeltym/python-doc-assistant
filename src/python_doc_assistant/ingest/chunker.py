"""Build symbol_chunk and section_chunk records from HTML + objects.inv.

See plans/v0-retrieval-eval.md §3 for the full contract.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bs4 import BeautifulSoup

from python_doc_assistant.ingest.fetch_docs import IngestError
from python_doc_assistant.ingest.parse_objects_inv import SymbolEntry

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CHUNK_TYPE_SYMBOL: Final[str] = "symbol"
CHUNK_TYPE_SECTION: Final[str] = "section"

# Max characters per section_chunk (decision per review §1.4 + plan decision points).
SECTION_CHUNK_MAX_CHARS: Final[int] = 1500


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class ChunkerError(IngestError):
    """HTML parsing or chunking failed."""


# ------------------------------------------------------------------
# Schema (v0 fixed; downstream eval/debug/citation/rerank depend on it)
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """Unified chunk schema (PLAN.md §8)."""

    chunk_id: str  # "symbol:pathlib.Path.read_text" | "section:library/pathlib#examples"
    chunk_type: str  # "symbol" | "section"
    docs_version: str  # "3.12"
    title: str  # "Path.read_text" | "Examples"
    text: str  # body text
    symbols: tuple[str, ...]  # ("pathlib.Path.read_text",); section_chunks may be empty ()
    canonical_url: str  # "library/pathlib.html#pathlib.Path.read_text"
    anchor: str | None  # "pathlib.Path.read_text"; None for sections without explicit id
    parent_module: str | None  # "pathlib"; None for unqualified or section_chunk
    source_path: str  # "library/pathlib.html"
    source_hash: str  # "sha256:abc123..." (full hex)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_chunks(
    docs_dir: Path,
    docs_version: str,
    symbols: list[SymbolEntry],
) -> list[Chunk]:
    """Build symbol_chunks + section_chunks for HTML files referenced by `symbols`.

    Two-pass strategy (per plan §3):
        1. Symbol pass: for each SymbolEntry, locate the matching <dl class="py ..."> by
           anchor in the corresponding HTML; emit one symbol_chunk per entry. Track which
           HTML nodes were absorbed so the section pass can skip them.
        2. Section pass: walk each HTML file's <section> elements; skip absorbed nodes;
           split the remaining text by h2/h3 boundaries; cap each section_chunk at
           SECTION_CHUNK_MAX_CHARS.

    All chunk_ids must be globally unique.

    Suggested flow:
        by_path: dict[str, list[SymbolEntry]] = group symbols by source_path
        chunks: list[Chunk] = []
        for source_path, file_symbols in by_path.items():
            soup, source_hash = _load_html(docs_dir / source_path)
            sym_chunks, absorbed = _symbol_chunks_for_html(
                soup, file_symbols, docs_version, source_path, source_hash
            )
            sec_chunks = _section_chunks_for_html(
                soup, docs_version, source_path, source_hash, absorbed
            )
            chunks.extend(sym_chunks)
            chunks.extend(sec_chunks)
        return chunks

    Raises ChunkerError on HTML parse failure.
    """
    by_path: dict[str, list[SymbolEntry]] = {}
    for symbol in symbols:
        path, _ = _path_and_anchor(symbol.uri)
        by_path.setdefault(path, []).append(symbol)
    chunks: list[Chunk] = []
    for path, symbols in by_path.items():
        doc_path = docs_dir / path
        soup, full_sha = _load_html(doc_path)
        symbol_chunks, absorbed_node_ids = _symbol_chunks_for_html(
            soup, by_path[path], docs_version, path, full_sha
        )
        chunks.extend(symbol_chunks)
        section_chunks = _section_chunks_for_html(
            soup, docs_version, path, full_sha, absorbed_node_ids
        )
        chunks.extend(section_chunks)
    return chunks


# ------------------------------------------------------------------
# Helpers — utilities (deterministic, easy to unit-test in isolation)
# ------------------------------------------------------------------


def _path_and_anchor(uri: str) -> tuple[str, str | None]:
    """Split a URI on the first '#' into (path, anchor).

    Examples:
        "library/pathlib.html#pathlib.Path.read_text"
            -> ("library/pathlib.html", "pathlib.Path.read_text")
        "library/pathlib.html"
            -> ("library/pathlib.html", None)
        "library/pathlib.html#"
            -> ("library/pathlib.html", "")
    """
    if "#" in uri:
        path, anchor = uri.split("#", 1)
        return path, anchor
    return uri, None


def _compute_source_hash(content: bytes) -> str:
    """Return 'sha256:<hex>' digest for `content` (used in Chunk.source_hash)."""
    h = hashlib.sha256()
    h.update(content)
    return f"sha256:{h.hexdigest()}"


def _make_chunk_id(chunk_type: str, key: str) -> str:
    """Format '<chunk_type>:<key>'."""
    return f"{chunk_type}:{key}"


def _slug(text: str) -> str:
    """Lowercase and dash-normalize text for stable section-chunk keys.

    Examples:
        "Examples"             -> "examples"
        "Path Methods"         -> "path-methods"
        "os.path & comparison" -> "os-path-comparison"
        "hello---world"        -> "hello-world"
    """
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
    return s.strip("-")


# ------------------------------------------------------------------
# Helpers — HTML processing (heavier; mostly exercised via build_chunks tests)
# ------------------------------------------------------------------


def _load_html(path: Path) -> tuple[BeautifulSoup, str]:
    """Read `path`, parse with bs4 (lxml backend), return (soup, source_hash).

    Raises ChunkerError on read or parse failure.
    """
    if not path.exists():
        raise ChunkerError(f"{path} is not found")
    if not path.is_file():
        raise ChunkerError(f"{path} is not a file")
    with path.open(encoding="utf-8") as f:
        html = f.read()
        soup = BeautifulSoup(html, "lxml")
        return soup, _compute_source_hash(path.read_bytes())


def _symbol_chunks_for_html(
    soup: BeautifulSoup,
    symbols: list[SymbolEntry],
    docs_version: str,
    source_path: str,
    source_hash: str,
) -> tuple[list[Chunk], set[int]]:
    """Build symbol_chunks for `symbols` whose anchor lives in `soup`.

    Locate the <dl class="py ..."> block whose <dt id> matches each symbol's anchor.
    Extract signature (<dt>) + docstring (<dd>) into Chunk.text.

    Returns (chunks, absorbed_node_ids) where each id() refers to an absorbed <dl>
    Tag in `soup`. The section pass uses these to skip duplicates.
    """
    chunks: list[Chunk] = []
    absorbed_node_ids: set[int] = set()
    for symbol in symbols:
        parts = symbol.name.rsplit(".", 1)
        title = parts[-1] if parts else symbol.name
        _, anchor = _path_and_anchor(symbol.uri)
        dt = soup.find("dt", id=symbol.name)
        dt_text = ""
        body_text = ""
        if dt:
            for link in dt.find_all("a", class_="headerlink"):
                link.decompose()
            dt_text = dt.get_text(separator=" ", strip=True)
            dl = dt.find_parent("dl", class_="py")
            if dl:
                absorbed_node_ids.add(id(dl))
                body = dl.find("dd")
                if body:
                    for link in body.find_all("a", class_="headerlink"):
                        link.decompose()
                    body_text = body.get_text(separator=" ", strip=True)
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id("symbol", symbol.name),
                chunk_type="symbol",
                docs_version=docs_version,
                title=title,
                text=f"{dt_text}\n\n{body_text}",
                symbols=(symbol.name,),
                canonical_url=symbol.uri,
                anchor=anchor,
                parent_module=symbol.module,
                source_path=source_path,
                source_hash=source_hash,
            )
        )

    return chunks, absorbed_node_ids


def _section_chunks_for_html(
    soup: BeautifulSoup,
    docs_version: str,
    source_path: str,
    source_hash: str,
    absorbed_node_ids: set[int],
    *,
    max_chars: int = SECTION_CHUNK_MAX_CHARS,
) -> list[Chunk]:
    """Build section_chunks from `soup`, one per <section> in DOM order.

    For each <section>:
      1. Take the first <h1>/<h2>/<h3> as `title` (strip Sphinx headerlink ¶).
      2. Clone the section (string round-trip) so decomposes don't mutate the
         original `soup`. From the clone strip:
           - the heading (already captured)
           - any <dl class="py ..."> (its content lives in symbol_chunks)
           - any nested <section> (each is processed in its own iteration)
      3. Take the remaining text. Skip the section if empty.
      4. If text > max_chars, split on paragraph boundaries.
      5. Use the section's HTML id when present, otherwise _slug(title), as the
         stable key for chunk_id and anchor.

    `absorbed_node_ids` is accepted for symmetry with the symbol pass; the
    class-based <dl class="py ..."> strip above already removes the same nodes,
    so we don't need to consult the set in v0.
    """
    del absorbed_node_ids  # see docstring; class-based strip handles dedup
    chunks: list[Chunk] = []
    for section in soup.find_all("section"):
        heading = section.find(["h1", "h2", "h3"])
        if heading is None:
            continue
        for link in heading.find_all("a", class_="headerlink"):
            link.decompose()
        title = heading.get_text(strip=True)
        if not title:
            continue

        working = BeautifulSoup(str(section), "lxml").find("section")
        if working is None:
            continue
        inner_heading = working.find(["h1", "h2", "h3"])
        if inner_heading is not None:
            inner_heading.decompose()
        for dl in working.select("dl.py"):
            dl.decompose()
        for nested in working.find_all("section"):
            nested.decompose()
        text = working.get_text(separator="\n", strip=True)
        if not text:
            continue

        pieces = [text] if len(text) <= max_chars else _split_paragraphs(text, max_chars)

        raw_id = section.get("id")
        section_id = raw_id if isinstance(raw_id, str) else _slug(title)
        canonical_url = f"{source_path}#{section_id}"
        path_no_ext = source_path.removesuffix(".html")
        key_base = f"{path_no_ext}#{section_id}"

        for i, piece in enumerate(pieces):
            suffix = f"-{i}" if len(pieces) > 1 else ""
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id("section", key_base + suffix),
                    chunk_type="section",
                    docs_version=docs_version,
                    title=title,
                    text=piece,
                    symbols=(),
                    canonical_url=canonical_url,
                    anchor=section_id,
                    parent_module=None,
                    source_path=source_path,
                    source_hash=source_hash,
                )
            )

    return chunks


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split `text` on blank-line paragraph boundaries; never exceed `max_chars`."""
    paras = text.split("\n\n")
    out: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 > max_chars and buf:
            out.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out
