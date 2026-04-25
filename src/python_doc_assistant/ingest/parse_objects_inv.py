"""Parse Sphinx objects.inv into typed SymbolEntry records.

See plans/v0-retrieval-eval.md §2 for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import sphobjinv

from python_doc_assistant.ingest.fetch_docs import IngestError

# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class ObjectsInvError(IngestError):
    """objects.inv is missing or cannot be parsed."""


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolEntry:
    """One Python symbol entry from objects.inv (filtered to py: domain)."""

    name: str  # e.g. "pathlib.Path.read_text"
    role: str  # e.g. "py:method"  (domain:role joined)
    uri: str  # e.g. "library/pathlib.html#pathlib.Path.read_text"
    module: str | None  # e.g. "pathlib"; None for unqualified names


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def parse_objects_inv(docs_dir: Path) -> list[SymbolEntry]:
    """Parse `docs_dir/objects.inv` into SymbolEntry records.

    Returns py: domain entries only; std:doc / std:label / etc. are filtered out.

    Raises ObjectsInvError on missing file or parse failure.

    Suggested flow:
        1. inv = _load_inventory(docs_dir / "objects.inv")
        2. for obj in inv.objects:
               if obj.domain != "py": continue
               entries.append(_to_symbol_entry(obj))
        3. return entries
    """
    inv = _load_inventory(docs_dir / "objects.inv")
    entries: list[SymbolEntry] = []
    for obj in inv.objects:
        if obj.domain != "py":
            continue
        entries.append(_to_symbol_entry(obj))
    return entries


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _load_inventory(path: Path) -> sphobjinv.Inventory:
    """Load and decompress an objects.inv file.

    Raises ObjectsInvError if the file is missing or sphobjinv cannot parse it.
    """
    if not path.exists():
        raise ObjectsInvError(f"objects.inv not found at {path}")
    if not path.is_file():
        raise ObjectsInvError(f"{path} exists but is not a regular file")
    try:
        return sphobjinv.Inventory(str(path))  # pyright: ignore[reportCallIssue]
    except Exception as e:
        raise ObjectsInvError(f"Failed to read {path}") from e


def _to_symbol_entry(obj: sphobjinv.DataObjStr) -> SymbolEntry:
    """Convert one sphobjinv object to a SymbolEntry.

    Caller is responsible for filtering by domain; this helper does not check.
    """
    return SymbolEntry(
        name=cast(str, obj.name),
        role=f"{obj.domain}:{obj.role}",
        uri=_resolve_uri(cast(str, obj.uri), cast(str, obj.name)),
        module=_extract_module(cast(str, obj.name)),
    )


def _extract_module(name: str) -> str | None:
    """First dotted segment, or None for a single-segment name.

    Examples:
        "pathlib.Path.read_text" -> "pathlib"
        "os.path.join"           -> "os"
        "int"                    -> None
        ""                       -> None
    """
    segments = name.split(".")
    return segments[0] if len(segments) > 1 else None


def _resolve_uri(raw_uri: str, name: str) -> str:
    """Expand sphobjinv's '$' shorthand: '$' in raw_uri is replaced by `name`.

    Examples:
        ("library/pathlib.html#$", "pathlib.Path.read_text")
            -> "library/pathlib.html#pathlib.Path.read_text"
        ("library/foo.html", "anything")
            -> "library/foo.html"
    """
    return raw_uri.replace("$", name)
