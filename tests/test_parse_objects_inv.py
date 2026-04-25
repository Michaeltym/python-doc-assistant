"""Tests for python_doc_assistant.ingest.parse_objects_inv.

Hermetic — every test builds an in-memory objects.inv via sphobjinv. No network,
no dependence on real downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sphobjinv

from python_doc_assistant.ingest.parse_objects_inv import (
    ObjectsInvError,
    SymbolEntry,
    _extract_module,
    _load_inventory,
    _resolve_uri,
    _to_symbol_entry,
    parse_objects_inv,
)

# ------------------------------------------------------------------
# Fixture builder
# ------------------------------------------------------------------


def _make_test_inventory(
    base_dir: Path,
    entries: list[tuple[str, str, str, str]] | None = None,
    *,
    project: str = "Python",
    version: str = "3.12",
) -> Path:
    """Build a docs.python.org-style objects.inv at base_dir/objects.inv.

    Each entry tuple: (name, domain, role, uri).
    """
    if entries is None:
        entries = [
            ("pathlib.Path.read_text", "py", "method", "library/pathlib.html#$"),
            ("pathlib.Path", "py", "class", "library/pathlib.html#$"),
            ("os.path.join", "py", "function", "library/os.path.html#$"),
            ("int", "py", "class", "library/functions.html#$"),
            ("print", "py", "function", "library/functions.html#$"),
            ("genindex", "std", "label", "genindex.html"),
            ("the-with-statement", "std", "label", "reference/compound_stmts.html"),
        ]
    inv = sphobjinv.Inventory()
    inv.project = project
    inv.version = version
    for name, domain, role, uri in entries:
        inv.objects.append(
            sphobjinv.DataObjStr(
                name=name,
                domain=domain,
                role=role,
                priority="1",
                uri=uri,
                dispname="-",
            )
        )
    path = base_dir / "objects.inv"
    path.write_bytes(sphobjinv.compress(inv.data_file()))
    return path


@pytest.fixture
def fake_objects_inv(tmp_path: Path) -> Path:
    """Default fake inventory with 5 py: entries + 2 std: entries."""
    return _make_test_inventory(tmp_path)


# ------------------------------------------------------------------
# _extract_module
# ------------------------------------------------------------------


def test_extract_module_dotted_returns_first_segment() -> None:
    assert _extract_module("pathlib.Path.read_text") == "pathlib"
    assert _extract_module("os.path.join") == "os"


def test_extract_module_single_segment_returns_none() -> None:
    assert _extract_module("int") is None
    assert _extract_module("print") is None


def test_extract_module_empty_returns_none() -> None:
    assert _extract_module("") is None


# ------------------------------------------------------------------
# _resolve_uri
# ------------------------------------------------------------------


def test_resolve_uri_expands_dollar() -> None:
    assert (
        _resolve_uri("library/pathlib.html#$", "pathlib.Path.read_text")
        == "library/pathlib.html#pathlib.Path.read_text"
    )


def test_resolve_uri_no_dollar_unchanged() -> None:
    assert _resolve_uri("library/foo.html", "anything") == "library/foo.html"


def test_resolve_uri_dollar_at_start() -> None:
    assert _resolve_uri("$", "pathlib.Path") == "pathlib.Path"


# ------------------------------------------------------------------
# _to_symbol_entry
# ------------------------------------------------------------------


def test_to_symbol_entry_dotted_name() -> None:
    obj = sphobjinv.DataObjStr(
        name="pathlib.Path.read_text",
        domain="py",
        role="method",
        priority="1",
        uri="library/pathlib.html#$",
        dispname="-",
    )
    entry = _to_symbol_entry(obj)
    assert entry == SymbolEntry(
        name="pathlib.Path.read_text",
        role="py:method",
        uri="library/pathlib.html#pathlib.Path.read_text",
        module="pathlib",
    )


def test_to_symbol_entry_unqualified_name_module_none() -> None:
    obj = sphobjinv.DataObjStr(
        name="int",
        domain="py",
        role="class",
        priority="1",
        uri="library/functions.html#$",
        dispname="-",
    )
    entry = _to_symbol_entry(obj)
    assert entry.role == "py:class"
    assert entry.module is None
    assert entry.uri == "library/functions.html#int"


# ------------------------------------------------------------------
# _load_inventory
# ------------------------------------------------------------------


def test_load_inventory_round_trips(fake_objects_inv: Path) -> None:
    inv = _load_inventory(fake_objects_inv)
    assert inv.project == "Python"
    assert inv.version == "3.12"
    assert len(inv.objects) == 7  # 5 py + 2 std


def test_load_inventory_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ObjectsInvError):
        _load_inventory(tmp_path / "nonexistent.inv")


def test_load_inventory_corrupt_raises(tmp_path: Path) -> None:
    bad = tmp_path / "objects.inv"
    bad.write_bytes(b"not a real inventory")
    with pytest.raises(ObjectsInvError):
        _load_inventory(bad)


# ------------------------------------------------------------------
# parse_objects_inv
# ------------------------------------------------------------------


def test_parse_objects_inv_filters_to_py_domain(fake_objects_inv: Path, tmp_path: Path) -> None:
    entries = parse_objects_inv(tmp_path)
    assert len(entries) == 5
    assert all(e.role.startswith("py:") for e in entries)


def test_parse_objects_inv_includes_pathlib_path_read_text(
    fake_objects_inv: Path, tmp_path: Path
) -> None:
    entries = parse_objects_inv(tmp_path)
    target = next(e for e in entries if e.name == "pathlib.Path.read_text")
    assert target.role == "py:method"
    assert target.module == "pathlib"
    assert target.uri == "library/pathlib.html#pathlib.Path.read_text"


def test_parse_objects_inv_preserves_module_field(fake_objects_inv: Path, tmp_path: Path) -> None:
    entries = parse_objects_inv(tmp_path)
    by_name = {e.name: e for e in entries}
    assert by_name["pathlib.Path"].module == "pathlib"
    assert by_name["os.path.join"].module == "os"
    assert by_name["int"].module is None
    assert by_name["print"].module is None


def test_parse_objects_inv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ObjectsInvError):
        parse_objects_inv(tmp_path)


def test_parse_objects_inv_corrupt_raises(tmp_path: Path) -> None:
    (tmp_path / "objects.inv").write_bytes(b"junk bytes")
    with pytest.raises(ObjectsInvError):
        parse_objects_inv(tmp_path)
