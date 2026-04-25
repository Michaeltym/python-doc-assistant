"""Tests for python_doc_assistant.ingest.fetch_docs.

Hermetic by design — no real network, no real downloads. We use:
    - pytest's tmp_path for isolated filesystem state
    - pytest's monkeypatch for stubbing requests.get / time.sleep / module-level helpers
    - in-memory tarballs built per test for fixtures
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import requests

from python_doc_assistant.ingest.fetch_docs import (
    ARCHIVE_URL_TEMPLATE,
    SHA_SHORT_LENGTH,
    ArchiveDownloadError,
    IngestManifest,
    ManifestParseError,
    ShaConflictError,
    _cache_archive,
    _check_sha_conflict,
    _compute_sha256,
    _download_archive,
    _extract_tar_info,
    _parse_docs_served_version,
    _read_manifest,
    _resolve_archive_url,
    _unpack_archive,
    _update_current_txt,
    _utc_now_iso,
    _write_manifest,
    ingest_docs,
)

FETCH_DOCS_MODULE = "python_doc_assistant.ingest.fetch_docs"


# ------------------------------------------------------------------
# Fake response and download helpers
# ------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for `requests.Response` for download tests."""

    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self._content = content
        self.status_code = status_code

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def _make_test_archive(
    base_dir: Path,
    suffix: str,
    *,
    served_version: str = "3.12.13",
    extra_marker: str = "",
) -> Path:
    """Build a docs.python.org-style fake tarball under base_dir.

    `extra_marker` makes the archive bytes differ even when served_version is the same,
    which is handy for sha-conflict tests.
    """
    staging = base_dir / f"staging-{suffix}"
    top = staging / "python-3.12-docs-html"
    top.mkdir(parents=True)
    (top / "index.html").write_text(
        f"<html><head><title>Python {served_version} documentation</title></head></html>",
        encoding="utf-8",
    )
    library = top / "library"
    library.mkdir()
    (library / "pathlib.html").write_text(
        f"<html><body>pathlib stub {suffix} {extra_marker}</body></html>",
        encoding="utf-8",
    )
    archive_path = base_dir / f"archive-{suffix}.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as tar:
        tar.add(top, arcname="python-3.12-docs-html")
    return archive_path


def _stub_download(src: Path) -> Callable[..., Path]:
    """Return a fake `_download_archive` that copies `src` to dest."""

    def fake_dl(url: str, dest: Path, **kwargs: Any) -> Path:
        shutil.copy(src, dest)
        return dest

    return fake_dl


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def fake_archive(tmp_path: Path) -> Path:
    """Default fake docs tarball used by most tests."""
    return _make_test_archive(tmp_path, "default")


@pytest.fixture
def fake_html_root(tmp_path: Path) -> Path:
    """Already-unpacked docs tree with a parseable <title>."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.html").write_text(
        "<html><head><title>Python 3.12.13 documentation</title></head></html>",
        encoding="utf-8",
    )
    return docs


@pytest.fixture
def malicious_tarball(tmp_path: Path) -> Path:
    """Tarball with a `../` traversal entry under the expected top dir."""
    archive_path = tmp_path / "evil.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as tar:
        info = tarfile.TarInfo(name="python-3.12-docs-html/../evil.html")
        data = b"<html>evil</html>"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return archive_path


# ------------------------------------------------------------------
# _resolve_archive_url
# ------------------------------------------------------------------


def test_resolve_archive_url_uses_template() -> None:
    expected = ARCHIVE_URL_TEMPLATE.format(version="3.12")
    assert _resolve_archive_url("3.12") == expected


def test_resolve_archive_url_substitutes_version() -> None:
    assert "3.13" in _resolve_archive_url("3.13")
    assert "3.12" not in _resolve_archive_url("3.13")


# ------------------------------------------------------------------
# _compute_sha256
# ------------------------------------------------------------------


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello\n")
    assert _compute_sha256(f) == hashlib.sha256(b"hello\n").hexdigest()


def test_compute_sha256_streaming_handles_large_file(tmp_path: Path) -> None:
    blob = b"x" * (3 << 20)  # 3 MB > default chunk_size
    f = tmp_path / "big.bin"
    f.write_bytes(blob)
    assert _compute_sha256(f, chunk_size=64 * 1024) == hashlib.sha256(blob).hexdigest()


# ------------------------------------------------------------------
# _download_archive
# ------------------------------------------------------------------


def test_download_archive_writes_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"fake archive bytes"
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResponse(payload))

    dest = tmp_path / "out.tar.bz2"
    result = _download_archive(
        "https://example/archive.tar.bz2",
        dest,
        timeout=30.0,
        retries=3,
        user_agent="t/0",
    )
    assert result == dest
    assert dest.read_bytes() == payload


def test_download_archive_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_count = {"n": 0}
    payload = b"final ok"

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise requests.ConnectionError("transient")
        return FakeResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    dest = tmp_path / "out.tar.bz2"
    _download_archive("https://x/y", dest, timeout=10.0, retries=3, user_agent="t/0")
    assert call_count["n"] == 3
    assert dest.read_bytes() == payload


def test_download_archive_raises_after_max_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def always_fail(*args: object, **kwargs: object) -> FakeResponse:
        raise requests.ConnectionError("dead")

    monkeypatch.setattr(requests, "get", always_fail)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    dest = tmp_path / "out.tar.bz2"
    with pytest.raises(ArchiveDownloadError) as exc_info:
        _download_archive("https://x/y", dest, timeout=10.0, retries=2, user_agent="t/0")
    assert isinstance(exc_info.value.__cause__, requests.ConnectionError)


def test_download_archive_sends_user_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse(b"ok")

    monkeypatch.setattr(requests, "get", fake_get)

    dest = tmp_path / "out.tar.bz2"
    _download_archive("https://x/y", dest, timeout=10.0, retries=1, user_agent="myUA/9.9")
    assert captured["headers"]["User-Agent"] == "myUA/9.9"


# ------------------------------------------------------------------
# _extract_tar_info
# ------------------------------------------------------------------


def test_extract_tar_info_strips_top_dir(fake_archive: Path) -> None:
    with tarfile.open(fake_archive, "r:bz2") as tar:
        names = {m.name for m in _extract_tar_info(tar.getmembers(), "3.12")}
    assert "index.html" in names
    assert "library/pathlib.html" in names
    assert "" not in names
    assert all(not n.startswith("python-3.12-docs-html/") for n in names)


def test_extract_tar_info_skips_unrelated_entries(tmp_path: Path) -> None:
    archive = tmp_path / "mixed.tar.bz2"
    payload = b"x"
    with tarfile.open(archive, "w:bz2") as tar:
        # entry with the right prefix
        good = tarfile.TarInfo("python-3.12-docs-html/index.html")
        good.size = len(payload)
        tar.addfile(good, io.BytesIO(payload))
        # entry with a different prefix — must be skipped
        bad = tarfile.TarInfo("not-our-prefix/foo.html")
        bad.size = len(payload)
        tar.addfile(bad, io.BytesIO(payload))

    with tarfile.open(archive, "r:bz2") as tar:
        names = {m.name for m in _extract_tar_info(tar.getmembers(), "3.12")}
    assert names == {"index.html"}


# ------------------------------------------------------------------
# _unpack_archive
# ------------------------------------------------------------------


def test_unpack_archive_flattens_top_dir(fake_archive: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    _unpack_archive(fake_archive, dest, "3.12")
    assert (dest / "index.html").is_file()
    assert (dest / "library" / "pathlib.html").is_file()
    assert not (dest / "python-3.12-docs-html").exists()


def test_unpack_archive_creates_dest_if_missing(fake_archive: Path, tmp_path: Path) -> None:
    dest = tmp_path / "deep" / "nested" / "out"
    assert not dest.exists()
    _unpack_archive(fake_archive, dest, "3.12")
    assert dest.is_dir()
    assert (dest / "index.html").is_file()


def test_unpack_archive_rejects_path_traversal(malicious_tarball: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    with pytest.raises(tarfile.TarError):
        _unpack_archive(malicious_tarball, dest, "3.12")
    assert not (tmp_path / "evil.html").exists()


# ------------------------------------------------------------------
# _parse_docs_served_version
# ------------------------------------------------------------------


def test_parse_docs_served_version_extracts_patch(fake_html_root: Path) -> None:
    assert _parse_docs_served_version(fake_html_root) == "3.12.13"


def test_parse_docs_served_version_raises_on_missing_title(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.html").write_text(
        "<html><head><title>Welcome</title></head></html>", encoding="utf-8"
    )
    with pytest.raises(ManifestParseError):
        _parse_docs_served_version(docs)


def test_parse_docs_served_version_raises_on_missing_file(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    with pytest.raises(ManifestParseError):
        _parse_docs_served_version(docs)


# ------------------------------------------------------------------
# _check_sha_conflict
# ------------------------------------------------------------------


def test_check_sha_conflict_no_current_returns_false(tmp_path: Path) -> None:
    assert _check_sha_conflict("3.12", "abc123def456", False, tmp_path) is False


def test_check_sha_conflict_same_sha_returns_true(tmp_path: Path) -> None:
    current = tmp_path / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True)
    current.write_text("abc123def456", encoding="utf-8")
    assert _check_sha_conflict("3.12", "abc123def456", False, tmp_path) is True


def test_check_sha_conflict_diff_no_force_raises(tmp_path: Path) -> None:
    current = tmp_path / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True)
    current.write_text("oldoldoldold", encoding="utf-8")
    with pytest.raises(ShaConflictError):
        _check_sha_conflict("3.12", "newnewnewnew", False, tmp_path)


def test_check_sha_conflict_diff_with_force_returns_false(tmp_path: Path) -> None:
    current = tmp_path / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True)
    current.write_text("oldoldoldold", encoding="utf-8")
    assert _check_sha_conflict("3.12", "newnewnewnew", True, tmp_path) is False


def test_check_sha_conflict_strips_trailing_whitespace(tmp_path: Path) -> None:
    current = tmp_path / "docs" / "3.12" / "current.txt"
    current.parent.mkdir(parents=True)
    current.write_text("abc123def456\n", encoding="utf-8")
    assert _check_sha_conflict("3.12", "abc123def456", False, tmp_path) is True


# ------------------------------------------------------------------
# _write_manifest + _read_manifest
# ------------------------------------------------------------------


def _sample_manifest() -> IngestManifest:
    return IngestManifest(
        docs_version="3.12",
        docs_served_version="3.12.13",
        docs_url="https://example/x.tar.bz2",
        docs_archive_sha256="a" * 64,
        ingest_timestamp="2026-04-25T15:00:00Z",
    )


def test_write_manifest_writes_5_fields(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    path = _write_manifest(tmp_path, manifest)
    assert path == tmp_path / "ingest_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {
        "docs_version",
        "docs_served_version",
        "docs_url",
        "docs_archive_sha256",
        "ingest_timestamp",
    }
    assert data["docs_version"] == "3.12"
    assert data["docs_served_version"] == "3.12.13"
    assert data["ingest_timestamp"] == "2026-04-25T15:00:00Z"


def test_read_manifest_round_trips_write(tmp_path: Path) -> None:
    original = _sample_manifest()
    path = _write_manifest(tmp_path, original)
    loaded = _read_manifest(path)
    assert loaded == original


def test_read_manifest_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(ManifestParseError):
        _read_manifest(tmp_path / "missing.json")


# ------------------------------------------------------------------
# _update_current_txt
# ------------------------------------------------------------------


def test_update_current_txt_writes_sha(tmp_path: Path) -> None:
    version_dir = tmp_path / "docs" / "3.12"
    version_dir.mkdir(parents=True)
    _update_current_txt(tmp_path, "3.12", "abc123def456")
    assert (version_dir / "current.txt").read_text(encoding="utf-8") == "abc123def456"


def test_update_current_txt_overwrites_existing(tmp_path: Path) -> None:
    version_dir = tmp_path / "docs" / "3.12"
    version_dir.mkdir(parents=True)
    (version_dir / "current.txt").write_text("oldoldoldold", encoding="utf-8")
    _update_current_txt(tmp_path, "3.12", "newnewnewnew")
    assert (version_dir / "current.txt").read_text(encoding="utf-8") == "newnewnewnew"


# ------------------------------------------------------------------
# _cache_archive
# ------------------------------------------------------------------


def test_cache_archive_copies_when_missing(tmp_path: Path) -> None:
    src = tmp_path / "src.tar.bz2"
    src.write_bytes(b"archive bytes")
    cache_root = tmp_path / "cache"
    full_sha = "a" * 64

    target = _cache_archive(src, cache_root, full_sha)

    expected = cache_root / "archives" / f"{full_sha}.tar.bz2"
    assert target == expected
    assert expected.read_bytes() == b"archive bytes"


def test_cache_archive_skips_when_already_present(tmp_path: Path) -> None:
    src = tmp_path / "src.tar.bz2"
    src.write_bytes(b"new bytes")
    cache_root = tmp_path / "cache"
    full_sha = "b" * 64

    archives = cache_root / "archives"
    archives.mkdir(parents=True)
    pre = archives / f"{full_sha}.tar.bz2"
    pre.write_bytes(b"sentinel")

    _cache_archive(src, cache_root, full_sha)
    assert pre.read_bytes() == b"sentinel"


# ------------------------------------------------------------------
# _utc_now_iso
# ------------------------------------------------------------------


def test_utc_now_iso_format() -> None:
    s = _utc_now_iso()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s) is not None


# ------------------------------------------------------------------
# ingest_docs (end-to-end with helpers; HTTP / time mocked)
# ------------------------------------------------------------------


def test_ingest_docs_first_run_writes_manifest_and_current(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_archive: Path
) -> None:
    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._download_archive", _stub_download(fake_archive))
    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._utc_now_iso", lambda: "2026-04-25T15:00:00Z")

    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"

    result = ingest_docs(version="3.12", data_root=data_root, cache_root=cache_root)

    assert result.skipped is False
    assert result.manifest.docs_version == "3.12"
    assert result.manifest.docs_served_version == "3.12.13"
    assert result.manifest.ingest_timestamp == "2026-04-25T15:00:00Z"
    assert len(result.sha_short) == SHA_SHORT_LENGTH
    assert result.manifest.docs_archive_sha256.startswith(result.sha_short)

    # docs_dir files extracted
    assert (result.docs_dir / "index.html").is_file()
    assert (result.docs_dir / "library" / "pathlib.html").is_file()

    # manifest written with 5 fields
    assert result.manifest_path.is_file()
    raw = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert raw["docs_archive_sha256"] == result.manifest.docs_archive_sha256

    # current.txt points at sha_short
    current = data_root / "docs" / "3.12" / "current.txt"
    assert current.read_text(encoding="utf-8").strip() == result.sha_short

    # archive cached under full sha
    cache_files = list((cache_root / "archives").iterdir())
    assert len(cache_files) == 1
    assert cache_files[0].name == f"{result.manifest.docs_archive_sha256}.tar.bz2"


def test_ingest_docs_same_sha_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_archive: Path
) -> None:
    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._download_archive", _stub_download(fake_archive))
    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._utc_now_iso", lambda: "2026-04-25T15:00:00Z")
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"

    first = ingest_docs(version="3.12", data_root=data_root, cache_root=cache_root)
    mtime_before = first.manifest_path.stat().st_mtime_ns

    second = ingest_docs(version="3.12", data_root=data_root, cache_root=cache_root)

    assert second.skipped is True
    assert second.sha_short == first.sha_short
    assert second.manifest == first.manifest
    assert second.manifest_path.stat().st_mtime_ns == mtime_before


def test_ingest_docs_sha_conflict_raises_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_v1 = _make_test_archive(tmp_path, "v1", served_version="3.12.13")
    archive_v2 = _make_test_archive(
        tmp_path, "v2", served_version="3.12.14", extra_marker="changed"
    )

    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._utc_now_iso", lambda: "2026-04-25T15:00:00Z")
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"

    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._download_archive", _stub_download(archive_v1))
    ingest_docs(version="3.12", data_root=data_root, cache_root=cache_root)

    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._download_archive", _stub_download(archive_v2))
    with pytest.raises(ShaConflictError):
        ingest_docs(version="3.12", data_root=data_root, cache_root=cache_root)


def test_ingest_docs_force_switch_creates_new_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive_v1 = _make_test_archive(tmp_path, "v1", served_version="3.12.13")
    archive_v2 = _make_test_archive(
        tmp_path, "v2", served_version="3.12.14", extra_marker="changed"
    )

    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._utc_now_iso", lambda: "2026-04-25T15:00:00Z")
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"

    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._download_archive", _stub_download(archive_v1))
    first = ingest_docs(version="3.12", data_root=data_root, cache_root=cache_root)

    monkeypatch.setattr(f"{FETCH_DOCS_MODULE}._download_archive", _stub_download(archive_v2))
    second = ingest_docs(
        version="3.12", force_switch=True, data_root=data_root, cache_root=cache_root
    )

    assert second.sha_short != first.sha_short

    # both subdirs preserved
    assert (data_root / "docs" / "3.12" / first.sha_short).is_dir()
    assert (data_root / "docs" / "3.12" / second.sha_short).is_dir()

    # current.txt points to new sha
    current = data_root / "docs" / "3.12" / "current.txt"
    assert current.read_text(encoding="utf-8").strip() == second.sha_short

    # cache has both archives
    cache_files = sorted((cache_root / "archives").iterdir())
    assert len(cache_files) == 2
