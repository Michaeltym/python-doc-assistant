"""Download Python documentation archives and record an ingest manifest.

See plans/v0-retrieval-eval.md §1 for the full contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Iterable, Iterator

import requests
from bs4 import BeautifulSoup

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

ARCHIVE_URL_TEMPLATE: Final[str] = (
    "https://docs.python.org/{version}/archives/python-{version}-docs-html.tar.bz2"
)

DEFAULT_HTTP_TIMEOUT_SECONDS: Final[float] = 60.0
DEFAULT_HTTP_RETRIES: Final[int] = 3
DEFAULT_USER_AGENT: Final[str] = "python-doc-assistant/0.1 (+local)"

SHA_SHORT_LENGTH: Final[int] = 12

DEFAULT_DATA_ROOT: Final[Path] = Path("data")
DEFAULT_CACHE_ROOT: Final[Path] = Path.home() / ".cache" / "python-doc-assistant"


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class IngestError(Exception):
    """Base class for ingest failures."""


class ShaConflictError(IngestError):
    """Newly downloaded archive sha differs from current.txt and force_switch=False."""


class ArchiveDownloadError(IngestError):
    """HTTP download or tar extraction failed."""


class ManifestParseError(IngestError):
    """Could not parse docs_served_version from the unpacked HTML."""


# ------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------


@dataclass(frozen=True)
class IngestManifest:
    """The 5 fields recorded in ingest_manifest.json (per PLAN.md §4)."""

    docs_version: str  # major.minor, e.g. "3.12"
    docs_served_version: str  # actual patch parsed from HTML, e.g. "3.12.13"
    docs_url: str
    docs_archive_sha256: str  # full hex digest (not the truncated short form)
    ingest_timestamp: str  # ISO 8601 UTC


@dataclass(frozen=True)
class IngestResult:
    """Output of ingest_docs(): corpus location + manifest snapshot."""

    sha_short: str  # first SHA_SHORT_LENGTH chars of docs_archive_sha256
    docs_dir: Path  # data/docs/<version>/<sha_short>/
    manifest_path: Path  # data/docs/<version>/<sha_short>/ingest_manifest.json
    manifest: IngestManifest
    skipped: bool  # True if same sha already active (no unpack performed)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def ingest_docs(
    version: str,
    *,
    force_switch: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    http_timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    http_retries: int = DEFAULT_HTTP_RETRIES,
    user_agent: str = DEFAULT_USER_AGENT,
) -> IngestResult:
    """Download, verify, and unpack the Python docs archive for `version`.

    Contract (per plans/v0-retrieval-eval.md §1):
        - Idempotent: same sha re-ingested → return with skipped=True (no unpack).
        - Sha conflict (different sha from current.txt) → ShaConflictError unless
          force_switch=True (then create a new <sha_short> subdir + update current.txt).
        - On success: writes ingest_manifest.json, updates current.txt, caches the archive.

    Suggested flow:
         1. url        = _resolve_archive_url(version)
         2. tmp_arch   = _download_archive(url, ..., timeout=..., retries=...)
         3. full_sha   = _compute_sha256(tmp_arch)
         4. sha_short  = full_sha[:SHA_SHORT_LENGTH]
         5. matched    = _check_sha_conflict(version, sha_short, force_switch, data_root)
                         # returns True if the sha is already active → caller skips
         6. docs_dir   = data_root / "docs" / version / sha_short
         7. _unpack_archive(tmp_arch, docs_dir)
         8. served     = _parse_docs_served_version(docs_dir)
         9. manifest   = IngestManifest(version, served, url, full_sha, _utc_now_iso())
        10. _write_manifest(docs_dir, manifest)
        11. _update_current_txt(data_root, version, sha_short)
        12. _cache_archive(tmp_arch, cache_root, full_sha)
        13. return IngestResult(sha_short, docs_dir, ..., skipped=False)
    """
    url = _resolve_archive_url(version)
    with tempfile.TemporaryDirectory(prefix="pdr-ingest-") as tmp_dir_str:
        tmp_archive_path = Path(tmp_dir_str) / "archive.tar.bz2"
        tmp_archive = _download_archive(
            url,
            tmp_archive_path,
            timeout=http_timeout,
            retries=http_retries,
            user_agent=user_agent,
        )
        new_sha = _compute_sha256(tmp_archive)
        new_sha_short = new_sha[:SHA_SHORT_LENGTH]
        should_skip = _check_sha_conflict(version, new_sha_short, force_switch, data_root)
        if not should_skip:
            unpack_path_dir = data_root / "docs" / version / new_sha_short
            _unpack_archive(tmp_archive, unpack_path_dir, version)
            served_version = _parse_docs_served_version(unpack_path_dir)
            manifest = IngestManifest(
                docs_version=version,
                docs_served_version=served_version,
                docs_url=url,
                docs_archive_sha256=new_sha,
                ingest_timestamp=_utc_now_iso(),
            )
            manifest_path = _write_manifest(unpack_path_dir, manifest)
            _update_current_txt(data_root, version, new_sha_short)
            _cache_archive(tmp_archive, cache_root, new_sha)
            return IngestResult(
                sha_short=new_sha_short,
                docs_dir=unpack_path_dir,
                manifest_path=manifest_path,
                manifest=manifest,
                skipped=False,
            )

        docs_dir = data_root / "docs" / version / new_sha_short
        manifest_path = docs_dir / "ingest_manifest.json"
        manifest = _read_manifest(manifest_path)
        return IngestResult(
            sha_short=new_sha_short,
            docs_dir=docs_dir,
            manifest_path=manifest_path,
            manifest=manifest,
            skipped=True,
        )


# ------------------------------------------------------------------
# Helpers (private)
# ------------------------------------------------------------------


def _resolve_archive_url(version: str) -> str:
    """Format ARCHIVE_URL_TEMPLATE with `version`."""
    return ARCHIVE_URL_TEMPLATE.format(version=version)


def _download_archive(
    url: str,
    dest: Path,
    *,
    timeout: float,
    retries: int,
    user_agent: str,
) -> Path:
    """Stream-download `url` to `dest`; retry transient failures with backoff.

    On final failure raise ArchiveDownloadError, chaining the original exception.
    """
    exception = None
    for retry in range(1, retries + 1):
        try:
            with requests.get(
                url, stream=True, timeout=timeout, headers={"User-Agent": user_agent}
            ) as r:
                r.raise_for_status()

                tmp_path = dest.parent / (dest.name + ".part")
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)

                tmp_path.replace(dest)
                return dest
        except Exception as e:
            exception = e
            if retry < retries:
                time.sleep(2**retry)

    raise ArchiveDownloadError from exception


def _compute_sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of `path` (full digest)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _extract_tar_info(
    members: Iterable[tarfile.TarInfo], version: str
) -> Iterator[tarfile.TarInfo]:
    top_level_name = f"python-{version}-docs-html/"
    for member in members:
        if not member.name.startswith(top_level_name):
            continue
        member.name = member.name.removeprefix(top_level_name)
        if not member.name:
            continue
        yield member


def _unpack_archive(archive_path: Path, dest_dir: Path, version: str) -> None:
    """Extract `archive_path` into `dest_dir`, flattening the top-level
    `python-<version>-docs-html/` directory so files land at e.g.
    `dest_dir/library/pathlib.html`.

    Use tarfile `filter='data'` (PEP 706) to reject absolute paths / parent traversal.
    """
    if not tarfile.is_tarfile(archive_path):
        raise ArchiveDownloadError(f"Invalid tar file: {archive_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:bz2") as tar:
        tar.extractall(members=_extract_tar_info(tar, version), path=dest_dir, filter="data")


def _parse_docs_served_version(docs_dir: Path) -> str:
    """Parse the patch version (e.g. "3.12.13") from the unpacked HTML.

    Strategy: read `docs_dir/index.html` (or a known landing page), grab the
    `<title>` tag, and pull out the major.minor.patch string.
    Raise ManifestParseError on miss.
    """
    path = docs_dir / "index.html"
    if not path.is_file():
        raise ManifestParseError(f"index.html missing in {docs_dir}")
    with path.open(encoding="utf-8") as f:
        html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            match = re.search(r"\b\d+\.\d+\.\d+\b", soup.title.string)
            if match:
                return match.group()

    raise ManifestParseError


def _check_sha_conflict(
    version: str,
    new_sha_short: str,
    force_switch: bool,
    data_root: Path,
) -> bool:
    """Compare `new_sha_short` against `data_root/docs/<version>/current.txt`.

    Returns:
        True  -- new sha matches the current active sha (idempotent; skip work).
        False -- no current.txt yet, OR sha differs and force_switch=True.
    Raises:
        ShaConflictError -- sha differs and force_switch=False.
    """
    path = data_root / "docs" / version / "current.txt"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            current_sha_short = f.read().strip()
            if current_sha_short == new_sha_short:
                return True
            elif not force_switch:
                raise ShaConflictError

    return False


def _write_manifest(docs_dir: Path, manifest: IngestManifest) -> Path:
    """Write `manifest` as JSON to `docs_dir/ingest_manifest.json`. Returns the path.

    Write atomically (tmp file + os.replace) so a killed process never leaves a
    torn manifest behind.
    """
    tmp_path = None
    try:
        path = docs_dir / "ingest_manifest.json"
        with tempfile.NamedTemporaryFile("w", dir=docs_dir, delete=False, encoding="utf-8") as f:
            json.dump(asdict(manifest), f)
            f.flush()
            os.fsync(f.fileno())
            tmp_path = f.name

        os.replace(tmp_path, path)
        return path
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _read_manifest(path: Path) -> IngestManifest:
    """Load and deserialize an existing ingest_manifest.json into IngestManifest."""
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
            return IngestManifest(**manifest)

    raise ManifestParseError(f"manifest not found at {path}")


def _update_current_txt(data_root: Path, version: str, sha_short: str) -> None:
    """Atomically write `sha_short` into `data_root/docs/<version>/current.txt`."""
    tmp_path = None
    try:
        path = data_root / "docs" / version / "current.txt"
        version_dir = data_root / "docs" / version
        with tempfile.NamedTemporaryFile("w", dir=version_dir, delete=False, encoding="utf-8") as f:
            f.write(sha_short)
            f.flush()
            os.fsync(f.fileno())
            tmp_path = f.name
        os.replace(tmp_path, path)

    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _cache_archive(archive_path: Path, cache_root: Path, full_sha: str) -> Path:
    """Copy `archive_path` to `cache_root/archives/<full_sha>.tar.bz2`.

    Skip if the cache file already exists (content-addressed, so identical by sha).
    Returns the cache path.
    """
    archives_dir = cache_root / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    target = cache_root / "archives" / f"{full_sha}.tar.bz2"
    if not target.exists():
        shutil.copy(archive_path, target)
    return target


def _utc_now_iso() -> str:
    """Current UTC time as ISO 8601 with second precision, e.g. "2026-04-25T14:30:00Z"."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
