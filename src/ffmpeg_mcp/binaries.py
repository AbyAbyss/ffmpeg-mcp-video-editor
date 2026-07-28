"""Locating (and if necessary downloading) the ffmpeg and ffprobe binaries.

Resolution order:

1. ``FFMPEG_MCP_FFMPEG_PATH`` / ``FFMPEG_MCP_FFPROBE_PATH`` if set.
2. A previously downloaded build cached in ``<workspace>/bin``.
3. A system ffmpeg on ``PATH``, provided it is new enough.
4. A static build downloaded for this OS/architecture.

Checksum policy: builds from BtbN (Linux, Windows) are verified against the
``checksums.sha256`` file published alongside the release, and a mismatch is
fatal. evermeet.cx (macOS) publishes only a GPG signature, no digest, so macOS
downloads are pinned on first use: the hash of the first download is recorded in
``<workspace>/bin/manifest.json`` and any later download of the same URL must
match it. This is noted in the README.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import httpx

from .config import Settings, get_settings
from .errors import BinaryNotFoundError, ChecksumMismatchError, UnsupportedPlatformError

log = logging.getLogger(__name__)

MIN_MAJOR_VERSION = 6
BTBN_RELEASE = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
EVERMEET_INFO = "https://evermeet.cx/ffmpeg/info/{tool}/release"
DOWNLOAD_TIMEOUT = 300.0


@dataclass(frozen=True)
class ResolvedBinaries:
    """Where ffmpeg and ffprobe live, and how they got there."""

    ffmpeg: Path
    ffprobe: Path
    version: str
    source: str  # "configured" | "cached" | "system" | "downloaded"


def _exe(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def read_version(binary: Path | str) -> str | None:
    """Return the first line of ``<binary> -version``, or None if it will not run."""
    try:
        proc = subprocess.run(
            [str(binary), "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    first = proc.stdout.splitlines()[0].strip() if proc.stdout else ""
    return first or None


def parse_major_version(version_line: str) -> int | None:
    """Pull the major version number out of an ffmpeg banner line.

    Handles release banners (``ffmpeg version 7.1``), distro banners
    (``ffmpeg version 6.1.1-3ubuntu5``), BtbN's ``n7.1-...`` tags, and returns
    ``None`` for git-snapshot banners that carry no number at all.
    """
    match = re.search(r"version\s+n?(\d+)\.", version_line)
    if match:
        return int(match.group(1))
    match = re.search(r"version\s+n?(\d+)\b", version_line)
    if match:
        return int(match.group(1))
    return None


def _usable(binary: Path, *, require_min_version: bool) -> str | None:
    line = read_version(binary)
    if line is None:
        return None
    if require_min_version:
        major = parse_major_version(line)
        # A git snapshot with no parseable number is newer than any release.
        if major is not None and major < MIN_MAJOR_VERSION:
            return None
    return line


def _platform_key() -> str:
    machine = platform.machine().lower()
    arm = machine in {"arm64", "aarch64"}
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "winarm64" if arm else "win64"
    if sys.platform.startswith("linux"):
        return "linuxarm64" if arm else "linux64"
    raise UnsupportedPlatformError(
        "No static ffmpeg build is published for this platform.",
        platform=sys.platform,
        machine=machine,
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(client: httpx.Client, url: str, dest: Path) -> None:
    log.info("Downloading %s", url)
    with client.stream("GET", url, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 256):
                handle.write(chunk)


def _pin_manifest(bin_dir: Path) -> Path:
    return bin_dir / "manifest.json"


def _check_pin(bin_dir: Path, url: str, digest: str) -> None:
    """Trust-on-first-use pinning for publishers that ship no digest."""
    manifest_path = _pin_manifest(bin_dir)
    manifest: dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    known = manifest.get(url)
    if known and known != digest:
        raise ChecksumMismatchError(
            "Downloaded archive does not match the hash recorded on first download.",
            url=url,
            expected=known,
            actual=digest,
        )
    if not known:
        log.warning(
            "No publisher checksum available for %s; pinning sha256 %s on first use.", url, digest
        )
        manifest[url] = digest
        manifest_path.write_text(json.dumps(manifest, indent=2))


def _extract_members(archive: Path, dest: Path, wanted: set[str]) -> dict[str, Path]:
    """Extract the named executables out of a tar.xz or zip, flattening the tree."""
    found: dict[str, Path] = {}
    if archive.suffix == ".zip" or zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base in wanted:
                    target = dest / base
                    with zf.open(name) as src, target.open("wb") as out:
                        shutil.copyfileobj(src, out)
                    found[base] = target
    else:
        with tarfile.open(archive) as tf:
            for entry in tf.getmembers():
                base = Path(entry.name).name
                if entry.isfile() and base in wanted:
                    extracted = tf.extractfile(entry)
                    if extracted is None:
                        continue
                    target = dest / base
                    with extracted, target.open("wb") as out:
                        shutil.copyfileobj(extracted, out)
                    found[base] = target
    for path in found.values():
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return found


def _download_btbn(bin_dir: Path, key: str) -> None:
    """Fetch a BtbN build, verified against the release's published sha256 list."""
    suffix = "zip" if key.startswith("win") else "tar.xz"
    name = f"ffmpeg-master-latest-{key}-gpl.{suffix}"
    url = f"{BTBN_RELEASE}/{name}"
    with httpx.Client() as client:
        sums = client.get(f"{BTBN_RELEASE}/checksums.sha256", follow_redirects=True, timeout=60.0)
        sums.raise_for_status()
        expected = None
        for line in sums.text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == name:
                expected = parts[0]
                break
        if expected is None:
            raise ChecksumMismatchError(
                "Release checksum list does not mention the requested build.", asset=name
            )
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / name
            _download(client, url, archive)
            actual = _sha256(archive)
            if actual != expected:
                raise ChecksumMismatchError(
                    "Downloaded ffmpeg archive failed checksum verification.",
                    url=url,
                    expected=expected,
                    actual=actual,
                )
            found = _extract_members(archive, bin_dir, {_exe("ffmpeg"), _exe("ffprobe")})
    missing = {_exe("ffmpeg"), _exe("ffprobe")} - set(found)
    if missing:
        raise BinaryNotFoundError(
            "Downloaded archive did not contain the expected binaries.", missing=sorted(missing)
        )


def _download_evermeet(bin_dir: Path) -> None:
    """Fetch macOS builds from evermeet.cx (one zip per tool, TOFU-pinned)."""
    with httpx.Client() as client:
        for tool in ("ffmpeg", "ffprobe"):
            info = client.get(EVERMEET_INFO.format(tool=tool), follow_redirects=True, timeout=60.0)
            info.raise_for_status()
            url = info.json()["download"]["zip"]["url"]
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / f"{tool}.zip"
                _download(client, url, archive)
                _check_pin(bin_dir, url, _sha256(archive))
                found = _extract_members(archive, bin_dir, {tool})
            if tool not in found:
                raise BinaryNotFoundError(
                    "Downloaded archive did not contain the expected binary.", tool=tool
                )


def download_binaries(settings: Settings) -> None:
    """Download and cache a static ffmpeg build for the current platform."""
    settings.ensure_dirs()
    key = _platform_key()
    if key == "macos":
        _download_evermeet(settings.bin_dir)
    else:
        _download_btbn(settings.bin_dir, key)


def _resolve(settings: Settings) -> ResolvedBinaries:
    ffmpeg_name, ffprobe_name = _exe("ffmpeg"), _exe("ffprobe")

    if settings.ffmpeg_path and settings.ffprobe_path:
        ffmpeg, ffprobe = Path(settings.ffmpeg_path), Path(settings.ffprobe_path)
        version = _usable(ffmpeg, require_min_version=False)
        if version and _usable(ffprobe, require_min_version=False):
            return ResolvedBinaries(ffmpeg, ffprobe, version, "configured")
        raise BinaryNotFoundError(
            "Configured ffmpeg/ffprobe paths are not executable.",
            ffmpeg=str(ffmpeg),
            ffprobe=str(ffprobe),
        )

    cached_ffmpeg = settings.bin_dir / ffmpeg_name
    cached_ffprobe = settings.bin_dir / ffprobe_name
    if cached_ffmpeg.exists() and cached_ffprobe.exists():
        version = _usable(cached_ffmpeg, require_min_version=False)
        if version and _usable(cached_ffprobe, require_min_version=False):
            return ResolvedBinaries(cached_ffmpeg, cached_ffprobe, version, "cached")

    system_ffmpeg = shutil.which("ffmpeg")
    system_ffprobe = shutil.which("ffprobe")
    if system_ffmpeg and system_ffprobe:
        version = _usable(Path(system_ffmpeg), require_min_version=True)
        if version and _usable(Path(system_ffprobe), require_min_version=True):
            return ResolvedBinaries(Path(system_ffmpeg), Path(system_ffprobe), version, "system")
        log.info("System ffmpeg is older than the minimum supported version; downloading a build.")

    if not settings.auto_download_ffmpeg:
        raise BinaryNotFoundError(
            "No usable ffmpeg found and automatic download is disabled.",
            hint="Install ffmpeg >= 6, or set FFMPEG_MCP_FFMPEG_PATH/FFPROBE_PATH.",
        )

    download_binaries(settings)
    version = _usable(cached_ffmpeg, require_min_version=False)
    if not version or not _usable(cached_ffprobe, require_min_version=False):
        raise BinaryNotFoundError("Downloaded ffmpeg build could not be executed.")
    return ResolvedBinaries(cached_ffmpeg, cached_ffprobe, version, "downloaded")


@lru_cache(maxsize=1)
def _cached_resolution(_key: str) -> ResolvedBinaries:
    return _resolve(get_settings())


def get_binaries(settings: Settings | None = None) -> ResolvedBinaries:
    """Resolve ffmpeg/ffprobe, downloading a static build if needed. Cached per process."""
    settings = settings or get_settings()
    key = f"{settings.workspace}|{settings.ffmpeg_path}|{settings.ffprobe_path}"
    return _cached_resolution(key)


def reset_binary_cache() -> None:
    """Forget the resolved binaries. Used by tests."""
    _cached_resolution.cache_clear()


def ffmpeg_env() -> dict[str, str]:
    """Environment for ffmpeg subprocesses: inherited, minus anything locale-noisy."""
    env = dict(os.environ)
    env.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
    return env
