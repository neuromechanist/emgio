"""Shared helper for tests that exercise a real, non-synthetic recording too
large to ship as a repo fixture (issue #123: the on008083 EDF regression test
needs an actual ~192 MiB file, not a hand-built stand-in, to prove the fix
against the exact allocation pattern that triggered the bug).

Real-data tests are opt-in and never run by accident: they need network
access, are slow, and are unrelated to biosigio's platform-independent unit
tests. Two independent gates, both a ``pytest.skip`` (never a hard failure):

* the ``BIOSIGIO_REAL_DATA`` environment variable must be set (any non-empty
  value) to opt in at all.
* the URL must actually be reachable -- a short-timeout request decides
  "offline" so a flaky network never fails the suite.

Usage::

    from biosigio.tests.real_data import fetch_real_recording

    def test_something():
        path = fetch_real_recording(
            "https://data.nemar.org/on008083/v1.0.0/sub-001/ses-01/eeg/"
            "sub-001_ses-01_task-HierPrior_eeg.edf",
            min_bytes=190_000_000,
        )
        ...

The downloaded file is cached under ``BIOSIGIO_REAL_DATA_CACHE`` (default
``~/.cache/biosigio/real_data``), OUTSIDE the repository -- never committed,
and shared across biosigio worktrees/branches on the same machine so a
192 MiB download only ever happens once per host, not once per worktree.

NOTE for other biosigio worktrees adding their own real-data test (e.g. the
view-chunks or units-rescale work): this module did not exist before issue
#123's fix and is intentionally minimal (one function, one cache policy).
Add your own ``fetch_real_recording(url, ...)`` call here rather than
hand-rolling a second download/cache/skip helper, so every real-data test in
the suite shares one cache directory and one opt-in policy.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ENV_VAR = "BIOSIGIO_REAL_DATA"
CACHE_ENV_VAR = "BIOSIGIO_REAL_DATA_CACHE"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "biosigio" / "real_data"
_TIMEOUT_S = 30.0
_CHUNK_BYTES = 4 * 1024 * 1024
# data.nemar.org (and presumably other hosts behind the same WAF/CDN) resets the
# connection outright for the default `Python-urllib/x.y` User-Agent urllib sends
# with none set -- confirmed interactively while wiring this up: identical
# request, only the header differs, `Python-urllib/3.14` -> ConnectionResetError,
# `python-requests/2.0` / a browser UA -> 200. Any non-default UA string clears
# it, so this is a generic anti-bot header check, not a package-specific block.
_USER_AGENT = "biosigio-tests/1.0 (+https://github.com/neuromechanist/biosigio)"


def cache_dir() -> Path:
    """The shared real-data cache directory (overridable via ``BIOSIGIO_REAL_DATA_CACHE``)."""
    return Path(os.environ.get(CACHE_ENV_VAR) or _DEFAULT_CACHE_DIR)


def fetch_real_recording(url: str, *, filename: str | None = None, min_bytes: int = 1) -> Path:
    """Return a local path to ``url``, downloading it into the shared cache on
    first use. Calls ``pytest.skip`` (never raises) when real-data tests are
    not opted into, or the URL cannot be fetched.

    Args:
        url: Direct HTTPS URL to the file.
        filename: Cache filename; defaults to the URL's last path segment.
        min_bytes: Sanity floor on the cached file's size, so a previous
            partial/failed download (or an HTML error page saved under the
            expected name) is not silently reused.

    Returns:
        Path: local path to the cached file. Guaranteed to exist and be at
        least ``min_bytes`` when this returns (otherwise the test was skipped).
    """
    if not os.environ.get(ENV_VAR):
        pytest.skip(
            f"real-data test skipped: set {ENV_VAR}=1 to opt in "
            "(downloads a real recording from the network)"
        )

    dest_dir = cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (filename or url.rsplit("/", 1)[-1])

    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest

    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(_CHUNK_BYTES)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        tmp.unlink(missing_ok=True)
        pytest.skip(f"real-data test skipped: could not fetch {url} ({e})")

    if dest.stat().st_size < min_bytes:
        dest.unlink(missing_ok=True)
        pytest.skip(f"real-data test skipped: download of {url} looked truncated or invalid")

    return dest
