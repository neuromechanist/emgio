"""Typed exceptions for biosigIO.

A small hierarchy so a caller (e.g. the NEMAR Zarr serving pipeline) can classify
*why* a recording could not be read/converted and surface a specific, stable
reason to users, instead of parsing English error strings out of MNE/pyedflib.

Every exception carries a machine-stable ``code`` (string). Catch :class:`BiosigIOError`
and branch on ``.code`` (or ``isinstance``); the codes are part of the public API
and are safe to map to user-facing copy downstream.

    from biosigio import BiosigIOError
    try:
        rec = Recording.from_file(path)
    except BiosigIOError as e:
        reason = REASONS.get(e.code, "could not be read")
        ...

The reverse-mapping of a low-level reader error to one of these types lives in
:func:`classify_read_error`, kept here (with the format knowledge) so importers
share one classifier rather than each re-inventing string matching.
"""

from __future__ import annotations

import errno


class BiosigIOError(ValueError):
    """Base for all biosigIO errors. ``code`` is a stable machine-readable tag.

    Subclasses ``ValueError`` so existing ``except ValueError`` / ``raises(ValueError)``
    callers keep working as these typed errors are introduced (they are all "bad
    input/value" conditions about a recording)."""

    code = "biosigio_error"


class UnsupportedFormatError(BiosigIOError):
    """The file's format/extension has no biosigIO reader (e.g. an unknown
    extension, or a format only a missing optional extra could read)."""

    code = "unsupported_format"


class FileReadError(BiosigIOError):
    """A recognized format failed to read. Base for the specific read failures;
    raised directly as the generic fallback when no specific cause is identified."""

    code = "file_read_error"


class NotContinuousRecordingError(FileReadError):
    """The file holds no continuous raw time series -- typically a trial-averaged
    (evoked, ``*-ave.fif``) or epoched (``*-epo.fif``) derivative. The data is
    valid, but the serving/viewer path only handles continuous recordings."""

    code = "not_continuous"


class CorruptFileError(FileReadError):
    """The source is truncated, not format-compliant, or part of an incomplete
    set: an EDF that fails the ``Filesize`` check, an incomplete CTF ``.meg4``,
    or a split recording missing chain members."""

    code = "corrupt_or_truncated"


class EmptyRecordingError(FileReadError):
    """The file read but contains no signal channels to plot."""

    code = "empty_recording"


class MixedSamplingRateError(BiosigIOError):
    """An EDF/BDF carries differing per-channel sampling rates and the caller's
    ``mixed_rate`` policy is ``"error"`` (the default). Not a read failure -- the
    caller can re-read with ``mixed_rate="resample"``. The serving pipeline does
    exactly that, so this rarely surfaces to users."""

    code = "mixed_sampling_rate"


# code -> default human-facing reason. Downstream UIs may override the copy, but
# the codes are stable.
REASONS: dict[str, str] = {
    UnsupportedFormatError.code: "This file format is not yet supported by the viewer.",
    NotContinuousRecordingError.code: (
        "This file is a trial-averaged or epoched derivative, not a continuous "
        "recording, so the time-series viewer is not available."
    ),
    CorruptFileError.code: (
        "This recording's data file appears truncated or corrupt, so the viewer "
        "could not be generated."
    ),
    EmptyRecordingError.code: "This recording contains no signal channels to display.",
    MixedSamplingRateError.code: (
        "This recording mixes per-channel sampling rates; a viewable copy needs "
        "resampling to a common grid."
    ),
    FileReadError.code: "This recording could not be prepared for viewing.",
    BiosigIOError.code: "This recording could not be prepared for viewing.",
}


# Thread/resource-exhaustion messages that surface as OSError/RuntimeError rather
# than MemoryError. Reader-agnostic substrings (CPython's own `_thread` module and
# libc's ENOMEM/EAGAIN strerror text), matched case-insensitively.
_THREAD_OR_RESOURCE_EXHAUSTION_SUBSTRINGS = (
    "can't start new thread",
    "cannot allocate memory",  # ENOMEM strerror text (not every OSError carries .errno)
)
_RESOURCE_EXHAUSTION_ERRNOS = (errno.ENOMEM, errno.EAGAIN)


def is_resource_exhaustion(exc: Exception) -> bool:
    """True when ``exc`` reflects the host running out of memory or threads,
    not a problem with the file being read.

    Resource exhaustion is never a property of the file: a ``MemoryError`` (or
    numpy's ``_ArrayMemoryError``, a ``MemoryError`` subclass) raised mid-read
    typically means the host was saturated (e.g. many parallel conversions),
    not that the recording is corrupt. Importers must let it -- and the
    thread/allocation-exhaustion ``OSError``/``RuntimeError`` variants below --
    propagate unchanged instead of reclassifying it as a permanent read
    failure, so a caller (e.g. NEMAR's converter) can retry instead of
    recording a deterministic failure. See :func:`classify_read_error`.
    """
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, (OSError, RuntimeError)):
        if getattr(exc, "errno", None) in _RESOURCE_EXHAUSTION_ERRNOS:
            return True
        low = str(exc).lower()
        if any(s in low for s in _THREAD_OR_RESOURCE_EXHAUSTION_SUBSTRINGS):
            return True
    return False


def classify_read_error(exc: Exception, filepath: str = "") -> BiosigIOError:
    """Map a low-level read failure to a typed biosigIO error with a stable code.

    Importers call this in their ``except`` and ``raise ... from exc`` the result,
    so the *why* is decided once, where the format context lives. An exception
    that is already a :class:`BiosigIOError` is returned unchanged.

    Resource exhaustion (:func:`is_resource_exhaustion`) is never classified as a
    read error either: this function re-raises it as-is (defence in depth for a
    caller that calls this directly, without its own pre-check guard -- see the
    importers, which all guard *before* calling this so the common path never
    even reaches here for a MemoryError).

    The matching is deliberately reader-agnostic (MNE and pyedflib phrase the same
    condition differently); unmatched failures fall through to :class:`FileReadError`
    so a caller always gets a typed error, never a bare ``Exception``.
    """
    if isinstance(exc, BiosigIOError):
        return exc
    if is_resource_exhaustion(exc):
        raise exc
    msg = str(exc)
    low = msg.lower()
    where = f" ({filepath})" if filepath else ""

    # Evoked / epoched / averaged derivatives -> no continuous raw time series.
    # MNE: "No raw data in <f>", "... is not a raw file", read_raw on an -ave/-epo.
    if (
        "no raw data" in low
        or "is not a raw" in low
        or "does not contain raw" in low
        or "this is not a raw" in low
    ):
        return NotContinuousRecordingError(f"No continuous raw recording{where}: {msg}")

    # Truncated / not format-compliant / incomplete set.
    # pyedflib: "... not EDF(+) or BDF(+) compliant (Filesize)"; MNE CTF/FIF:
    # incomplete .meg4; split chain: "... split-02 ... does not exist".
    if (
        "filesize" in low
        or "not edf" in low
        or "not bdf" in low
        or "compliant" in low
        or "truncate" in low  # matches "truncated" too
        or "corrupt" in low
        or ("split" in low and "does not exist" in low)
        # CTF .meg4 chopped mid-trial (read_raw_ctf): the data no longer divides
        # into whole trials -- the on004398 truncation signature.
        or "even multiple of the trial" in low
        or "not an even multiple" in low
    ):
        return CorruptFileError(f"Truncated, incomplete, or corrupt file{where}: {msg}")

    # Recognized format, no specific cause -> generic (still typed) read error.
    return FileReadError(f"Could not read recording{where}: {msg}")
