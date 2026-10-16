"""A refusal the streaming EDF path cannot recover is a typed file failure.

The in-memory importer wraps pyedflib's refusal with ``classify_read_error``;
the streaming path used to re-raise pyedflib's bare ``OSError``, which callers
cannot tell apart from a host I/O failure (issue #149).
"""

import errno
import os
import sys

import pyedflib
import pytest

from biosigio import stream_to_zarr
from biosigio.exceptions import BiosigIOError, CorruptFileError

from .test_zarr_stream import _write_edf


def test_stream_refused_edf_raises_corrupt_file_error(tmp_path):
    path = str(tmp_path / "truncated.edf")
    _write_edf(path, 100.0, 5.0, 2)
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - 200)  # chop off part of the last record

    with pytest.raises(CorruptFileError, match="Filesize") as info:
        stream_to_zarr(path, str(tmp_path / "store.zarr"), bids_channels="off")

    # The reader's own refusal rides along as the cause, as on the in-memory path.
    cause = info.value.__cause__
    assert isinstance(cause, OSError)
    assert "Filesize" in str(cause)


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not deny reads to root or on Windows",
)
def test_stream_unreadable_edf_raises_the_os_error_unchanged(tmp_path):
    """The file is fine; this host may not read it. pyedflib reports the refused
    ``open()`` without its errno, so the streaming source recovers the real
    ``PermissionError`` and raises it, exactly as the in-memory importer does,
    instead of typing it as a permanent file failure."""
    path = str(tmp_path / "unreadable.edf")
    _write_edf(path, 100.0, 5.0, 2)
    os.chmod(path, 0)
    try:
        with pytest.raises(PermissionError) as expected:  # the fixture really is unreadable
            open(path, "rb").close()

        with pytest.raises(OSError) as info:
            stream_to_zarr(path, str(tmp_path / "store.zarr"), bids_channels="off")
    finally:
        os.chmod(path, 0o600)

    assert type(info.value) is type(expected.value) is PermissionError
    assert info.value.errno == expected.value.errno == errno.EACCES
    assert not isinstance(info.value, BiosigIOError)


def test_stream_memory_error_on_open_propagates_unchanged(tmp_path, monkeypatch):
    """Streaming counterpart of ``test_edf_memory_error_on_open_propagates_unchanged``
    (test_edf_memory_error.py), mirrored the same way: ``EdfReader.__init__`` is
    patched to raise ``MemoryError``, which simulates the operating system
    refusing an allocation rather than faking pyedflib's parsing. A host
    condition at open must not be re-typed as a file failure."""
    path = str(tmp_path / "rec.edf")
    _write_edf(path, 100.0, 5.0, 2)

    def raise_memory_error(self, *args, **kwargs):
        raise MemoryError("Unable to allocate 500.0 MiB for an array")

    monkeypatch.setattr(pyedflib.EdfReader, "__init__", raise_memory_error)

    with pytest.raises(MemoryError) as info:
        stream_to_zarr(path, str(tmp_path / "store.zarr"), bids_channels="off")
    assert not isinstance(info.value, BiosigIOError)


def test_stream_truncated_and_degenerate_raises_corrupt(tmp_path):
    """Streaming twin of ``test_truncated_and_degenerate_still_raises_corrupt``
    (test_edf_fallback.py): a file that is BOTH truncated AND carries a
    degenerate channel. pyedflib names the degenerate range, not the truncation,
    so the streaming source takes the tolerant fallback, whose truncation check
    raises a bare errno-less ``OSError``. It must be typed like the in-memory
    path types it, not leak out of ``stream_to_zarr`` untyped."""
    pytest.importorskip("mne", reason="the tolerant EDF fallback requires the 'meg' extra (mne)")
    import numpy as np

    from .test_edf_fallback import _channel, _patch_signal_field
    from .test_edf_fallback import _write_edf as _write_channels_edf

    path = str(tmp_path / "truncated_degenerate.edf")
    rng = np.random.default_rng(2)
    channels = [_channel("C0", 100.0, -100.0, 100.0), _channel("REF", 100.0, -100.0, 100.0)]
    _write_channels_edf(path, channels, [rng.uniform(-100, 100, 400) for _ in channels])
    _patch_signal_field(path, 1, "physical_max", b"-100")
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - 200)

    # pyedflib reports the degenerate range, NOT the truncation, so the fallback
    # is genuinely what runs here.
    with pytest.raises(OSError, match=r"(?i)physical maximum"):
        pyedflib.EdfReader(path)

    with pytest.raises(CorruptFileError, match="truncated") as info:
        stream_to_zarr(path, str(tmp_path / "store.zarr"), bids_channels="off")

    cause = info.value.__cause__
    assert isinstance(cause, OSError)
    assert "file is truncated" in str(cause)


# --- Host conditions inside the tolerant fallback ---------------------------------
#
# The streaming source recovers a file pyedflib refuses by reading it through
# MNE (``read_edf_tolerant``). A failure of that read is typed like the in-memory
# importer types it, EXCEPT a host condition: the fallback site calls
# ``classify_read_error``, which re-raises those unchanged. Without these tests
# the site could be replaced by an unconditional typed error and nothing here
# would notice.

_FALLBACK_KINDS = {
    "degenerate_range": r"(?i)physical maximum",
    "nul_padded_field": r"(?i)number of datarecords",
}


def _write_fallback_edf(path: str, kind: str) -> None:
    """A real EDF that pyedflib refuses and the probe recovers.

    Built with the byte patches ``test_edf_fallback.py`` uses for the same two
    conditions, so the fixture is the file those tests already prove recoverable.
    """
    import numpy as np

    from .test_edf_fallback import _channel, _patch_main_header_field, _patch_signal_field
    from .test_edf_fallback import _write_edf as _write_channels_edf

    rng = np.random.default_rng(150)
    channels = [_channel("C0", 100.0, -100.0, 100.0), _channel("REF", 100.0, -100.0, 100.0)]
    _write_channels_edf(path, channels, [rng.uniform(-100, 100, 400) for _ in channels])
    if kind == "degenerate_range":
        _patch_signal_field(path, 1, "physical_max", b"-100")
    else:
        # Four one-second data records, NUL-padded instead of space-padded.
        _patch_main_header_field(path, 236, 8, b"4", pad=b"\x00")

    # pyedflib refuses the file, so the fallback is genuinely what runs.
    with pytest.raises(OSError, match=_FALLBACK_KINDS[kind]):
        pyedflib.EdfReader(path)


def _fail_fallback_read(monkeypatch, error: BaseException) -> list:
    """Make the fallback's own read raise ``error``.

    ``read_edf_tolerant`` opens the recording with ``mne.io.read_raw``, the
    lowest read call it makes, so that is what is patched: the file, the probe,
    the truncation check and pyedflib's refusal are all real, and only the
    operating system's failure of this one read is simulated. Returns the list
    of calls, which proves the failure was raised from inside the fallback.
    """
    mne = pytest.importorskip("mne", reason="the tolerant EDF fallback requires the 'meg' extra")
    calls: list = []

    def read_raw(*args, **kwargs):
        calls.append(args)
        raise error

    monkeypatch.setattr(mne.io, "read_raw", read_raw)
    return calls


@pytest.mark.parametrize("kind", list(_FALLBACK_KINDS))
def test_stream_fallback_memory_error_propagates_unchanged(tmp_path, monkeypatch, kind):
    """A ``MemoryError`` inside the fallback read is the host running out of
    memory, not a property of the recording: it must reach the caller as the
    very object raised, not re-typed as a permanent file failure."""
    pytest.importorskip("mne", reason="the tolerant EDF fallback requires the 'meg' extra")
    path = str(tmp_path / f"{kind}.edf")
    _write_fallback_edf(path, kind)
    error = MemoryError("Unable to allocate 500.0 MiB for an array")
    calls = _fail_fallback_read(monkeypatch, error)

    with pytest.raises(MemoryError) as info:
        stream_to_zarr(path, str(tmp_path / "store.zarr"), bids_channels="off")

    assert len(calls) == 1  # raised from inside the fallback, not before it
    assert info.value is error
    assert not isinstance(info.value, BiosigIOError)


@pytest.mark.parametrize("kind", list(_FALLBACK_KINDS))
def test_stream_fallback_eio_propagates_as_the_os_error(tmp_path, monkeypatch, kind):
    """A failing disk (``EIO``) during the fallback read says something about the
    host, not the recording. The ``OSError`` carries a host error as its cause, as
    a reader that re-raises from a lower-level failure produces; it must reach the
    caller as the raw ``OSError`` with its errno, never as a ``BiosigIOError``."""
    pytest.importorskip("mne", reason="the tolerant EDF fallback requires the 'meg' extra")
    path = str(tmp_path / f"{kind}.edf")
    _write_fallback_edf(path, kind)
    device_failure = OSError(errno.EIO, os.strerror(errno.EIO))
    error = OSError(errno.EIO, "Could not read the recording")
    error.__cause__ = device_failure
    calls = _fail_fallback_read(monkeypatch, error)

    with pytest.raises(OSError) as info:
        stream_to_zarr(path, str(tmp_path / "store.zarr"), bids_channels="off")

    assert len(calls) == 1
    assert info.value is error
    assert type(info.value) is OSError
    assert info.value.errno == errno.EIO
    assert info.value.__cause__ is device_failure
    assert not isinstance(info.value, BiosigIOError)
