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
