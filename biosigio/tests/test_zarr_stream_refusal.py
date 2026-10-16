"""A refusal the streaming EDF path cannot recover is a typed file failure.

The in-memory importer wraps pyedflib's refusal with ``classify_read_error``;
the streaming path used to re-raise pyedflib's bare ``OSError``, which callers
cannot tell apart from a host I/O failure (issue #149).
"""

import os

import pytest

from biosigio import stream_to_zarr
from biosigio.exceptions import CorruptFileError

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
