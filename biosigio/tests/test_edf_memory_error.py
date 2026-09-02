"""Resource exhaustion must propagate unchanged out of the EDF importer (#123).

`biosigio/importers/edf.py` used to wrap the whole ``load()`` body -- including
the per-channel ``readSignal`` loop -- in one ``except Exception as e: raise
classify_read_error(e, filepath) from e``. A ``MemoryError`` raised mid-loop
(numpy's ``_ArrayMemoryError``, a subclass) was therefore re-typed as a
permanent ``FileReadError(code="file_read_error")``: NEMAR's converter has a
dedicated ``except MemoryError`` branch that yields the retryable
``recording_memory_exceeded`` code, but it never saw the MemoryError because
biosigio had already reclassified it. 36 of 43 recordings in dataset on008083
were published as a deterministic (never-retried) failure this way.

Two tests here:

* a synthetic file, monkeypatching ``pyedflib.EdfReader.readSignal`` to raise
  ``MemoryError`` inside the channel loop -- this simulates the operating
  system's resource-exhaustion condition (the OS refusing an allocation), not
  business logic, so it is not the mock-of-business-logic the project's NO
  MOCK policy forbids; nothing about pyedflib's own read/parse behavior is
  faked.
* the real on008083 recording (project owner requirement: a synthetic
  reproduction alone is not sufficient proof), gated behind
  ``biosigio.tests.real_data`` so it only runs when explicitly opted into.
"""

import os
import tempfile

import numpy as np
import pyedflib
import pytest

from biosigio import Recording
from biosigio.exceptions import FileReadError
from biosigio.tests.real_data import fetch_real_recording

# The exact file from the issue: dataset on008083, sub-001, 43 recordings of
# which 36 hit this bug during a saturated (24-parallel-conversion) Hallu run.
# NOTE: the URL in the issue omits the dataset version segment
# (`/on008083/sub-001/...` -> 404); data.nemar.org serves under
# `/on008083/v1.0.0/...` per its own manifest.json. ~192 MiB (201,611,808 bytes).
_ON008083_SUB001_URL = (
    "https://data.nemar.org/on008083/v1.0.0/sub-001/ses-01/eeg/"
    "sub-001_ses-01_task-HierPrior_eeg.edf"
)
_ON008083_SUB001_MIN_BYTES = 190_000_000  # sanity floor; real file is ~192.3 MiB
# sha256 of the exact bytes at the URL above, per its dataset manifest
# (https://data.nemar.org/on008083/v1.0.0/manifest.json) and independently
# confirmed via `shasum -a 256` on the downloaded file. Guards the cache
# against a truncated/substituted/corrupted entry (see
# biosigio.tests.real_data.fetch_real_recording's sha256 parameter).
_ON008083_SUB001_SHA256 = "545bb6afeea54b987c62f47382ecd0ca9738667177cebfb727e0c83cfde36efc"


def _write_minimal_edf(path: str, n_channels: int = 2, n_samples: int = 1000) -> None:
    """A tiny real EDF, written with pyedflib's own writer (no hand-crafted bytes)."""
    rate = 100.0
    w = pyedflib.EdfWriter(path, n_channels)
    try:
        w.setSignalHeaders(
            [
                {
                    "label": f"EEG{i}",
                    "dimension": "uV",
                    "sample_frequency": rate,
                    "physical_max": 100.0,
                    "physical_min": -100.0,
                    "digital_max": 32767,
                    "digital_min": -32768,
                    "prefilter": "n/a",
                    "transducer": "n/a",
                }
                for i in range(n_channels)
            ]
        )
        w.writeSamples([np.zeros(n_samples) for _ in range(n_channels)])
    finally:
        w.close()


# -- synthetic: MemoryError inside the readSignal loop --------------------------


def test_edf_memory_error_in_channel_loop_propagates_unchanged(monkeypatch):
    """Provoke the exact on008083 failure mode: readSignal raises MemoryError
    partway through the per-channel loop. The importer must let it propagate
    as MemoryError, not reclassify it as FileReadError."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rec.edf")
        _write_minimal_edf(path, n_channels=3)

        real_read_signal = pyedflib.EdfReader.readSignal
        calls = {"n": 0}

        def flaky_read_signal(self, signal_idx, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:  # fail on the second channel, mid-loop
                raise MemoryError(
                    "Unable to allocate 24.8 MiB for an array with shape "
                    "(3248000,) and data type float64"
                )
            return real_read_signal(self, signal_idx, *args, **kwargs)

        monkeypatch.setattr(pyedflib.EdfReader, "readSignal", flaky_read_signal)

        with pytest.raises(MemoryError) as exc_info:
            Recording.from_file(path)
        assert not isinstance(exc_info.value, FileReadError)
        assert calls["n"] >= 2  # actually reached the loop, not an early failure


def test_edf_memory_error_on_open_propagates_unchanged(monkeypatch):
    """A MemoryError raised while opening the file (before the channel loop)
    takes a different internal path (the pyedflib-open except/fallback branch),
    which must also let it through unchanged rather than routing it into the
    tolerant-fallback or classify_read_error path."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rec.edf")
        _write_minimal_edf(path, n_channels=1)

        def raise_memory_error(self, *args, **kwargs):
            raise MemoryError("Unable to allocate 500.0 MiB for an array")

        monkeypatch.setattr(pyedflib.EdfReader, "__init__", raise_memory_error)

        with pytest.raises(MemoryError):
            Recording.from_file(path)


# -- real data: the actual on008083 recording (project owner requirement) ------


def test_real_on008083_edf_reads_normally():
    """The real on008083 sub-001 recording reads cleanly end-to-end (no
    provocation) -- the baseline this bug's false-positive rejections
    contradicted (every file reads cleanly standalone per the issue)."""
    path = fetch_real_recording(
        _ON008083_SUB001_URL, min_bytes=_ON008083_SUB001_MIN_BYTES, sha256=_ON008083_SUB001_SHA256
    )
    rec = Recording.from_file(str(path), mixed_rate="resample")
    assert rec.signals is not None
    assert len(rec.channels) > 0
    assert len(rec.signals) > 0


def test_real_on008083_edf_memory_error_propagates_unchanged(monkeypatch):
    """Same real file, with the exact provocation from the issue: readSignal
    raises MemoryError partway through the channel loop. Must surface as
    MemoryError, not FileReadError."""
    path = fetch_real_recording(
        _ON008083_SUB001_URL, min_bytes=_ON008083_SUB001_MIN_BYTES, sha256=_ON008083_SUB001_SHA256
    )

    real_read_signal = pyedflib.EdfReader.readSignal
    calls = {"n": 0}

    def flaky_read_signal(self, signal_idx, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise MemoryError(
                "Unable to allocate 24.8 MiB for an array with shape "
                "(3248000,) and data type float64"
            )
        return real_read_signal(self, signal_idx, *args, **kwargs)

    monkeypatch.setattr(pyedflib.EdfReader, "readSignal", flaky_read_signal)

    with pytest.raises(MemoryError) as exc_info:
        Recording.from_file(str(path), mixed_rate="resample")
    assert not isinstance(exc_info.value, FileReadError)
    assert calls["n"] >= 2
