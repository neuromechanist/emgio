"""Typed read-error classification (biosigio.exceptions).

A caller (the NEMAR Zarr pipeline) needs to know *why* a recording could not be
read so it can surface a specific reason. These tests pin the classifier and the
importers' typed raises on REAL files (no mocks):

* an unknown extension -> UnsupportedFormatError
* a real MNE evoked (``*-ave.fif``: averaged, no continuous raw) -> NotContinuousRecordingError
* a truncated EDF -> CorruptFileError
* mixed-rate EDF -> MixedSamplingRateError (still a ValueError, for back-compat)

The string-matching unit tests cover the reader phrasings we map without needing
each format on hand.
"""

import importlib.util
import os
import tempfile

import numpy as np
import pyedflib
import pytest

from biosigio import Recording
from biosigio.exceptions import (
    REASONS,
    BiosigIOError,
    CorruptFileError,
    FileReadError,
    MixedSamplingRateError,
    NotContinuousRecordingError,
    UnsupportedFormatError,
    classify_read_error,
)

_HAS_MNE = importlib.util.find_spec("mne") is not None


# -- classify_read_error: reader-phrasing -> typed error -----------------------


def test_classify_no_raw_data_is_not_continuous():
    err = classify_read_error(ValueError("No raw data in /x/sub-01_task-a-ave.fif"))
    assert isinstance(err, NotContinuousRecordingError)
    assert err.code == "not_continuous"


def test_classify_not_a_raw_file_is_not_continuous():
    err = classify_read_error(RuntimeError("This is not a raw file"))
    assert isinstance(err, NotContinuousRecordingError)


def test_classify_filesize_is_corrupt():
    err = classify_read_error(OSError("the file is not EDF(+) or BDF(+) compliant (Filesize)"))
    assert isinstance(err, CorruptFileError)
    assert err.code == "corrupt_or_truncated"


def test_classify_incomplete_split_is_corrupt():
    err = classify_read_error(
        ValueError("Split raw file detected but next file ..._split-02_meg.fif does not exist")
    )
    assert isinstance(err, CorruptFileError)


def test_classify_unknown_falls_back_to_file_read_error():
    err = classify_read_error(RuntimeError("something unexpected"))
    assert type(err) is FileReadError
    assert err.code == "file_read_error"


def test_classify_passes_through_existing_typed_error():
    original = NotContinuousRecordingError("already typed")
    assert classify_read_error(original) is original


def test_classify_includes_filepath_when_given():
    err = classify_read_error(ValueError("No raw data"), "/data/sub-01_task-a-ave.fif")
    assert "sub-01_task-a-ave.fif" in str(err)


# -- typed errors are ValueErrors (back-compat) + have stable codes ------------


def test_typed_errors_are_valueerrors():
    for exc in (
        UnsupportedFormatError,
        FileReadError,
        NotContinuousRecordingError,
        CorruptFileError,
        MixedSamplingRateError,
    ):
        assert issubclass(exc, BiosigIOError)
        assert issubclass(exc, ValueError)


def test_every_code_has_a_reason():
    for exc in (
        BiosigIOError,
        UnsupportedFormatError,
        FileReadError,
        NotContinuousRecordingError,
        CorruptFileError,
        MixedSamplingRateError,
    ):
        assert exc.code in REASONS and REASONS[exc.code]


# -- real files ----------------------------------------------------------------


def test_unknown_extension_raises_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fh:
        with pytest.raises(UnsupportedFormatError):
            Recording.from_file(fh.name)


@pytest.mark.skipif(not _HAS_MNE, reason="MEG path needs the 'meg' extra (mne)")
def test_real_evoked_ave_fif_is_not_continuous():
    """A trial-averaged MNE evoked (``*-ave.fif``) is valid data but carries no
    continuous raw recording -> NotContinuousRecordingError (the on005261 case)."""
    import mne

    info = mne.create_info(["MEG0011", "MEG0021"], sfreq=100.0, ch_types="mag")
    evoked = mne.EvokedArray(np.zeros((2, 50), dtype=float), info, tmin=0.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub-01_task-x_ave.fif")  # MNE wants *-ave.fif
        evoked.save(path)
        with pytest.raises(NotContinuousRecordingError):
            Recording.from_file(path)


def test_truncated_edf_is_corrupt():
    """An EDF cut short of its declared length fails pyedflib's Filesize check ->
    CorruptFileError."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rec.edf")
        rate, n = 100.0, 1000
        w = pyedflib.EdfWriter(path, 1)
        try:
            w.setSignalHeaders(
                [
                    {
                        "label": "EEG1",
                        "dimension": "uV",
                        "sample_frequency": rate,
                        "physical_max": 100.0,
                        "physical_min": -100.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    }
                ]
            )
            w.writeSamples([np.zeros(n)])
        finally:
            w.close()
        # Lop off the back half of the data records -> declared > actual size.
        full = os.path.getsize(path)
        with open(path, "r+b") as fh:
            fh.truncate(full // 2)
        with pytest.raises(CorruptFileError):
            Recording.from_file(path)


def test_mixed_rate_edf_raises_typed_but_valueerror():
    """The mixed-rate policy error is now MixedSamplingRateError yet still a
    ValueError, so existing callers keep catching it."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "psg.edf")
        w = pyedflib.EdfWriter(path, 2)
        try:
            w.setSignalHeaders(
                [
                    {
                        "label": "EEG",
                        "dimension": "uV",
                        "sample_frequency": 200.0,
                        "physical_max": 100.0,
                        "physical_min": -100.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    },
                    {
                        "label": "SpO2",
                        "dimension": "%",
                        "sample_frequency": 12.5,
                        "physical_max": 100.0,
                        "physical_min": 0.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    },
                ]
            )
            w.writeSamples([np.zeros(200), np.zeros(13)])
        finally:
            w.close()
        with pytest.raises(MixedSamplingRateError):
            Recording.from_file(path, mixed_rate="error")
        with pytest.raises(ValueError):  # back-compat
            Recording.from_file(path, mixed_rate="error")
