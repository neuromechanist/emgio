"""One-minute EDF+C and BDF+C exemplars, read back by biosigio and by pyedflib.

``examples/one_minute/{edf,bdf}/one_minute.*`` hold the first 8 channels of the repository's
EEG fixture (60 s at 250 Hz, three events), written by pyedflib as continuous EDF+ and BDF+
(provenance, license and quantization are in the ``NOTICE.md`` beside each file).
The tests read the real files; nothing is mocked and no stand-in data is synthesized.

* ``test_reads_exactly_like_pyedflib``: biosigio's samples, labels, rates and events are
  identical to what pyedflib returns for the same file.
* ``test_quantization_stays_within_one_digital_step``: the files really are the fixture's
  data, to within the documented quantization.
"""

import pathlib

import numpy as np
import pyedflib
import pytest

from biosigio import Recording

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "examples/one_minute"
FIXTURE = REPO / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"

SFREQ = 250.0
N_SAMPLES = 15000  # 60 s at 250 Hz
CHANNELS = ["FP1", "AF7", "FPZ", "FP2", "AF8", "F9", "F7", "F5"]
EVENTS = [(0.5, "eyes_open_start"), (20.0, "fixation_check"), (40.0, "rest_marker")]

# (path, the "reserved" header field that marks a continuous file, the largest quantization
# error in microvolts: one digital step of the widest channel, from NOTICE.md).
FILES = [
    pytest.param(DATA_DIR / "edf" / "one_minute.edf", b"EDF+C", 0.25, id="edf"),
    pytest.param(DATA_DIR / "bdf" / "one_minute.bdf", b"BDF+C", 0.001, id="bdf"),
]


@pytest.mark.parametrize(("path", "continuous", "max_error_uv"), FILES)
def test_reads_exactly_like_pyedflib(path, continuous, max_error_uv):
    # The file is a continuous EDF+/BDF+, as documented.
    assert path.read_bytes()[192 : 192 + len(continuous)] == continuous

    rec = Recording.from_file(str(path))
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = [header["label"] for header in reader.getSignalHeaders()]
        rates = list(reader.getSampleFrequencies())
        samples = [reader.readSignal(i) for i in range(reader.signals_in_file)]
        onsets, durations, descriptions = reader.readAnnotations()
    finally:
        reader.close()

    # pyedflib's own read, pinned, so that "identical to pyedflib" cannot hide an empty file.
    assert labels == CHANNELS
    assert rates == [SFREQ] * len(CHANNELS)
    assert [len(s) for s in samples] == [N_SAMPLES] * len(CHANNELS)

    assert list(rec.signals.columns) == labels
    assert {info["sample_frequency"] for info in rec.channels.values()} == {SFREQ}
    assert {info["physical_dimension"] for info in rec.channels.values()} == {"uV"}
    assert "edf_tolerant_read" not in rec.metadata  # a compliant file, read by pyedflib itself
    for i, label in enumerate(labels):
        np.testing.assert_array_equal(rec.signals[label].to_numpy(), samples[i])

    np.testing.assert_array_equal(rec.events["onset"].to_numpy(), onsets)
    np.testing.assert_array_equal(rec.events["duration"].to_numpy(), durations)
    assert list(rec.events["description"]) == list(descriptions)
    assert list(zip(rec.events["onset"], rec.events["description"], strict=True)) == EVENTS


@pytest.mark.parametrize(("path", "continuous", "max_error_uv"), FILES)
def test_quantization_stays_within_one_digital_step(path, continuous, max_error_uv):
    """The exemplar is the fixture's first 8 channels, quantized to 16 or 24 bits."""
    fixture = Recording.from_file(str(FIXTURE), importer="eeglab", bids_channels="off")
    rec = Recording.from_file(str(path))
    assert list(fixture.signals.columns)[: len(CHANNELS)] == CHANNELS
    assert fixture.channels["FP1"]["physical_dimension"] == "uV"

    expected = fixture.signals[CHANNELS].to_numpy(dtype=np.float64)
    got = rec.signals[CHANNELS].to_numpy()
    assert got.shape == expected.shape == (N_SAMPLES, len(CHANNELS))
    error = np.abs(got - expected)
    assert error.max() < max_error_uv
    # Not an exact copy: a 16 or 24 bit grid cannot hold a 32-bit float, so a zero error
    # would mean the quantization the NOTICE documents never happened.
    assert error.max() > 0
