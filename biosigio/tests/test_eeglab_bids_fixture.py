"""Regression test: the real EEGLAB BIDS EEG fixture imports cleanly.

Guards the EEGLAB importer fixes:
- chanlocs X/Y/Z were read with ``float(field_value[0])`` on ``(1, 1)``-shaped
  scipy arrays, which crashed import of any real ``.set`` with electrode coords;
- the time index was built from EEGLAB's millisecond ``times`` field divided by
  the sampling rate, which mis-scaled it.

Uses the real CC0 fixture under ``examples/bids/eeg`` (NO MOCKS).
"""

import pathlib

import pytest

from biosigio import Recording

# Anchor to the repo root (this file is at biosigio/tests/) so the test does not
# silently skip when pytest runs from a different working directory.
EEG_SET = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"
)


@pytest.mark.skipif(not EEG_SET.exists(), reason="EEG BIDS fixture missing")
def test_eeglab_bids_eeg_fixture_imports():
    rec = Recording.from_file(str(EEG_SET), importer="eeglab")

    # All 64 channels imported with their real 10-10 montage labels.
    assert len(rec.channels) == 64
    assert "FP1" in rec.signals.columns

    col = rec.signals.columns[0]
    assert rec.channels[col]["sample_frequency"] == 250

    # 60 s at 250 Hz.
    n = len(rec.signals[col])
    assert n == 15000

    # Time index is seconds derived from the sample count, not the old
    # milliseconds/srate mis-scaling (which gave ~240 s for a 60 s recording).
    assert rec.signals.index[-1] == pytest.approx((n - 1) / 250, rel=1e-6)

    # Events were parsed from the .set event struct into the events table.
    assert not rec.events.empty and len(rec.events) == 3
