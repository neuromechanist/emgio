"""Tests for the MNE-backed BrainVision importer (issue #54).

Uses a small CC0 BrainVision fixture (8 EEG channels, 250 Hz, 5 s, 3 markers)
derived from the real EEG .set fixture (see examples/brainvision/README.md).
Verifies channel typing/units, marker -> event parsing, and an EDF round-trip.
NO MOCKS. Skips cleanly if the optional ``meg`` extra (mne) is not installed.
"""

import pathlib

import numpy as np
import pytest

from biosigio import Recording

pytest.importorskip("mne", reason="BrainVision import requires the optional 'meg' extra (mne)")

_REPO = pathlib.Path(__file__).resolve().parents[2]
VHDR = _REPO / "examples/brainvision/sub-01_task-rest_eeg.vhdr"

pytestmark = pytest.mark.skipif(not VHDR.exists(), reason="BrainVision fixture missing")


@pytest.fixture(scope="module")
def bv_emg():
    return Recording.from_file(str(VHDR))


def test_brainvision_channels_and_units(bv_emg):
    assert list(bv_emg.signals.columns) == ["FP1", "AF7", "FPZ", "FP2", "AF8", "F9", "F7", "F5"]
    assert {i["channel_type"] for i in bv_emg.channels.values()} == {"EEG"}
    assert {i["modality"] for i in bv_emg.channels.values()} == {"EEG"}
    assert {i["physical_dimension"] for i in bv_emg.channels.values()} == {"V"}  # MNE SI units
    assert {i["sample_frequency"] for i in bv_emg.channels.values()} == {250.0}
    assert bv_emg.signals.shape == (1250, 8)  # 5 s @ 250 Hz
    # No modality creep: an EEG recording must not invent EMG channels.
    assert "EMG" not in {i["channel_type"] for i in bv_emg.channels.values()}


def test_brainvision_markers_become_events(bv_emg):
    """.vmrk markers (MNE annotations) are read into events, sorted, in seconds."""
    assert bv_emg.events is not None and len(bv_emg.events) == 3
    assert np.allclose(bv_emg.events["onset"].to_numpy(), [1.0, 2.5, 4.0])
    assert bv_emg.events["onset"].dtype == np.float64
    assert all(bv_emg.events["description"].str.contains("Stimulus"))


def test_brainvision_no_markers_gives_empty_events():
    """A recording with no .vmrk markers must not crash; events stay empty."""
    import mne

    from biosigio.importers.brainvision import BrainVisionImporter

    raw = mne.io.read_raw_brainvision(str(VHDR), preload=True, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset=[], duration=[], description=[]))
    events = BrainVisionImporter()._read_events(raw)
    assert events.empty


def test_brainvision_roundtrip_through_edf(bv_emg, tmp_path):
    """Channels survive an EDF/BDF export + reimport (r > 0.99)."""
    out = tmp_path / "bv.edf"
    bv_emg.to_edf(str(out), format="bdf", bypass_analysis=True)
    written = out if out.exists() else out.with_suffix(".bdf")
    reloaded = Recording.from_file(str(written), bids_channels="off")
    assert len(reloaded.channels) == len(bv_emg.channels)
    for ch in bv_emg.signals.columns:
        original = bv_emg.signals[ch].values.astype(float)
        roundtripped = reloaded.signals[ch].values[: len(original)].astype(float)
        if np.std(original) == 0:
            continue
        r = float(np.corrcoef(original, roundtripped)[0, 1])
        assert r > 0.99, f"{ch}: round-trip correlation {r}"
