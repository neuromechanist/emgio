"""Tests for the MNE-backed MEG importer (issue #53).

Uses the real CTF fixture (ds002908: 305 channels, 100 Hz, 30 s, 58 trigger
events, Tesla-unit sensors). Verifies that the distinct MEG sensor types are
preserved (not collapsed), units are carried, stim triggers become events, and
the Tesla-magnitude signals survive an EDF/BDF round-trip. NO MOCKS.

Skips cleanly if the optional ``meg`` extra (mne) is not installed.
"""

import pathlib
from collections import Counter

import numpy as np
import pytest

from biosigio import Recording

pytest.importorskip("mne", reason="MEG import requires the optional 'meg' extra (mne)")

_REPO = pathlib.Path(__file__).resolve().parents[2]
MEG = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"

pytestmark = pytest.mark.skipif(not MEG.exists(), reason="MEG fixture missing")


@pytest.fixture(scope="module")
def meg_rec():
    return Recording.from_file(str(MEG))


def test_meg_channel_types_preserved(meg_rec):
    """Sensor types stay distinct (mag vs reference vs stim), not collapsed."""
    types = Counter(i["channel_type"] for i in meg_rec.channels.values())
    assert types["MEGMAG"] == 274
    assert types["MEGREFMAG"] == 29
    assert types["TRIG"] == 2
    assert sum(types.values()) == 305
    # No fabricated EMG on a MEG recording.
    assert "EMG" not in types


def test_meg_units_and_modalities(meg_rec):
    by_type = {i["channel_type"]: i for i in meg_rec.channels.values()}
    assert by_type["MEGMAG"]["physical_dimension"] == "T"  # Tesla magnetometers
    assert by_type["MEGREFMAG"]["physical_dimension"] == "T"
    assert by_type["TRIG"]["physical_dimension"] == "V"
    assert by_type["MEGMAG"]["modality"] == "MEG"
    assert by_type["TRIG"]["modality"] == "MISC"


def test_meg_sampling_and_shape(meg_rec):
    rates = {i["sample_frequency"] for i in meg_rec.channels.values()}
    assert rates == {100.0}
    assert meg_rec.signals.shape == (3000, 305)  # 30 s @ 100 Hz, 305 channels


def test_meg_stim_events(meg_rec):
    """Stim-channel triggers are read into events (onsets in seconds, sorted)."""
    assert meg_rec.events is not None and len(meg_rec.events) == 58
    onsets = meg_rec.events["onset"].to_numpy()
    assert (onsets[:-1] <= onsets[1:]).all()  # sorted
    assert onsets.min() >= 0.0 and onsets.max() <= 30.0  # within the recording
    assert meg_rec.events["onset"].dtype == np.float64
    # Descriptions are the stringified trigger codes.
    assert all(d.isdigit() for d in meg_rec.events["description"])


def test_meg_roundtrip_preserves_tesla_signals(meg_rec, tmp_path):
    """Tesla-magnitude MEG channels survive EDF/BDF export+reimport (r > 0.99).

    Magnetometer values are ~1e-12 T, exercising the exporter's small-magnitude
    physical-bound path; BDF (auto-selected for the dynamic range) must keep them.
    """
    out = tmp_path / "meg.edf"
    meg_rec.to_edf(str(out), format="bdf", bypass_analysis=True)
    written = out if out.exists() else out.with_suffix(".bdf")
    reloaded = Recording.from_file(str(written), bids_channels="off")
    assert len(reloaded.channels) == len(meg_rec.channels)

    # Check a handful of magnetometer channels on a 10 s window.
    mag = [c for c, i in meg_rec.channels.items() if i["channel_type"] == "MEGMAG"][:5]
    window = slice(0, 1000)  # 10 s @ 100 Hz
    for ch in mag:
        original = meg_rec.signals[ch].values[window].astype(float)
        roundtripped = reloaded.signals[ch].values[window].astype(float)
        if np.std(original) == 0:
            continue
        r = float(np.corrcoef(original, roundtripped)[0, 1])
        assert r > 0.99, f"{ch}: round-trip correlation {r}"


# -- Format dispatch (no fixture needed; just the extension -> importer mapping) --


def test_meg_extensions_dispatch_to_meg_importer():
    """FIF, CTF .ds, and KIT .con/.sqd/.kdf all route to the 'meg' importer."""
    for ext in (".fif", ".ds", ".con", ".sqd", ".kdf"):
        assert Recording._infer_importer(f"sub-01/meg/sub-01_task-x_meg{ext}") == "meg"


def test_ctf_ds_trailing_slash_dispatches_to_meg():
    """A CTF .ds passed as a directory path (trailing slash) still resolves."""
    assert Recording._infer_importer("sub-01/meg/sub-01_task-x_meg.ds/") == "meg"
