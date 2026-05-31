"""Tests for BIDS channels.tsv-driven per-channel typing (issue #57).

The authoritative per-channel type/units in a BIDS dataset live in the sibling
``_channels.tsv``, not in the data file's headers. On import emgio applies it so
e.g. iEEG channels become ``SEEG`` instead of the EDF importer's ``OTHER``.
NO MOCKS: uses the real iEEG BIDS fixture.
"""

import pathlib

import pytest

from emgio import EMG
from emgio.bids import find_channels_tsv

_REPO = pathlib.Path(__file__).resolve().parents[2]
IEEG = (
    _REPO
    / "examples/bids/ieeg/sub-01/ses-postimp/ieeg/sub-01_ses-postimp_task-stim_run-08_ieeg.edf"
)


@pytest.mark.skipif(not IEEG.exists(), reason="iEEG BIDS fixture missing")
def test_channels_tsv_assigns_seeg_types():
    emg = EMG.from_file(str(IEEG))
    assert {info["channel_type"] for info in emg.channels.values()} == {"SEEG"}
    assert {info["modality"] for info in emg.channels.values()} == {"IEEG"}


@pytest.mark.skipif(not IEEG.exists(), reason="iEEG BIDS fixture missing")
def test_bids_channels_off_falls_back_to_header_inference():
    emg = EMG.from_file(str(IEEG), bids_channels="off")
    # Without the sidecar the EDF header cannot supply SEEG; assert the sidecar
    # was not consulted rather than pinning the exact inferred type.
    assert "SEEG" not in {info["channel_type"] for info in emg.channels.values()}


@pytest.mark.skipif(not IEEG.exists(), reason="iEEG BIDS fixture missing")
def test_find_channels_tsv_resolves_sibling():
    assert find_channels_tsv(str(IEEG)) is not None
    assert find_channels_tsv("/tmp/nonexistent_eeg.edf") is None
