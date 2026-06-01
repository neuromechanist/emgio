"""Integration tests for the explicit-modality model (issue #46).

Verifies that channel_type is required (no silent EMG default), that modality is
derived/validated, the new selectors/mutator work, and that real EEG data no
longer creeps to EMG on import. NO MOCKS.
"""

import pathlib

import numpy as np
import pytest

from biosigio import Recording
from biosigio.core.modality import VALID_CHANNEL_TYPES
from biosigio.importers.csv import CSVImporter

_REPO = pathlib.Path(__file__).resolve().parents[2]
EEG_SET = _REPO / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"
XDF_FILE = _REPO / "examples/multi_stream_test.xdf"


def test_add_channel_requires_channel_type():
    rec = Recording()
    with pytest.raises(TypeError):
        rec.add_channel("C", np.zeros(10), 100, "uV")  # channel_type omitted


def test_add_channel_validates_and_infers_modality():
    rec = Recording()
    rec.add_channel("E1", np.zeros(10), 100, "uV", "eeg")  # case-insensitive
    assert rec.channels["E1"]["channel_type"] == "EEG"
    assert rec.channels["E1"]["modality"] == "EEG"

    rec.add_channel("S1", np.zeros(10), 100, "uV", "ECOG")
    assert rec.channels["S1"]["modality"] == "IEEG"

    with pytest.raises(ValueError):
        rec.add_channel("X", np.zeros(10), 100, "uV", "BOGUS")


def test_set_channel_and_modality_selectors():
    rec = Recording()
    rec.add_channel("a", np.zeros(10), 100, "uV", "EEG")
    rec.add_channel("b", np.zeros(10), 100, "mV", "EMG")

    assert set(rec.get_modalities()) == {"EEG", "EMG"}
    assert rec.get_channels_by_modality("EEG") == ["a"]
    assert rec.select_channels(modality="EMG").signals.columns.tolist() == ["b"]

    rec.set_channel("b", channel_type="ECG")
    assert rec.channels["b"]["channel_type"] == "ECG"
    assert rec.channels["b"]["modality"] == "MISC"  # re-derived from new type


def test_csv_time_column_is_not_an_invalid_type():
    """A time-named column must classify as a valid type (not the bogus 'TIME')."""
    ct = CSVImporter()._infer_channel_type("timestamp")
    assert ct in VALID_CHANNEL_TYPES


@pytest.mark.skipif(not XDF_FILE.exists(), reason="XDF fixture missing")
def test_xdf_channels_carry_valid_type_and_modality():
    rec = Recording.from_file(str(XDF_FILE))
    assert len(rec.channels) > 0
    for info in rec.channels.values():
        # No raw/unvalidated LSL type strings leak through, and every channel
        # carries a modality (so the modality selectors work for XDF data).
        assert info["channel_type"] in VALID_CHANNEL_TYPES
        assert info.get("modality") is not None


@pytest.mark.skipif(not EEG_SET.exists(), reason="EEG BIDS fixture missing")
def test_real_eeg_does_not_creep_to_emg():
    rec = Recording.from_file(str(EEG_SET), importer="eeglab")
    types = {info["channel_type"] for info in rec.channels.values()}
    # The headline bug: EEG data must not be silently relabelled EMG.
    assert "EMG" not in types
    # Every channel carries a modality after the explicit-modality migration.
    assert all("modality" in info for info in rec.channels.values())
