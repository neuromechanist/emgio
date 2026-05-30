"""Integration tests for the explicit-modality model (issue #46).

Verifies that channel_type is required (no silent EMG default), that modality is
derived/validated, the new selectors/mutator work, and that real EEG data no
longer creeps to EMG on import. NO MOCKS.
"""

import pathlib

import numpy as np
import pytest

from emgio import EMG

EEG_SET = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"
)


def test_add_channel_requires_channel_type():
    emg = EMG()
    with pytest.raises(TypeError):
        emg.add_channel("C", np.zeros(10), 100, "uV")  # channel_type omitted


def test_add_channel_validates_and_infers_modality():
    emg = EMG()
    emg.add_channel("E1", np.zeros(10), 100, "uV", "eeg")  # case-insensitive
    assert emg.channels["E1"]["channel_type"] == "EEG"
    assert emg.channels["E1"]["modality"] == "EEG"

    emg.add_channel("S1", np.zeros(10), 100, "uV", "ECOG")
    assert emg.channels["S1"]["modality"] == "IEEG"

    with pytest.raises(ValueError):
        emg.add_channel("X", np.zeros(10), 100, "uV", "BOGUS")


def test_set_channel_and_modality_selectors():
    emg = EMG()
    emg.add_channel("a", np.zeros(10), 100, "uV", "EEG")
    emg.add_channel("b", np.zeros(10), 100, "mV", "EMG")

    assert set(emg.get_modalities()) == {"EEG", "EMG"}
    assert emg.get_channels_by_modality("EEG") == ["a"]
    assert emg.select_channels(modality="EMG").signals.columns.tolist() == ["b"]

    emg.set_channel("b", channel_type="ECG")
    assert emg.channels["b"]["channel_type"] == "ECG"
    assert emg.channels["b"]["modality"] == "MISC"  # re-derived from new type


@pytest.mark.skipif(not EEG_SET.exists(), reason="EEG BIDS fixture missing")
def test_real_eeg_does_not_creep_to_emg():
    emg = EMG.from_file(str(EEG_SET), importer="eeglab")
    types = {info["channel_type"] for info in emg.channels.values()}
    # The headline bug: EEG data must not be silently relabelled EMG.
    assert "EMG" not in types
    # Every channel carries a modality after the explicit-modality migration.
    assert all("modality" in info for info in emg.channels.values())
