"""Tests for the BIDS channel-type / modality vocabulary module.

Pure-function tests, no fixtures or external data required (NO MOCKS).
"""

import pytest

from biosigio.core.modality import (
    VALID_CHANNEL_TYPES,
    VALID_MODALITIES,
    infer_modality_from_channel_type,
    validate_channel_type,
    validate_modality,
)

# Expected modality for representative channel types.
EXPECTED_MODALITY = {
    "EEG": "EEG",
    "EMG": "EMG",
    "SEEG": "IEEG",
    "ECOG": "IEEG",
    "DBS": "IEEG",
    "MEGMAG": "MEG",
    "MEGGRADAXIAL": "MEG",
    "MEGGRADPLANAR": "MEG",
    "MEGREFMAG": "MEG",
    "ECG": "MISC",
    "EKG": "MISC",
    "EOG": "MISC",
    "ACC": "MISC",
    "GYRO": "MISC",
    "TRIG": "MISC",
    "REF": "MISC",
    "MISC": "MISC",
    "OTHER": "MISC",
    "PPG": "MISC",
}


@pytest.mark.parametrize("channel_type", sorted(VALID_CHANNEL_TYPES))
def test_validate_channel_type_accepts_canonical(channel_type):
    """Every canonical channel type validates to itself."""
    assert validate_channel_type(channel_type) == channel_type


@pytest.mark.parametrize("raw,expected", [("eeg", "EEG"), ("Emg", "EMG"), ("  ecg  ", "ECG")])
def test_validate_channel_type_normalizes(raw, expected):
    """Case and surrounding whitespace are normalized."""
    assert validate_channel_type(raw) == expected


@pytest.mark.parametrize("bad", ["BOGUS", "muscle", "eeg2", "channel"])
def test_validate_channel_type_rejects_unknown(bad):
    """Unknown channel types raise ValueError listing the valid set."""
    with pytest.raises(ValueError, match="Valid channel types"):
        validate_channel_type(bad)


@pytest.mark.parametrize("na", ["n/a", "N/A", "na", "", "  "])
def test_validate_channel_type_rejects_na(na):
    """'n/a' / empty is rejected; OTHER/MISC must be used for unknown types."""
    with pytest.raises(ValueError):
        validate_channel_type(na)


@pytest.mark.parametrize("channel_type,modality", sorted(EXPECTED_MODALITY.items()))
def test_infer_modality(channel_type, modality):
    assert infer_modality_from_channel_type(channel_type) == modality


@pytest.mark.parametrize("channel_type", sorted(VALID_CHANNEL_TYPES))
def test_infer_modality_is_always_valid(channel_type):
    """Inference returns a member of VALID_MODALITIES for every valid type."""
    assert infer_modality_from_channel_type(channel_type) in VALID_MODALITIES


def test_infer_modality_rejects_unknown():
    with pytest.raises(ValueError):
        infer_modality_from_channel_type("BOGUS")


@pytest.mark.parametrize("raw,expected", [("eeg", "EEG"), ("Misc", "MISC"), ("ieeg", "IEEG")])
def test_validate_modality_normalizes(raw, expected):
    assert validate_modality(raw) == expected


@pytest.mark.parametrize("bad", ["BOGUS", "ecog", "n/a", ""])
def test_validate_modality_rejects_unknown(bad):
    with pytest.raises(ValueError, match="Valid modalities"):
        validate_modality(bad)
