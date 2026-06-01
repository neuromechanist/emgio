"""Tests for the Recording convenience accessors (added in 1.0.1). NO MOCKS."""

import numpy as np
import pytest

from biosigio import Recording


def _rec():
    rec = Recording()
    rec.add_channel("C1", np.zeros(1000), 500, "uV", "EEG")
    rec.add_channel("C2", np.zeros(1000), 500, "uV", "EEG")
    return rec


def test_get_n_channels():
    assert _rec().get_n_channels() == 2
    assert Recording().get_n_channels() == 0


def test_get_n_samples():
    assert _rec().get_n_samples() == 1000
    assert Recording().get_n_samples() == 0


def test_get_sampling_frequency_uniform():
    assert _rec().get_sampling_frequency() == 500.0


def test_get_sampling_frequency_empty_raises():
    with pytest.raises(ValueError, match="No channels"):
        Recording().get_sampling_frequency()


def test_get_sampling_frequency_mixed_raises():
    rec = Recording()
    rec.add_channel("A", np.zeros(100), 500, "uV", "EEG")
    rec.add_channel("B", np.zeros(100), 1000, "uV", "EMG")  # different declared rate
    with pytest.raises(ValueError, match="differing sampling"):
        rec.get_sampling_frequency()


def test_get_duration():
    rec = _rec()  # 1000 samples @ 500 Hz -> last index at 999/500 s
    assert rec.get_duration() == pytest.approx(999 / 500)
    assert Recording().get_duration() == 0.0


def test_has_metadata():
    rec = _rec()
    rec.set_metadata("subject", "S1")
    assert rec.has_metadata("subject") is True
    assert rec.has_metadata("missing") is False
