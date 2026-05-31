"""The core class is `Recording`; `EMG` is a deprecated backward-compat alias.

When the package handled only EMG the class was named `EMG`; it is now a
modality-agnostic biosignal recording (EEG/EMG/iEEG/MEG/...) named `Recording`.
`EMG` still resolves to `Recording` but now emits a ``DeprecationWarning`` and
will be removed in biosigio 1.0.0. NO MOCKS.
"""

import warnings

import numpy as np
import pytest

import emgio
from emgio import Recording


def test_emg_alias_resolves_to_recording_with_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="EMG.*deprecated"):
        cls = emgio.EMG
    assert cls is Recording


def test_emg_from_core_module_also_warns():
    with pytest.warns(DeprecationWarning, match="EMG.*deprecated"):
        from emgio.core.emg import EMG
    assert EMG is Recording


def test_emg_alias_is_still_constructible():
    with pytest.warns(DeprecationWarning):
        cls = emgio.EMG
    rec = cls()
    rec.add_channel("C", np.zeros(100), 100, "uV", "EMG")
    assert isinstance(rec, Recording)


def test_recording_is_the_canonical_class_name():
    rec = Recording()
    assert type(rec).__name__ == "Recording"


def test_recording_access_emits_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert emgio.Recording is Recording


def test_unknown_attribute_still_raises_attributeerror():
    with pytest.raises(AttributeError):
        _ = emgio.DoesNotExist
