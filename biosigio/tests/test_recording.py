"""`Recording` is the core class; the deprecated `EMG` alias was removed in 1.0.0.

Before biosigIO the class was named `EMG`; it became the modality-agnostic
`Recording` (EEG/EMG/iEEG/MEG/...), with `EMG` kept as a deprecation alias through
0.5/0.6. The alias is now gone. NO MOCKS.
"""

import warnings

import numpy as np
import pytest

import biosigio
from biosigio import Recording


def test_recording_is_the_canonical_class():
    rec = Recording()
    rec.add_channel("C", np.zeros(100), 100, "uV", "EMG")
    assert type(rec).__name__ == "Recording"


def test_recording_access_emits_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert biosigio.Recording is Recording


def test_emg_alias_removed():
    # The deprecated EMG alias was dropped in 1.0.0; it is no longer accessible.
    with pytest.raises(AttributeError):
        _ = biosigio.EMG
    with pytest.raises(ImportError):
        from biosigio.core.emg import EMG  # noqa: F401


def test_unknown_attribute_still_raises_attributeerror():
    with pytest.raises(AttributeError):
        _ = biosigio.DoesNotExist
