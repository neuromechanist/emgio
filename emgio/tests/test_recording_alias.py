"""The core class is `Recording`; `EMG` is a deprecated backward-compat alias.

When the package handled only EMG the class was named `EMG`; it is now a
modality-agnostic biosignal recording (EEG/EMG/iEEG/MEG/...) named `Recording`,
with `EMG` retained as an alias so existing code keeps working. NO MOCKS.
"""

import numpy as np

from emgio import EMG, Recording


def test_emg_is_alias_of_recording():
    assert EMG is Recording


def test_recording_is_the_canonical_class_name():
    rec = Recording()
    assert type(rec).__name__ == "Recording"


def test_alias_works_for_construction_and_isinstance():
    rec = Recording()
    rec.add_channel("C", np.zeros(100), 100, "uV", "EMG")
    via_alias = EMG()
    via_alias.add_channel("C", np.zeros(100), 100, "uV", "EMG")
    # Instances of one are instances of the other (same class).
    assert isinstance(rec, EMG)
    assert isinstance(via_alias, Recording)
