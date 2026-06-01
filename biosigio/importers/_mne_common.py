"""Shared helpers for the MNE-backed importers (MEG, BrainVision).

MNE is an optional, heavy dependency, so it is imported lazily via
:func:`require_mne` (with a clear install hint) rather than at module import.
Both importers turn an MNE ``Raw`` into a :class:`~biosigio.core.emg.Recording` with the
same channel-type/unit mapping (:func:`raw_to_recording`); they differ only in how the
``Raw`` is read and how events are extracted.
"""

import warnings

import pandas as pd

from ..core.emg import Recording

# MNE channel type (raw.get_channel_types()) -> biosigio/BIDS channel type. Each MNE
# type maps to its own biosigio type so distinct sensor streams are preserved, not
# collapsed. MNE reports CTF axial gradiometers as 'mag' and lumps reference
# sensors as 'ref_meg' (a coil-type-precise split is a possible future refinement).
_MNE_TYPE_TO_biosigIO = {
    "mag": "MEGMAG",
    "grad": "MEGGRADPLANAR",
    "ref_meg": "MEGREFMAG",
    "eeg": "EEG",
    "seeg": "SEEG",
    "ecog": "ECOG",
    "dbs": "DBS",
    "eog": "EOG",
    "ecg": "ECG",
    "emg": "EMG",
    "stim": "TRIG",
    "resp": "RESP",
    "gsr": "GSR",
    "temperature": "TEMP",
    "bio": "MISC",
    "misc": "MISC",
    "syst": "MISC",
    "chpi": "MISC",
    "exci": "MISC",
    "ias": "MISC",
}

# FIFF physical-unit code (raw.info['chs'][i]['unit']) -> dimension string.
_FIFF_UNIT_TO_DIM = {107: "V", 112: "T", 201: "T/m"}


def require_mne():
    """Import MNE lazily, raising a clear install hint when it is absent."""
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "MNE-backed import (MEG, BrainVision) requires MNE-Python, an optional "
            "dependency. Install it with: uv sync --extra meg  (or, for an existing "
            "install, uv pip install 'biosigio[meg]')."
        ) from e
    return mne


def raw_to_recording(raw) -> Recording:
    """Build a Recording from an MNE ``Raw``: channels with mapped types and FIFF units.

    Does not read events (the two importers extract them differently) and does not
    set ``source_file`` (the caller has the path). High-channel-count recordings
    (e.g. MEG) fragment the DataFrame as channels are added one at a time; the
    expected pandas warning is suppressed and the frame de-fragmented once after
    (root-cause perf drift tracked in #66).
    """
    rec = Recording()
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data()  # (n_channels, n_samples) in SI units
    mne_types = raw.get_channel_types()
    rec.set_metadata("number_of_signals", len(raw.ch_names))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        for i, name in enumerate(raw.ch_names):
            channel_type = _MNE_TYPE_TO_biosigIO.get(mne_types[i], "OTHER")
            unit_code = int(raw.info["chs"][i]["unit"])
            rec.add_channel(
                label=name,
                data=data[i],
                sample_frequency=sfreq,
                physical_dimension=_FIFF_UNIT_TO_DIM.get(unit_code, "n/a"),
                channel_type=channel_type,
            )
    if rec.signals is not None:
        rec.signals = rec.signals.copy()  # de-fragment after many inserts
    return rec
