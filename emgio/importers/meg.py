"""MEG importer backed by MNE-Python (.fif and CTF .ds).

MNE is an optional dependency (it is heavy), so it is imported lazily inside
:meth:`MEGImporter.load` with a clear install hint rather than at module import.

MEG recordings mix several sensor types in one file (magnetometers, gradiometers,
reference sensors) alongside stim/EEG/EOG/ECG channels. Each MNE channel type is
mapped to its own emgio/BIDS channel type so the distinctions are preserved
(they are NOT collapsed into a single "MEG" type), and stim-channel triggers are
read into ``EMG.events``.
"""

import os
import warnings

import numpy as np
import pandas as pd

from ..core.emg import EMG
from .base import BaseImporter

# MNE channel type (raw.get_channel_types()) -> emgio/BIDS channel type.
# Note: MNE reports CTF axial gradiometers as 'mag' and lumps all reference
# sensors as 'ref_meg'; a coil-type-precise split (MEGGRADAXIAL / MEGREFGRAD*)
# is a possible refinement, but the MNE channel type already preserves the
# mag / ref / stim / EEG / EOG / ECG distinctions this importer must keep.
_MNE_TYPE_TO_EMGIO = {
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


class MEGImporter(BaseImporter):
    """Importer for MEG recordings via MNE-Python (.fif and CTF .ds)."""

    @staticmethod
    def _require_mne():
        try:
            import mne
        except ImportError as e:
            raise ImportError(
                "MEG import requires MNE-Python, an optional dependency. Install it with: "
                "uv pip install 'emgio[meg]'  (or: pip install mne)."
            ) from e
        return mne

    def _read_raw(self, mne, filepath: str):
        """Read a .fif file or a CTF .ds directory into an MNE Raw object."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".ds":
            return mne.io.read_raw_ctf(filepath, preload=True, verbose="ERROR")
        return mne.io.read_raw_fif(filepath, preload=True, verbose="ERROR")

    def _read_events(self, mne, raw, sfreq: float) -> pd.DataFrame:
        """Read stim-channel triggers into an events DataFrame (onsets in seconds)."""
        try:
            events = mne.find_events(raw, verbose="ERROR")
        except (ValueError, RuntimeError):
            # No stim channel / no transitions: no events.
            events = np.empty((0, 3), dtype=int)
        first_samp = int(raw.first_samp)
        rows = [
            ((int(sample) - first_samp) / sfreq, 0.0, str(int(code)))
            for sample, _prev, code in events
        ]
        evt = pd.DataFrame(rows, columns=["onset", "duration", "description"])
        if not evt.empty:
            evt = evt.sort_values("onset").reset_index(drop=True)
            evt["onset"] = evt["onset"].astype("float64")
            evt["duration"] = evt["duration"].astype("float64")
        return evt

    def load(self, filepath: str) -> EMG:
        """Load a MEG recording into an EMG object.

        Args:
            filepath: Path to a ``.fif`` file or a CTF ``.ds`` directory.

        Returns:
            EMG: channels carry their MEG/EEG/stim type and physical unit; stim
            triggers are read into ``EMG.events``.
        """
        mne = self._require_mne()
        try:
            raw = self._read_raw(mne, filepath)
        except Exception as e:
            raise ValueError(f"Error reading MEG file {filepath}: {e}") from e

        emg = EMG()
        sfreq = float(raw.info["sfreq"])
        data = raw.get_data()  # (n_channels, n_samples) in SI units
        mne_types = raw.get_channel_types()

        emg.set_metadata("source_file", filepath)
        emg.set_metadata("number_of_signals", len(raw.ch_names))

        # MEG files routinely carry 300+ channels; adding them one at a time
        # fragments the DataFrame (an expected, benign pandas PerformanceWarning),
        # so suppress it here and de-fragment once after the bulk add.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            for i, name in enumerate(raw.ch_names):
                channel_type = _MNE_TYPE_TO_EMGIO.get(mne_types[i], "OTHER")
                unit_code = int(raw.info["chs"][i]["unit"])
                emg.add_channel(
                    label=name,
                    data=data[i],
                    sample_frequency=sfreq,
                    physical_dimension=_FIFF_UNIT_TO_DIM.get(unit_code, "n/a"),
                    channel_type=channel_type,
                )
        if emg.signals is not None:
            emg.signals = emg.signals.copy()  # de-fragment after many inserts

        events = self._read_events(mne, raw, sfreq)
        if not events.empty:
            emg.events = events

        return emg
