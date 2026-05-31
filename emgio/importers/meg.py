"""MEG importer backed by MNE-Python (.fif and CTF .ds).

MNE is an optional dependency (it is heavy), so it is imported lazily with a
clear install hint rather than at module import. MEG recordings mix several
sensor types in one file (magnetometers, gradiometers, reference sensors)
alongside stim/EEG/EOG/ECG channels; the shared :mod:`_mne_common` mapping keeps
each type distinct (they are NOT collapsed into a single "MEG" type), and
stim-channel triggers are read into ``Recording.events``.
"""

import os

import numpy as np
import pandas as pd

from ..core.emg import Recording
from ._mne_common import raw_to_emg, require_mne
from .base import BaseImporter


class MEGImporter(BaseImporter):
    """Importer for MEG recordings via MNE-Python (.fif and CTF .ds)."""

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

    def load(self, filepath: str) -> Recording:
        """Load a MEG recording into a Recording object.

        Args:
            filepath: Path to a ``.fif`` file or a CTF ``.ds`` directory.

        Returns:
            Recording: channels carry their MEG/EEG/stim type and physical unit; stim
            triggers are read into ``Recording.events``.
        """
        mne = require_mne()
        try:
            raw = self._read_raw(mne, filepath)
        except Exception as e:
            raise ValueError(f"Error reading MEG file {filepath}: {e}") from e

        emg = raw_to_emg(raw)
        emg.set_metadata("source_file", filepath)

        events = self._read_events(mne, raw, float(raw.info["sfreq"]))
        if not events.empty:
            emg.events = events

        return emg
