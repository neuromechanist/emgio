"""MEG importer backed by MNE-Python (.fif, CTF .ds, KIT/Yokogawa .con/.sqd/.kdf).

MNE is an optional dependency (it is heavy), so it is imported lazily with a
clear install hint rather than at module import. MEG recordings mix several
sensor types in one file (magnetometers, gradiometers, reference sensors)
alongside stim/EEG/EOG/ECG channels; the shared :mod:`_mne_common` mapping keeps
each type distinct (they are NOT collapsed into a single "MEG" type), and
stim-channel triggers are read into ``Recording.events``.

Formats and their MNE reader:
    ``.fif``                 -> ``read_raw_fif`` (Neuromag / Elekta / MEGIN)
    ``.ds`` (a directory)    -> ``read_raw_ctf`` (CTF / VSM)
    ``.con`` / ``.sqd`` / ``.kdf`` -> ``read_raw_kit`` (KIT / Yokogawa / RICOH)
"""

import logging
import os

import numpy as np
import pandas as pd

from ..core.emg import Recording
from ._mne_common import raw_to_recording, require_mne
from .base import BaseImporter


class MEGImporter(BaseImporter):
    """Importer for MEG recordings via MNE-Python (.fif, CTF .ds, KIT .con/.sqd/.kdf)."""

    def _read_raw(self, mne, filepath: str):
        """Read the recording into an MNE Raw object, dispatching by extension.

        ``.ds`` is a CTF directory; ``.con``/``.sqd``/``.kdf`` are KIT/Yokogawa
        single files (markers/headshape sidecars are optional and not needed for
        the signal copy); everything else is treated as Neuromag ``.fif``.
        """
        ext = os.path.splitext(filepath.rstrip("/\\"))[1].lower()
        if ext == ".ds":
            return mne.io.read_raw_ctf(filepath, preload=True, verbose="ERROR")
        if ext in (".con", ".sqd", ".kdf"):
            return mne.io.read_raw_kit(filepath, preload=True, verbose="ERROR")
        return mne.io.read_raw_fif(filepath, preload=True, verbose="ERROR")

    def _read_events(self, mne, raw, sfreq: float) -> pd.DataFrame:
        """Read stim-channel triggers into an events DataFrame (onsets in seconds)."""
        try:
            events = mne.find_events(raw, verbose="ERROR")
        except (ValueError, RuntimeError) as exc:
            # "No stim channels found" is the common, benign case (many MEG
            # recordings carry no triggers, or events come from a BIDS events.tsv
            # applied separately) -- stay quiet for that. Any OTHER failure
            # (malformed/ambiguous triggers) would silently drop real events, so
            # surface it rather than swallowing it.
            if "stim" not in str(exc).lower():
                logging.warning("MEG find_events failed; events left empty: %s", exc)
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
            filepath: Path to a ``.fif`` file, a CTF ``.ds`` directory, or a KIT/
                Yokogawa ``.con``/``.sqd``/``.kdf`` file.

        Returns:
            Recording: channels carry their MEG/EEG/stim type and physical unit; stim
            triggers are read into ``Recording.events``.
        """
        mne = require_mne()
        try:
            raw = self._read_raw(mne, filepath)
        except Exception as e:
            raise ValueError(f"Error reading MEG file {filepath}: {e}") from e

        rec = raw_to_recording(raw)
        rec.set_metadata("source_file", filepath)

        events = self._read_events(mne, raw, float(raw.info["sfreq"]))
        if not events.empty:
            rec.events = events

        return rec
