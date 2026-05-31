"""BrainVision importer backed by MNE-Python (.vhdr + .vmrk + .eeg).

Many OpenNeuro BIDS EEG/iEEG datasets ship the BrainVision triplet rather than
EDF or EEGLAB .set. MNE reads it natively (only writing needs pybv), and is an
optional dependency imported lazily with a clear install hint. Channel types and
units come from the shared :mod:`_mne_common` mapping; ``.vmrk`` markers (which
MNE exposes as annotations) are read into ``Recording.events``.
"""

import pandas as pd

from ..core.emg import Recording
from ._mne_common import raw_to_emg, require_mne
from .base import BaseImporter


class BrainVisionImporter(BaseImporter):
    """Importer for BrainVision recordings via MNE-Python (.vhdr)."""

    def _read_events(self, raw) -> pd.DataFrame:
        """Read .vmrk markers (MNE annotations) into an events DataFrame."""
        annotations = raw.annotations
        # strict=True: the three arrays come from one Annotations object and are
        # co-length by MNE's contract; a mismatch is a bug we want surfaced, not
        # silently truncated.
        rows = [
            (float(onset), float(duration), str(description))
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description, strict=True
            )
        ]
        evt = pd.DataFrame(rows, columns=["onset", "duration", "description"])
        if not evt.empty:
            evt = evt.sort_values("onset").reset_index(drop=True)
            evt["onset"] = evt["onset"].astype("float64")
            evt["duration"] = evt["duration"].astype("float64")
        return evt

    def load(self, filepath: str) -> Recording:
        """Load a BrainVision recording (pass the ``.vhdr`` header path).

        Args:
            filepath: Path to the BrainVision ``.vhdr`` header (its ``.vmrk`` and
                ``.eeg`` siblings are resolved by MNE).

        Returns:
            Recording: channels carry their type and physical unit; ``.vmrk`` markers
            are read into ``Recording.events``.
        """
        mne = require_mne()
        try:
            raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose="ERROR")
        except Exception as e:
            raise ValueError(f"Error reading BrainVision file {filepath}: {e}") from e

        emg = raw_to_emg(raw)
        emg.set_metadata("source_file", filepath)

        events = self._read_events(raw)
        if not events.empty:
            emg.events = events

        return emg
