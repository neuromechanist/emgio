"""MEG importer backed by MNE-Python (.fif, CTF .ds, KIT/Yokogawa .con/.sqd/.kdf,
4D Neuroimaging/BTi).

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
    4D/BTi directory (no extension) -> ``read_raw_bti`` (see below)

4D/BTi is genuinely a MEG system (a whole-head magnetometer array), so it lives
here alongside CTF/KIT rather than in its own module -- unlike MEF3 (iEEG), which
is a different modality entirely and gets its own module (see
:mod:`biosigio.importers.mef3`). The one wrinkle: BIDS names the BTi recording
directory with NO extension (``sub-<label>[_ses-<label>]_task-<label>[_run-<index>]_meg/``,
containing the processed-data file -- conventionally ``c,rfDC`` -- plus ``config``
and, usually, ``hs_file``), so it cannot be recognized by extension the way
``.ds``/``.con``/``.sqd``/``.kdf`` are. Detection is therefore content-based
(:func:`_find_bti_pdf`): a directory qualifies only if it directly contains both
a ``c,rf*``-prefixed file (the actual PDF data) and a sibling ``config`` file --
checked non-recursively, so an unrelated nested ``config`` (every datalad-tracked
dataset has one at ``.datalad/config``) can never false-positive.
"""

import logging
import os

import numpy as np
import pandas as pd

from ..core.emg import Recording
from ..exceptions import UnsupportedFormatError, classify_read_error
from ._mne_common import raw_to_recording, require_mne
from .base import BaseImporter


def _find_bti_pdf(dirpath: str) -> str | None:
    """Return the path to a 4D/BTi processed-data file directly inside ``dirpath``,
    or None if this doesn't look like a BTi recording directory.

    Looks for a top-level entry whose name starts with ``c,rf`` (the conventional
    PDF basename is ``c,rfDC``; filtered copies such as ``c,rfDC,fn50,o`` also
    match) alongside a sibling ``config`` file. Both checks are non-recursive and
    look only at ``dirpath`` itself -- never a subdirectory -- so a dataset's
    ``.datalad/config`` (present in almost every datalad-tracked repo) can never
    be mistaken for the BTi sidecar.
    """
    try:
        entries = os.listdir(dirpath)
    except OSError:
        return None
    if "config" not in entries or not os.path.isfile(os.path.join(dirpath, "config")):
        return None
    for name in entries:
        candidate = os.path.join(dirpath, name)
        if name.startswith("c,rf") and os.path.isfile(candidate):
            return candidate
    return None


class MEGImporter(BaseImporter):
    """Importer for MEG recordings via MNE-Python (.fif, CTF .ds, KIT .con/.sqd/.kdf,
    4D/BTi)."""

    def _read_bti(self, mne, dirpath: str, pdf_fname: str):
        """Read a 4D/BTi recording found at ``pdf_fname`` inside ``dirpath``.

        ``config_fname``/``head_shape_fname`` are passed as the plain BIDS basenames
        (``"config"``/``"hs_file"``); MNE resolves a non-absolute name against
        ``dirname(pdf_fname)`` internally, which is exactly this directory -- so no
        path-joining is needed here. ``hs_file`` is optional in BIDS (not every BTi
        recording ships digitized head-shape points), so it is passed as None rather
        than a guessed filename when absent; MNE raises on a missing file it was
        never told to skip.
        """
        hs_fname = os.path.join(dirpath, "hs_file")
        head_shape_fname = "hs_file" if os.path.isfile(hs_fname) else None
        return mne.io.read_raw_bti(
            pdf_fname,
            config_fname="config",
            head_shape_fname=head_shape_fname,
            preload=True,
            verbose="ERROR",
        )

    def _read_raw(self, mne, filepath: str):
        """Read the recording into an MNE Raw object, dispatching by extension
        (or, for 4D/BTi, by directory content -- see :func:`_find_bti_pdf`).

        ``.ds`` is a CTF directory; ``.con``/``.sqd``/``.kdf`` are KIT/Yokogawa
        single files (markers/headshape sidecars are optional and not needed for
        the signal copy); an extension-less directory is checked for a BTi PDF;
        everything else is treated as Neuromag ``.fif``.
        """
        stripped = filepath.rstrip("/\\")
        ext = os.path.splitext(stripped)[1].lower()
        if ext == ".ds":
            return mne.io.read_raw_ctf(filepath, preload=True, verbose="ERROR")
        if ext in (".con", ".sqd", ".kdf"):
            return mne.io.read_raw_kit(filepath, preload=True, verbose="ERROR")
        if ext == "" and os.path.isdir(stripped):
            pdf_fname = _find_bti_pdf(stripped)
            if pdf_fname is not None:
                return self._read_bti(mne, stripped, pdf_fname)
            raise UnsupportedFormatError(
                f"{filepath!r} has no file extension and does not look like a "
                "4D/BTi recording directory (expected a 'c,rf*' processed-data "
                "file alongside a 'config' file). MEG import supports .fif, CTF "
                ".ds, KIT .con/.sqd/.kdf, and 4D/BTi directories."
            )
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
            filepath: Path to a ``.fif`` file, a CTF ``.ds`` directory, a KIT/
                Yokogawa ``.con``/``.sqd``/``.kdf`` file, or a 4D/BTi recording
                directory (no extension; detected by content, see
                :func:`_find_bti_pdf`).

        Returns:
            Recording: channels carry their MEG/EEG/stim type and physical unit; stim
            triggers are read into ``Recording.events``.
        """
        mne = require_mne()
        try:
            raw = self._read_raw(mne, filepath)
        except Exception as e:
            # Classify into a typed biosigIO error (not_continuous for an
            # evoked/epoched derivative, corrupt_or_truncated for a bad/incomplete
            # file, ...) so callers can surface a specific reason. The Pyright
            # import-resolution noise here is the editor lacking the venv; the
            # module exists.
            raise classify_read_error(e, filepath) from e

        rec = raw_to_recording(raw)
        rec.set_metadata("source_file", filepath)

        events = self._read_events(mne, raw, float(raw.info["sfreq"]))
        if not events.empty:
            rec.events = events

        return rec
