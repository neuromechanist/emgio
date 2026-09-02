"""MEF3 importer backed by MNE-Python (``.mefd``, Multiscale Electrophysiology
Format v3).

MEF3 is an iEEG (stereo-EEG / intracranial) archive format; a ``.mefd`` session
is a DIRECTORY, not a single file (``<name>.mefd/<CHANNEL>.timd/<CHANNEL>-000000
.segd/`` holding ``.tdat``/``.tidx``/``.tmet`` per channel segment -- a real
session can hold well over a hundred ``.timd`` channel directories).

Kept in its own module rather than folded into :mod:`biosigio.importers.meg`
because it is a different MODALITY (iEEG, not MEG) with its own, narrower
dependency footprint: ``mne.io.read_raw_mef`` was only added in MNE 1.12 and
additionally requires the optional ``pymef`` package (a wrapper around the MEF3
C reference library MNE's reader delegates to), both stricter than the
``mne>=1.6`` the rest of the MNE-backed importers need. Folding MEF3 into
``meg.py`` would force that newer MNE floor onto every MEG user; see
:func:`require_mne_mef` for the runtime gate that keeps it opt-in instead, and
the ``mef3`` extra in ``pyproject.toml`` for the install path.

Channels default to the MNE ``seeg`` type (mapped to biosigIO ``SEEG`` by the
shared :mod:`_mne_common` table); reassign with ``raw.set_channel_types()`` if a
recording is actually ECoG/DBS. MEF3's internal records/table-of-contents gaps
surface as MNE annotations, which are read into ``Recording.events`` the same
way BrainVision's ``.vmrk`` markers are.
"""

import re

import pandas as pd

from ..core.emg import Recording
from ..exceptions import classify_read_error, is_resource_exhaustion
from ._mne_common import raw_to_recording, require_mne
from .base import BaseImporter

# mne.io.read_raw_mef was added in MNE 1.12; older MNE has no such attribute at
# all, so calling it unguarded would fail with a confusing AttributeError deep
# inside mne.io rather than a clear, actionable message pointing at the fix.
_MIN_MNE_VERSION = (1, 12)


def _mne_version_tuple(raw_version: str) -> tuple[int, int]:
    """Parse the leading ``major.minor`` out of an MNE version string.

    Handles plain releases (``"1.12.1"``) and pre-release/dev suffixes
    (``"1.13.0.dev0"``) alike; only major/minor are compared since MEF3 support
    is an all-or-nothing feature introduced in a specific minor release.
    """
    match = re.match(r"(\d+)\.(\d+)", raw_version)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def require_mne_mef():
    """Import MNE lazily and enforce the version/dependency floor MEF3 needs.

    MEF3 needs strictly more than the rest of the MNE-backed importers: MNE
    itself must be >=1.12 (``mne.io.read_raw_mef`` is new in that release), and
    the optional ``pymef`` package must be installed (MNE's reader delegates all
    actual MEF3 parsing to it). Both are checked here, up front, so the failure
    is one clear, actionable biosigIO error naming the exact version and the
    install command -- not a bare ``AttributeError`` (missing MNE version) or
    MNE's own ``pymef``-not-found message (which suggests ``pip``/``conda``,
    neither of which this project uses).
    """
    mne = require_mne()
    have = _mne_version_tuple(mne.__version__)
    if have < _MIN_MNE_VERSION:
        raise ImportError(
            f"MEF3 import (.mefd) requires mne>=1.12 (mne.io.read_raw_mef was added "
            f"in MNE 1.12); found mne=={mne.__version__}. Install a compatible MNE "
            "with: uv sync --extra mef3  (or, for an existing install, "
            "uv pip install 'biosigio[mef3]')."
        )
    try:
        import pymef  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "MEF3 import (.mefd) requires pymef, the MEF3 C-library wrapper MNE's "
            "reader delegates to. Install it with: uv sync --extra mef3  (or, for "
            "an existing install, uv pip install 'biosigio[mef3]')."
        ) from e
    return mne


class MEF3Importer(BaseImporter):
    """Importer for MEF3 recordings via MNE-Python (``.mefd``; requires the
    ``mef3`` extra)."""

    def _read_events(self, raw) -> pd.DataFrame:
        """Read MEF3 records / TOC gaps (exposed by MNE as annotations) into an
        events DataFrame, the same way :class:`BrainVisionImporter` reads ``.vmrk``
        markers."""
        annotations = raw.annotations
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

    def load(self, filepath: str, password: str = "") -> Recording:
        """Load a MEF3 recording into a Recording object.

        Args:
            filepath: Path to a ``.mefd`` session directory.
            password: Password for an encrypted MEF3 session. Default ``""`` for
                unencrypted data (the common case).

        Returns:
            Recording: channels default to ``SEEG`` type (reassign with
            ``raw.set_channel_types()`` beforehand via a custom pipeline if a
            recording is actually ECoG/DBS); MEF3 records/TOC gaps are read into
            ``Recording.events``.
        """
        mne = require_mne_mef()
        try:
            raw = mne.io.read_raw_mef(filepath, password=password, preload=True, verbose="ERROR")
        except Exception as e:
            # Resource exhaustion is a host condition, not a file problem --
            # propagate unchanged rather than reclassifying it as a permanent
            # read failure (see biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            raise classify_read_error(e, filepath) from e

        rec = raw_to_recording(raw)
        rec.set_metadata("source_file", filepath)

        events = self._read_events(raw)
        if not events.empty:
            rec.events = events

        return rec
