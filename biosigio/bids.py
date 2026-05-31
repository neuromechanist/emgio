"""Brain Imaging Data Structure (BIDS) sidecar helpers.

When a data file follows the BIDS layout, the authoritative per-channel
metadata lives in a sibling ``*_channels.tsv`` (the ``type`` and ``units``
columns), not in the data file's own headers. These helpers locate that sidecar
and apply it to an :class:`~biosigio.core.emg.Recording` object so imported channels get
their real BIDS types (e.g. ``SEEG``) instead of header/label guesses.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .core.emg import Recording


def _find_sidecar(data_filepath: str, kind: str) -> str | None:
    """Return the sibling BIDS ``_<kind>.tsv`` path for a data file, or None.

    BIDS names a sidecar with the data file's entities but a different suffix,
    e.g. ``sub-01_task-rest_ieeg.edf`` -> ``sub-01_task-rest_channels.tsv``.

    Args:
        data_filepath: Path to a (BIDS) data file.
        kind: Sidecar kind, e.g. ``"channels"`` or ``"events"``.

    Returns:
        The sidecar path if it exists next to the data file, else ``None``.
    """
    directory, filename = os.path.split(data_filepath)
    stem = os.path.splitext(filename)[0]
    candidates = []
    # BIDS-correct: drop the trailing _<suffix> entity (e.g. _ieeg, _eeg, _meg).
    if "_" in stem:
        candidates.append(f"{stem.rsplit('_', 1)[0]}_{kind}.tsv")
    # Fallback: the full-stem form biosigio's own EDF exporter currently writes,
    # so a to_edf -> from_file round-trip also finds the sidecar.
    candidates.append(f"{stem}_{kind}.tsv")
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def find_channels_tsv(data_filepath: str) -> str | None:
    """Return the sibling BIDS ``_channels.tsv`` path for a data file, or None."""
    return _find_sidecar(data_filepath, "channels")


def find_events_tsv(data_filepath: str) -> str | None:
    """Return the sibling BIDS ``_events.tsv`` path for a data file, or None."""
    return _find_sidecar(data_filepath, "events")


def apply_channels_tsv(emg: Recording, channels_tsv_path: str) -> int:
    """Override per-channel ``type``/``units`` in ``emg`` from a ``_channels.tsv``.

    Rows are matched to channels by the ``name`` column; ``n/a`` and empty
    values are skipped (the importer-inferred value is kept). An unrecognized
    ``type`` is warned about and skipped rather than raising.

    Args:
        emg: The Recording object to update in place.
        channels_tsv_path: Path to the BIDS ``_channels.tsv``.

    Returns:
        The number of channels updated.
    """
    df = pd.read_csv(channels_tsv_path, sep="\t", dtype=str, keep_default_na=False)
    if "name" not in df.columns:
        logging.warning("channels.tsv has no 'name' column: %s", channels_tsv_path)
        return 0

    updated = 0
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        if name not in emg.channels:
            continue
        kwargs: dict[str, str] = {}
        ctype = str(row.get("type", "")).strip()
        units = str(row.get("units", "")).strip()
        if ctype and ctype.lower() != "n/a":
            kwargs["channel_type"] = ctype
        if units and units.lower() != "n/a":
            kwargs["physical_dimension"] = units
        if not kwargs:
            continue
        try:
            emg.set_channel(name, **kwargs)
            updated += 1
        except ValueError:
            logging.warning(
                "channels.tsv type %r for channel %r is not a known channel type; "
                "keeping the importer-inferred type.",
                ctype,
                name,
            )
    return updated
