"""Brain Imaging Data Structure (BIDS) sidecar helpers.

When a data file follows the BIDS layout, the authoritative per-channel
metadata lives in a sibling ``*_channels.tsv`` (the ``type`` and ``units``
columns), not in the data file's own headers. These helpers locate that sidecar
and apply it to an :class:`~biosigio.core.emg.Recording` object so imported channels get
their real BIDS types (e.g. ``SEEG``) instead of header/label guesses.

The ``units`` column is a claim about the numbers, not just a label, so adopting
it rescales the samples (see :func:`_adopt_units`). Applying a sidecar therefore
leaves values and unit in agreement, which relabelling alone did not.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING

import pandas as pd

from .units import conversion_factor

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


def _keep_importer_unit(info: dict, label: str, current: str, declared: str, reason: str) -> bool:
    """Record the sidecar's unit without asserting it over contradicting values.

    The BIDS metadata is real and worth keeping, so it lands under ``bids_unit``
    rather than being dropped; what it must not do is overwrite a
    ``physical_dimension`` that correctly describes the samples.

    Args:
        info: The channel's metadata dict, mutated in place.
        label: Channel name, for the warning.
        current: The unit the samples are actually in.
        declared: The sidecar's ``units`` value.
        reason: Why the sidecar's unit was not adopted, for the warning.

    Returns:
        True, always: the channel record changed.
    """
    info["bids_unit"] = declared
    logging.warning(
        "channels.tsv declares units %r for channel %r, whose values are in %r and %s; "
        "keeping the importer's values and unit, and recording the declared unit "
        "as 'bids_unit'.",
        declared,
        label,
        current,
        reason,
    )
    return True


def _adopt_units(rec: Recording, label: str, declared: str) -> bool:
    """Move one channel onto the sidecar's unit, rescaling its samples to match.

    A unit label is a claim about the numbers next to it, so the two move
    together or not at all. The sidecar's ``units`` describes the values *as the
    data file stores them*, while ``physical_dimension`` describes the values
    biosigIO currently holds -- and those differ whenever the importer rescaled
    on the way in (every MNE-backed importer returns SI volts regardless of the
    file's own µV). Adopting the label without the conversion is what issue #122
    reports: volts relabelled as microvolts, wrong by 10^6.

    Three outcomes, and only the first one touches the samples:

    - **Convertible** (same quantity, e.g. ``V`` -> ``uV``): multiply the samples
      by the ratio and set the label. A ratio of exactly 1 (``uV`` -> ``µV``, a
      spelling difference) sets the label alone, which is not a semantic relabel.
    - **Already there** (labels equal): nothing at all, which is what makes
      repeated application idempotent -- the second pass finds the units already
      adopted and cannot double-convert.
    - **Not convertible** (unparsable on either side, different quantities, or no
      samples to scale): keep the importer's values *and* its label, and record
      the sidecar's claim under ``bids_unit`` so the BIDS metadata is preserved
      without being asserted over numbers that would contradict it.

    Args:
        rec: The Recording object to update in place.
        label: Channel name, already known to exist in ``rec.channels``.
        declared: The sidecar's ``units`` value, already known to be non-empty
            and not ``n/a``.

    Returns:
        True if the channel record changed (unit adopted, or ``bids_unit``
        recorded), False if it was already in the sidecar's unit.
    """
    info = rec.channels[label]
    current = str(info.get("physical_dimension") or "").strip()
    if current == declared:
        return False

    factor = conversion_factor(current, declared)
    if factor is None:
        return _keep_importer_unit(
            info, label, current, declared, f"is not convertible to {declared!r}"
        )

    if factor != 1.0:
        signals = rec.signals
        if signals is None or label not in signals:
            return _keep_importer_unit(info, label, current, declared, "has no samples to rescale")
        signals[label] = signals[label] * factor

    info["physical_dimension"] = declared
    return True


def apply_channels_tsv(rec: Recording, channels_tsv_path: str) -> int:
    """Override per-channel ``type``/``units`` in ``rec`` from a ``_channels.tsv``.

    Rows are matched to channels by the ``name`` column; ``n/a`` and empty
    values are skipped (the importer-inferred value is kept). An unrecognized
    ``type`` is warned about and skipped rather than raising.

    Adopting the sidecar's ``units`` **converts the channel's samples** into that
    unit rather than merely relabelling them (issue #122); see
    :func:`_adopt_units` for the per-channel rule and for what happens when the
    two units are not convertible. Type and units are applied independently, so
    an unrecognized ``type`` no longer costs the row its unit correction.

    Args:
        rec: The Recording object to update in place.
        channels_tsv_path: Path to the BIDS ``_channels.tsv``.

    Returns:
        The number of channels whose record changed.
    """
    df = pd.read_csv(channels_tsv_path, sep="\t", dtype=str, keep_default_na=False)
    if "name" not in df.columns:
        logging.warning("channels.tsv has no 'name' column: %s", channels_tsv_path)
        return 0

    updated = 0
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        if name not in rec.channels:
            continue
        changed = False
        ctype = str(row.get("type", "")).strip()
        if ctype and ctype.lower() != "n/a":
            try:
                rec.set_channel(name, channel_type=ctype)
                changed = True
            except ValueError:
                logging.warning(
                    "channels.tsv type %r for channel %r is not a known channel type; "
                    "keeping the importer-inferred type.",
                    ctype,
                    name,
                )
        units = str(row.get("units", "")).strip()
        if units and units.lower() != "n/a":
            changed = _adopt_units(rec, name, units) or changed
        if changed:
            updated += 1
    return updated


def read_events_tsv(
    events_tsv_path: str,
    *,
    description_column: str | None = None,
) -> pd.DataFrame | None:
    """Parse a BIDS ``_events.tsv`` into the standard events frame.

    Returns a DataFrame with ``onset``/``duration``/``description`` columns
    (sorted by onset), the same shape :attr:`Recording.events` uses, or **None**
    when the sidecar is unparsable (no ``onset`` column, or a forced
    ``description_column`` is absent) -- distinct from a valid-but-empty table (an
    empty DataFrame). Callers use None to leave any existing events untouched
    rather than wiping them. Used by :func:`apply_events_tsv` and by the streaming
    Zarr exporter (which has no Recording to mutate). See :func:`apply_events_tsv`
    for the column rules.
    """
    df = pd.read_csv(events_tsv_path, sep="\t", dtype=str, keep_default_na=False)
    if "onset" not in df.columns:
        logging.warning("events.tsv has no 'onset' column: %s", events_tsv_path)
        return None

    def _is_na(value: str) -> bool:
        v = value.strip()
        return v == "" or v.lower() == "n/a"

    if description_column is not None:
        if description_column not in df.columns:
            logging.warning("events.tsv has no %r column: %s", description_column, events_tsv_path)
            return None
        desc_columns = [description_column]
    else:
        desc_columns = [c for c in ("trial_type", "value") if c in df.columns]

    onsets: list[float] = []
    durations: list[float] = []
    descriptions: list[str] = []
    skipped = 0
    for _, row in df.iterrows():
        raw_onset = str(row["onset"]).strip()
        try:
            onset = float(raw_onset)
        except ValueError:
            skipped += 1
            continue
        if not math.isfinite(onset):  # NaN or +/-inf
            skipped += 1
            continue
        raw_duration = str(row.get("duration", "")).strip()
        try:
            duration = 0.0 if _is_na(raw_duration) else float(raw_duration)
        except ValueError:
            duration = 0.0
        description = "n/a"
        for col in desc_columns:
            candidate = str(row.get(col, "")).strip()
            if not _is_na(candidate):
                description = candidate
                break
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(description)

    if skipped:
        logging.warning(
            "events.tsv: skipped %d row(s) with missing/non-numeric onset: %s",
            skipped,
            events_tsv_path,
        )

    return (
        pd.DataFrame({"onset": onsets, "duration": durations, "description": descriptions})
        .sort_values(by="onset")
        .reset_index(drop=True)
    )


def apply_events_tsv(
    rec: Recording,
    events_tsv_path: str,
    *,
    description_column: str | None = None,
) -> int:
    """Replace ``rec.events`` with the authoritative BIDS ``_events.tsv`` list.

    A BIDS ``_events.tsv`` is the curated event table for a recording; it is
    richer and more reliable than the data file's own markers (e.g. a
    BrainVision ``.vmrk`` or an EEGLAB event struct), so when it is present it
    is the source of truth. This loads it into ``rec.events`` as the standard
    ``onset``/``duration``/``description`` frame, overwriting any
    importer-loaded events.

    Columns:
        ``onset`` (required, seconds) and ``duration`` (seconds; ``n/a`` or
        missing -> ``0.0``) follow the BIDS spec. The ``description`` is taken
        from ``description_column`` if given, else per row from the first of
        ``trial_type`` then ``value`` that is present and not ``n/a`` (BIDS
        names the categorical label ``trial_type`` and the raw marker
        ``value``; datasets populate one or both). Rows whose ``onset`` is
        missing or non-numeric are skipped.

    Args:
        rec: The Recording object to update in place.
        events_tsv_path: Path to the BIDS ``_events.tsv``.
        description_column: Force the description to come from this column.

    Returns:
        The number of events loaded into ``rec.events``. An unparsable sidecar
        (no ``onset`` column / missing forced column) loads nothing and leaves any
        importer-loaded events intact, returning 0.
    """
    events = read_events_tsv(events_tsv_path, description_column=description_column)
    if events is None:
        return 0
    rec.events = events
    return len(rec.events)
