"""Canonical biosigIO tabular serialization schema (Parquet / Arrow / Feather).

A :class:`~emgio.core.emg.Recording` is stored as a single columnar table whose
columns are the channel signals (the time index is preserved). All non-signal
state -- recording metadata, per-channel info, and events -- is serialized as one
JSON blob under the ``biosigio`` schema-metadata key, so the file is fully
self-describing and round-trips losslessly. The same schema backs both the
exporter and the importer (and is intended to back the future Zarr path), so it
lives here once rather than being duplicated.

pyarrow is an optional dependency (``arrow`` extra), imported lazily.
"""

from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import pyarrow as pa

    from .core.emg import Recording

# Schema-metadata key + format tag/version, so a reader can recognize and
# version-check a biosigIO tabular file.
METADATA_KEY = b"biosigio"
FORMAT = "biosigio-tabular"
VERSION = 1


def require_pyarrow():
    """Import pyarrow lazily, raising a clear install hint when it is absent."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Parquet/Arrow serialization requires pyarrow, an optional dependency. "
            "Install it with: uv sync --extra arrow  (or, for an existing install, "
            "uv pip install 'emgio[arrow]')."
        ) from e
    import pyarrow

    return pyarrow


# Marks a JSON object that encodes a non-JSON-native value (e.g. a datetime) so
# it can be reconstructed to its original type on read, instead of silently
# degrading to a string.
_TYPE_KEY = "__biosigio_type__"


def _json_default(obj: Any):
    """Encode non-JSON-native values losslessly; RAISE rather than silently coerce.

    datetimes/dates become a typed envelope (reconstructed by ``_json_object_hook``)
    and numpy scalars/arrays become their Python equivalents. Anything else raises
    a TypeError so unexpected metadata is surfaced, never silently str()-ified
    (metadata loss is data loss).
    """
    if isinstance(obj, datetime.datetime):
        return {_TYPE_KEY: "datetime", "value": obj.isoformat()}
    if isinstance(obj, datetime.date):
        return {_TYPE_KEY: "date", "value": obj.isoformat()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f"Object of type {type(obj).__name__!r} in recording metadata is not JSON-"
        "serializable and cannot be stored in the biosigIO tabular schema; convert "
        "it to a primitive (or datetime) before exporting."
    )


def _json_object_hook(d: dict) -> Any:
    """Reconstruct typed envelopes written by :func:`_json_default`."""
    kind = d.get(_TYPE_KEY)
    if kind == "datetime":
        return datetime.datetime.fromisoformat(d["value"])
    if kind == "date":
        return datetime.date.fromisoformat(d["value"])
    return d


def recording_to_table(rec: Recording) -> pa.Table:
    """Build a pyarrow Table from a Recording (signals + biosigio metadata blob)."""
    pa = require_pyarrow()
    if rec.signals is None:
        raise ValueError("No signals to serialize")

    table = pa.Table.from_pandas(rec.signals, preserve_index=True)
    events = (
        rec.events.to_dict("records") if rec.events is not None and not rec.events.empty else []
    )
    blob = json.dumps(
        {
            "format": FORMAT,
            "version": VERSION,
            "metadata": dict(rec.metadata),
            "channels": {name: dict(info) for name, info in rec.channels.items()},
            "events": events,
        },
        default=_json_default,
    ).encode("utf-8")
    existing = table.schema.metadata or {}
    return table.replace_schema_metadata({**existing, METADATA_KEY: blob})


def table_to_recording(table: pa.Table) -> Recording:
    """Reconstruct a Recording from a biosigIO tabular Table."""
    from .core.emg import Recording

    blob = (table.schema.metadata or {}).get(METADATA_KEY)
    if blob is None:
        raise ValueError(
            "Not a biosigIO tabular file: missing the 'biosigio' schema metadata. "
            "Only files written by emgio's Parquet/Arrow exporter can be imported."
        )
    meta = json.loads(blob, object_hook=_json_object_hook)
    if meta.get("format") != FORMAT:
        raise ValueError(f"Unexpected tabular format tag: {meta.get('format')!r}")
    if meta.get("version") != VERSION:
        raise ValueError(
            f"Unsupported biosigIO tabular schema version {meta.get('version')!r} "
            f"(this build reads version {VERSION})."
        )

    rec = Recording()
    rec.signals = table.to_pandas()  # restores the preserved (time) index
    rec.metadata = meta.get("metadata", {})
    rec.channels = meta.get("channels", {})

    event_rows = meta.get("events", [])
    if event_rows:
        events = pd.DataFrame(event_rows, columns=["onset", "duration", "description"])
        events["onset"] = events["onset"].astype("float64")
        events["duration"] = events["duration"].astype("float64")
        rec.events = events
    return rec
