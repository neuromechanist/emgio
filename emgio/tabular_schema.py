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


def _json_default(obj: Any):
    """Make numpy scalars/arrays JSON-serializable; fall back to str."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


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
    meta = json.loads(blob)
    if meta.get("format") != FORMAT:
        raise ValueError(f"Unexpected tabular format tag: {meta.get('format')!r}")

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
