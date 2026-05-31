"""Importer for the columnar biosigIO formats (Parquet, Arrow/Feather).

Reads a biosigIO tabular file (written by :class:`TabularExporter`) back into a
:class:`~emgio.core.emg.Recording`, reconstructing signals (with their time
index), per-channel metadata, events, and recording metadata from the
self-describing ``biosigio`` schema blob. pyarrow is an optional dependency
(the ``arrow`` extra), imported lazily.
"""

import os

from ..core.emg import Recording
from ..tabular_schema import require_pyarrow, table_to_recording
from .base import BaseImporter


class TabularImporter(BaseImporter):
    """Importer for .parquet and .feather/.arrow biosigIO files."""

    def load(self, filepath: str) -> Recording:
        require_pyarrow()
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == ".parquet":
                import pyarrow.parquet as pq

                table = pq.read_table(filepath)
            else:  # .feather / .arrow
                import pyarrow.feather as feather

                table = feather.read_table(filepath)
        except Exception as e:
            raise ValueError(f"Error reading tabular file {filepath}: {e}") from e
        return table_to_recording(table)
