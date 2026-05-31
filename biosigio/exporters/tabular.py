"""Parquet and Arrow/Feather exporters (biosigIO tabular schema).

Thin wrappers over :mod:`biosigio.tabular_schema`: a Recording becomes one columnar
table (channels = columns, time index preserved) with a self-describing
``biosigio`` metadata blob, written as Parquet (analytics) or Arrow/Feather
(fast IPC). Both round-trip losslessly via ``Recording.from_file``. pyarrow is an
optional dependency (the ``arrow`` extra), imported lazily.
"""

from ..core.emg import Recording
from ..tabular_schema import recording_to_table, require_pyarrow


class TabularExporter:
    """Exporter for the columnar biosigIO formats (Parquet, Arrow/Feather)."""

    @staticmethod
    def to_parquet(rec: Recording, filepath: str) -> str:
        """Write ``rec`` to a self-describing biosigIO Parquet file.

        Args:
            rec: Source recording.
            filepath: Output ``.parquet`` path.

        Returns:
            The written file path.
        """
        require_pyarrow()  # clear install hint before touching the pyarrow submodule
        table = recording_to_table(rec)
        import pyarrow.parquet as pq

        pq.write_table(table, filepath)
        return filepath

    @staticmethod
    def to_arrow(rec: Recording, filepath: str) -> str:
        """Write ``rec`` to a biosigIO Arrow/Feather file (fast zero-copy IPC).

        Args:
            rec: Source recording.
            filepath: Output ``.feather`` / ``.arrow`` path.

        Returns:
            The written file path.
        """
        require_pyarrow()
        table = recording_to_table(rec)
        import pyarrow.feather as feather

        feather.write_feather(table, filepath)
        return filepath
