"""Parquet and Arrow/Feather exporters (biosigIO tabular schema).

Thin wrappers over :mod:`emgio.tabular_schema`: a Recording becomes one columnar
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
        require_pyarrow()
        import pyarrow.parquet as pq

        pq.write_table(recording_to_table(rec), filepath)
        return filepath

    @staticmethod
    def to_arrow(rec: Recording, filepath: str) -> str:
        require_pyarrow()
        import pyarrow.feather as feather

        feather.write_feather(recording_to_table(rec), filepath)
        return filepath
