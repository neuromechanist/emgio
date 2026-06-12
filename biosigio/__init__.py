"""biosigIO: import/export biosignal recordings across formats.

The core class is :class:`Recording` (modality-agnostic: EEG/EMG/iEEG/MEG/...).
"""

from .core.emg import Recording
from .exporters.edf import EDFExporter
from .exporters.zarr_stream import stream_to_zarr
from .importers.trigno import TrignoImporter
from .version import __version__, __version_info__

__all__ = [
    "Recording",
    "TrignoImporter",
    "EDFExporter",
    "stream_to_zarr",
    "__version__",
    "__version_info__",
]
