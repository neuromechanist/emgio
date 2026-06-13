"""biosigIO: import/export biosignal recordings across formats.

The core class is :class:`Recording` (modality-agnostic: EEG/EMG/iEEG/MEG/...).
"""

from .core.emg import Recording
from .exceptions import (
    REASONS,
    BiosigIOError,
    CorruptFileError,
    EmptyRecordingError,
    FileReadError,
    MixedSamplingRateError,
    NotContinuousRecordingError,
    UnsupportedFormatError,
    classify_read_error,
)
from .exporters.edf import EDFExporter
from .exporters.zarr_stream import stream_to_zarr
from .importers.trigno import TrignoImporter
from .version import __version__, __version_info__

__all__ = [
    "Recording",
    "TrignoImporter",
    "EDFExporter",
    "stream_to_zarr",
    "BiosigIOError",
    "UnsupportedFormatError",
    "FileReadError",
    "NotContinuousRecordingError",
    "CorruptFileError",
    "EmptyRecordingError",
    "MixedSamplingRateError",
    "classify_read_error",
    "REASONS",
    "__version__",
    "__version_info__",
]
