"""emgio (evolving into biosigIO): import/export biosignal recordings across formats.

The core class is :class:`Recording` (modality-agnostic: EEG/EMG/iEEG/MEG/...).
``EMG`` is a deprecated alias of ``Recording`` kept for backward compatibility.
"""

from .core.emg import EMG, Recording
from .exporters.edf import EDFExporter
from .importers.trigno import TrignoImporter
from .version import __version__, __version_info__

__all__ = [
    "Recording",
    "EMG",
    "TrignoImporter",
    "EDFExporter",
    "__version__",
    "__version_info__",
]
