"""EMGIO: A Python package for EMG data import/export and manipulation."""

from .core.emg import EMG
from .exporters.edf import EDFExporter
from .importers.trigno import TrignoImporter
from .version import __version__, __version_info__

__all__ = ["EMG", "TrignoImporter", "EDFExporter", "__version__", "__version_info__"]
