"""EMGIO: A Python package for EMG data import/export and manipulation."""

from .core.emg import EMG
from .importers.trigno import TrignoImporter
from .exporters.edf import EDFExporter

__version__ = '0.1.0'

__all__ = ['EMG', 'TrignoImporter', 'EDFExporter']
