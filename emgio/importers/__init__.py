from .edf import EDFImporter
from .trigno import TrignoImporter
from .otb import OTBImporter
from .wfdb import WFDBImporter

__all__ = ['BaseImporter', 'EEG LabImporter', 'CSVImporter',
           'EDFImporter', 'TrignoImporter', 'OTBImporter', 'WFDBImporter']

# Mapping from file extensions (or format names) to importer classes
IMPORTER_MAP = {
    '.set': EEG LabImporter,
    '.csv': CSVImporter,
    '.edf': EDFImporter,
    '.bdf': EDFImporter,
    '.h5': TrignoImporter,
    '.mat': OTBImporter,
    '.hea': WFDBImporter
}

def get_importer(filepath: str):
    // ... existing code ...
