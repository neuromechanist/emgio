import os
import shutil

# import numpy as np # Not directly used here
import pandas as pd
import pytest

from emgio.core.emg import EMG
from emgio.importers.wfdb import WFDBImporter

# Directory containing the test WFDB files (relative to project root)
EXAMPLE_DIR = "examples"
WFDB_RECORD_NAME = "100"


@pytest.fixture
def wfdb_data(tmp_path):
    """Fixture to provide WFDB test files in a temporary directory."""
    source_dir = os.path.abspath(EXAMPLE_DIR)
    dest_dir = tmp_path

    # Files needed for record '100'
    files_to_copy = [
        f"{WFDB_RECORD_NAME}.hea",
        f"{WFDB_RECORD_NAME}.dat",
        f"{WFDB_RECORD_NAME}.atr",
    ]

    copied_files = {}
    missing_files = []

    for filename in files_to_copy:
        source_path = os.path.join(source_dir, filename)
        dest_path = dest_dir / filename
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            copied_files[filename] = dest_path
        else:
            missing_files.append(filename)

    if missing_files:
        pytest.skip(f"Missing WFDB example files in '{EXAMPLE_DIR}': {missing_files}")

    return {
        "dir": dest_dir,
        "hea": copied_files.get(f"{WFDB_RECORD_NAME}.hea"),
        "dat": copied_files.get(f"{WFDB_RECORD_NAME}.dat"),
        "atr": copied_files.get(f"{WFDB_RECORD_NAME}.atr"),
    }


@pytest.fixture
def wfdb_importer():
    """Fixture to provide a WFDBImporter instance."""
    return WFDBImporter()


# --- Test Cases ---


def test_wfdb_load_basic(wfdb_importer, wfdb_data):
    """Test basic loading of a WFDB record (.hea file)."""
    emg = wfdb_importer.load(str(wfdb_data["hea"]))

    assert isinstance(emg, EMG)
    assert emg.signals is not None
    assert isinstance(emg.signals, pd.DataFrame)
    # Record 100 has 2 signals
    assert emg.signals.shape[1] == 2
    # Check if time index is created
    assert isinstance(emg.signals.index, pd.Index)

    # Check metadata
    assert emg.get_metadata("record_name") == WFDB_RECORD_NAME
    assert emg.get_metadata("sampling_frequency") == 360.0  # Known frequency for record 100
    assert emg.get_metadata("source_file") == str(wfdb_data["hea"])

    # Check channels
    assert len(emg.channels) == 2
    assert "MLII" in emg.channels
    assert "V5" in emg.channels
    assert emg.channels["MLII"]["physical_dimension"] == "mV"
    assert emg.channels["V5"]["physical_dimension"] == "mV"
    assert emg.channels["MLII"]["sample_frequency"] == 360.0
    assert emg.channels["V5"]["sample_frequency"] == 360.0


def test_wfdb_load_annotations(wfdb_importer, wfdb_data):
    """Test loading WFDB record with annotations (.atr file present)."""
    emg = wfdb_importer.load(str(wfdb_data["hea"]))

    assert emg.events is not None
    assert isinstance(emg.events, pd.DataFrame)
    assert not emg.events.empty
    assert list(emg.events.columns) == ["onset", "duration", "description"]
    # Check specific values for record 100 annotations
    assert len(emg.events) == 2274  # Known number of annotations for record 100.atr
    assert emg.events.iloc[0]["onset"] == pytest.approx(18 / 360.0)
    assert emg.events.iloc[0]["duration"] == 0
    assert emg.events.iloc[0]["description"] == "WFDB Annotation: +"


def test_wfdb_load_no_annotations_file(wfdb_importer, wfdb_data):
    """Test loading WFDB record when the annotation file is missing."""
    # Remove the .atr file
    os.remove(wfdb_data["atr"])

    emg = wfdb_importer.load(str(wfdb_data["hea"]))

    # Events DataFrame should be empty
    assert emg.events is not None
    assert isinstance(emg.events, pd.DataFrame)
    assert emg.events.empty

    # Check metadata status
    assert emg.get_metadata("annotation_status") == "Annotation file (.atr) not found."
    assert emg.get_metadata("annotation_error") is None


def test_wfdb_load_file_not_found(wfdb_importer, tmp_path):
    """Test loading a non-existent WFDB record."""
    non_existent_path = str(tmp_path / "non_existent.hea")
    with pytest.raises(FileNotFoundError) as excinfo:
        wfdb_importer.load(non_existent_path)
    # Check for the new specific message from the importer
    assert "WFDB header file not found" in str(excinfo.value)


def test_wfdb_load_dat_missing(wfdb_importer, wfdb_data):
    """Test loading when the data file (.dat) is missing."""
    # Remove the .dat file
    os.remove(wfdb_data["dat"])

    with pytest.raises(ValueError) as excinfo:
        wfdb_importer.load(str(wfdb_data["hea"]))
    # Check for the specific ValueError raised when .dat is missing during read
    assert "Error reading WFDB record" in str(excinfo.value)
    assert "Data file missing or unreadable" in str(excinfo.value)


def test_wfdb_importer_handles_record_name_input(wfdb_importer, wfdb_data):
    """Test that the importer can load using just the record name within the directory."""
    # Provide the path to the directory and the base record name
    record_base_path = os.path.join(wfdb_data["dir"], WFDB_RECORD_NAME)
    emg = wfdb_importer.load(record_base_path)

    assert isinstance(emg, EMG)
    assert emg.get_metadata("record_name") == WFDB_RECORD_NAME
    assert not emg.events.empty  # Check annotations loaded correctly too
