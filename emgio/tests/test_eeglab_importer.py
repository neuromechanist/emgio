import os
import tempfile

import numpy as np
import pytest
import scipy.io

from ..importers.eeglab import EEGLABImporter


@pytest.fixture
def sample_eeglab_set():
    """Create a sample EEGLAB .set file."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".set", delete=False) as f:
        temp_path = f.name

    # Create sample data
    n_channels = 32
    n_samples = 1000
    sampling_freq = 1000  # Hz

    # Create synthetic EMG data
    data = np.random.randn(n_channels, n_samples) * 0.1  # Small amplitude noise

    # Add some EMG-like signals to a few channels
    t = np.arange(n_samples) / sampling_freq
    for i in range(5):
        # Add 50Hz sine wave (EMG-like) with random phase and amplitude
        phase = np.random.rand() * 2 * np.pi
        amp = np.random.rand() * 0.5 + 0.5
        data[i, :] += amp * np.sin(2 * np.pi * 50 * t + phase)

    # Create channel locations
    chanlocs = np.zeros(
        (1, n_channels),
        dtype=[
            ("labels", "O"),
            ("type", "O"),
            ("X", "O"),
            ("Y", "O"),
            ("Z", "O"),
            ("sph_theta", "O"),
            ("sph_phi", "O"),
            ("sph_radius", "O"),
            ("theta", "O"),
            ("radius", "O"),
            ("ref", "O"),
            ("urchan", "O"),
        ],
    )

    # Fill channel info
    for i in range(n_channels):
        chanlocs[0, i] = (
            np.array([f"emg{i}_left" if i < 16 else f"emg{i - 16}_right"]),  # labels
            np.array(["EMG"]),  # type
            np.array([]),  # X
            np.array([]),  # Y
            np.array([]),  # Z
            np.array([[0]]),  # sph_theta
            np.array([[0]]),  # sph_phi
            np.array([[0]]),  # sph_radius
            np.array([[0]]),  # theta
            np.array([[0]]),  # radius
            np.array([]),  # ref
            np.array([]),  # urchan
        )

    # Create events
    events = np.zeros(
        (1, 5),
        dtype=[
            ("latency", "O"),
            ("duration", "O"),
            ("sample", "O"),
            ("trial_type", "O"),
            ("type", "O"),
            ("urevent", "O"),
        ],
    )

    # Fill event info
    for i in range(5):
        events[0, i] = (
            np.array([[i * 200 + 100]]),  # latency
            np.array([[0]]),  # duration
            np.array([[i * 200 + 100]]),  # sample
            np.array([f"key/{chr(97 + i)}"]),  # trial_type (key/a, key/b, etc.)
            np.array([f"{i + 1}"]),  # type
            np.array([[i + 1]]),  # urevent
        )

    # Create EEGLAB .set file structure
    eeglab_data = {
        "setname": np.array(["test_emg"]),
        "filename": np.array(["test_emg.set"]),
        "filepath": np.array([""]),
        "subject": np.array(["test_subject"]),
        "group": np.array(["test_group"]),
        "condition": np.array(["test_condition"]),
        "session": np.array(["001"]),
        "comments": np.array(["Test EMG data"]),
        "nbchan": np.array([[n_channels]]),
        "trials": np.array([[1]]),
        "pnts": np.array([[n_samples]]),
        "srate": np.array([[sampling_freq]]),
        "xmin": np.array([[0.0]]),
        "xmax": np.array([[n_samples / sampling_freq]]),
        "times": np.array([np.arange(n_samples) / sampling_freq]),
        "data": data,
        "chanlocs": chanlocs,
        "event": events,
    }

    # Save to .set file
    scipy.io.savemat(temp_path, eeglab_data)

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


def test_eeglab_importer(sample_eeglab_set):
    """Test EEGLAB importer with sample data."""
    importer = EEGLABImporter()
    emg = importer.load(sample_eeglab_set)

    # Check if metadata was loaded
    assert emg.get_metadata("subject") == "test_subject"
    assert emg.get_metadata("srate") == 1000
    assert emg.get_metadata("device") == "EEGLAB"

    # Check if signals were loaded
    assert emg.signals is not None
    assert emg.signals.shape[0] == 1000  # 1000 samples
    assert emg.signals.shape[1] == 32  # 32 channels

    # Check if channel info was loaded
    assert len(emg.channels) == 32
    for _ch_name, ch_info in emg.channels.items():
        assert ch_info["channel_type"] == "EMG"
        assert ch_info["sample_frequency"] == 1000
        assert ch_info["physical_dimension"] == "uV"

    # Check if events were loaded
    assert "events" in emg.metadata
    events = emg.metadata["events"]
    assert len(events) == 5
    for i, event in enumerate(events):
        assert event["latency"] == i * 200 + 100
        assert event["type"] == f"{i + 1}"


def test_eeglab_file_not_found():
    """Test error handling for non-existent file."""
    importer = EEGLABImporter()
    with pytest.raises(ValueError):
        importer.load("nonexistent.set")


def test_eeglab_metadata_extraction(sample_eeglab_set):
    """Test metadata extraction from EEGLAB file."""
    importer = EEGLABImporter()
    emg = importer.load(sample_eeglab_set)

    # Test basic metadata
    assert emg.get_metadata("setname") == "test_emg"
    assert emg.get_metadata("subject") == "test_subject"
    assert emg.get_metadata("group") == "test_group"
    assert emg.get_metadata("condition") == "test_condition"
    assert emg.get_metadata("session") == "001"
    assert emg.get_metadata("comments") == "Test EMG data"

    # Test recording parameters
    assert emg.get_metadata("srate") == 1000
    assert emg.get_metadata("nbchan") == 32
    assert emg.get_metadata("trials") == 1
    assert emg.get_metadata("pnts") == 1000
    assert emg.get_metadata("xmin") == 0.0
    assert emg.get_metadata("xmax") == 1.0


def test_eeglab_channel_type_detection(sample_eeglab_set):
    """Test channel type detection from EEGLAB file."""
    importer = EEGLABImporter()
    emg = importer.load(sample_eeglab_set)

    # All channels should be EMG type
    for _ch_name, ch_info in emg.channels.items():
        assert ch_info["channel_type"] == "EMG"

    # Test channel naming
    for i in range(16):
        assert f"emg{i}_left" in emg.channels
    for i in range(16):
        assert f"emg{i}_right" in emg.channels


def test_eeglab_event_processing(sample_eeglab_set):
    """Test event processing from EEGLAB file."""
    importer = EEGLABImporter()
    emg = importer.load(sample_eeglab_set)

    # Check if events were loaded
    assert "events" in emg.metadata
    events = emg.metadata["events"]
    assert len(events) == 5

    # Check event properties
    for i, event in enumerate(events):
        assert event["latency"] == i * 200 + 100
        assert event["type"] == f"{i + 1}"
        assert "trial_type" in event
        assert event["trial_type"] == f"key/{chr(97 + i)}"


def test_eeglab_to_edf_export(sample_eeglab_set):
    """Test exporting EEGLAB data to EDF format."""
    importer = EEGLABImporter()
    emg = importer.load(sample_eeglab_set)

    # Export to EDF
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        edf_path = f.name

    try:
        emg.to_edf(edf_path)

        # Check if EDF file was created
        assert os.path.exists(edf_path) or os.path.exists(os.path.splitext(edf_path)[0] + ".bdf")

        # Check if channels.tsv was created
        channels_tsv_path = os.path.splitext(edf_path)[0] + "_channels.tsv"
        assert os.path.exists(channels_tsv_path)
    finally:
        # Cleanup
        if os.path.exists(edf_path):
            os.unlink(edf_path)
        if os.path.exists(os.path.splitext(edf_path)[0] + ".bdf"):
            os.unlink(os.path.splitext(edf_path)[0] + ".bdf")
        if os.path.exists(channels_tsv_path):
            os.unlink(channels_tsv_path)
