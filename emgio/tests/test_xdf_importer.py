"""Tests for XDF importer functionality."""

import os
import tempfile

import numpy as np
import pytest

from ..core.emg import EMG
from ..importers.xdf import XDFImporter, XDFStreamInfo, XDFSummary, summarize_xdf

# Path to the sample XDF file
SAMPLE_XDF_PATH = "examples/test.xdf"


def test_summarize_xdf():
    """Test summarize_xdf function with sample data."""
    summary = summarize_xdf(SAMPLE_XDF_PATH)

    # Check summary structure
    assert isinstance(summary, XDFSummary)
    assert summary.filepath == SAMPLE_XDF_PATH
    assert len(summary.streams) == 1

    # Check stream info
    stream = summary.streams[0]
    assert isinstance(stream, XDFStreamInfo)
    assert stream.name == "obci_neeg1"
    assert stream.stream_type == "EEG"
    assert stream.channel_count == 8
    assert stream.nominal_srate == 1000.0
    assert stream.sample_count == 9520
    assert stream.duration_seconds > 9.0


def test_summarize_xdf_str_representation():
    """Test string representation of XDF summary."""
    summary = summarize_xdf(SAMPLE_XDF_PATH)

    # Convert to string
    summary_str = str(summary)

    # Check key elements are in string
    assert "test.xdf" in summary_str
    assert "obci_neeg1" in summary_str
    assert "EEG" in summary_str
    assert "8" in summary_str  # channel count
    assert "1000" in summary_str  # sample rate


def test_summarize_xdf_stream_lookup():
    """Test stream lookup methods in XDFSummary."""
    summary = summarize_xdf(SAMPLE_XDF_PATH)

    # Test get_streams_by_type
    eeg_streams = summary.get_streams_by_type("EEG")
    assert len(eeg_streams) == 1
    assert eeg_streams[0].name == "obci_neeg1"

    # Test get_streams_by_type with non-existent type
    emg_streams = summary.get_streams_by_type("EMG")
    assert len(emg_streams) == 0

    # Test get_stream_by_name
    stream = summary.get_stream_by_name("obci_neeg1")
    assert stream is not None
    assert stream.stream_type == "EEG"

    # Test get_stream_by_name with non-existent name
    stream = summary.get_stream_by_name("nonexistent")
    assert stream is None


def test_xdf_importer_basic():
    """Test XDF importer with sample data."""
    importer = XDFImporter()
    emg = importer.load(SAMPLE_XDF_PATH)

    # Check if channels were loaded
    assert len(emg.channels) == 8

    # Check channel properties
    first_channel = list(emg.channels.keys())[0]
    assert emg.channels[first_channel]["sample_frequency"] == 1000.0

    # Check data shape
    assert emg.signals.shape == (9520, 8)

    # Check metadata
    assert emg.get_metadata("device") == "XDF"
    assert emg.get_metadata("source_file") == SAMPLE_XDF_PATH
    assert emg.get_metadata("stream_count") == 1


def test_xdf_importer_from_emg_class():
    """Test XDF import through EMG.from_file."""
    emg = EMG.from_file(SAMPLE_XDF_PATH)

    # Check basic loading
    assert emg.signals is not None
    assert len(emg.channels) == 8
    assert emg.signals.shape[0] == 9520


def test_xdf_importer_stream_selection_by_type():
    """Test XDF importer with stream type selection."""
    importer = XDFImporter()

    # Select by type - should match the stream
    emg = importer.load(SAMPLE_XDF_PATH, stream_types=["EEG"])
    assert len(emg.channels) == 8

    # Select by non-matching type - should raise error
    with pytest.raises(ValueError, match="No matching streams"):
        importer.load(SAMPLE_XDF_PATH, stream_types=["EMG"])


def test_xdf_importer_stream_selection_by_name():
    """Test XDF importer with stream name selection."""
    importer = XDFImporter()

    # Select by name (case-insensitive)
    emg = importer.load(SAMPLE_XDF_PATH, stream_names=["obci_neeg1"])
    assert len(emg.channels) == 8

    # Select by non-matching name
    with pytest.raises(ValueError, match="No matching streams"):
        importer.load(SAMPLE_XDF_PATH, stream_names=["nonexistent"])


def test_xdf_importer_channel_labels():
    """Test channel labeling in XDF importer."""
    importer = XDFImporter()
    emg = importer.load(SAMPLE_XDF_PATH)

    # Check channel names contain stream name prefix
    channel_names = list(emg.channels.keys())
    assert all("obci_neeg1" in name for name in channel_names)


def test_xdf_importer_timestamps():
    """Test timestamp handling in XDF importer."""
    importer = XDFImporter()
    emg = importer.load(SAMPLE_XDF_PATH)

    # Check time index starts at 0
    assert emg.signals.index[0] == pytest.approx(0.0, abs=1e-6)

    # Check duration matches expected
    duration = emg.signals.index[-1]
    assert duration > 9.0  # Should be about 9.5 seconds


def test_xdf_export_roundtrip():
    """Test XDF import and EDF export roundtrip."""
    # Load XDF file
    emg = EMG.from_file(SAMPLE_XDF_PATH)

    # Export to EDF
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        temp_path = f.name

    try:
        emg.to_edf(temp_path, format="edf")

        # Reload from EDF
        emg_reloaded = EMG.from_file(temp_path)

        # Check basic structure preserved
        assert len(emg_reloaded.channels) == len(emg.channels)

        # Check data is similar (allowing for EDF precision loss)
        for orig_ch, reload_ch in zip(
            emg.channels.keys(), emg_reloaded.channels.keys(), strict=False
        ):
            orig_data = emg.signals[orig_ch].values
            reload_data = emg_reloaded.signals[reload_ch].values

            # Trim to same length (EDF may have different sample count)
            min_len = min(len(orig_data), len(reload_data))
            correlation = np.corrcoef(orig_data[:min_len], reload_data[:min_len])[0, 1]
            assert correlation > 0.99, f"Correlation for {orig_ch} is {correlation}"

    finally:
        # Cleanup
        os.unlink(temp_path)
        # Also cleanup the channels.tsv file if it exists
        tsv_path = temp_path.replace(".edf", "_channels.tsv")
        if os.path.exists(tsv_path):
            os.unlink(tsv_path)


def test_xdf_file_not_found():
    """Test error handling for non-existent file."""
    importer = XDFImporter()
    # pyxdf raises a generic Exception for missing files
    with pytest.raises((FileNotFoundError, OSError, Exception), match="does not exist"):
        importer.load("nonexistent.xdf")


def test_xdf_default_channel_type():
    """Test default channel type assignment."""
    importer = XDFImporter()
    emg = importer.load(SAMPLE_XDF_PATH, default_channel_type="EEG")

    # Since the stream has no explicit channel types in desc,
    # channels should get the default type
    for channel_info in emg.channels.values():
        assert channel_info["channel_type"] == "EEG"


# ============================================================================
# Multi-stream XDF tests
# ============================================================================

MULTI_STREAM_XDF_PATH = "examples/multi_stream_test.xdf"


def test_multistream_summarize():
    """Test summarize_xdf with multi-stream file."""
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)

    # Check we have 4 streams
    assert len(summary.streams) == 4

    # Check stream types
    stream_types = {s.stream_type for s in summary.streams}
    assert "EEG" in stream_types
    assert "EMG" in stream_types
    assert "Mocap" in stream_types
    assert "Markers" in stream_types


def test_multistream_import_all():
    """Test importing all numeric streams from multi-stream file."""
    emg = EMG.from_file(MULTI_STREAM_XDF_PATH)

    # Should have channels from EEG, EMG, and Mocap (not Markers - string stream)
    # EEG: 8, EMG: 2, Mocap: 6 = 16 total
    assert len(emg.channels) == 16


def test_multistream_import_by_type():
    """Test importing specific stream types from multi-stream file."""
    # Import only EMG
    emg = EMG.from_file(MULTI_STREAM_XDF_PATH, stream_types=["EMG"])
    assert len(emg.channels) == 2
    channel_names = list(emg.channels.keys())
    assert "EMG_L" in channel_names
    assert "EMG_R" in channel_names

    # Import only EEG
    eeg = EMG.from_file(MULTI_STREAM_XDF_PATH, stream_types=["EEG"])
    assert len(eeg.channels) == 8
    assert all("EEG" in name for name in eeg.channels.keys())

    # Import only Mocap
    mocap = EMG.from_file(MULTI_STREAM_XDF_PATH, stream_types=["Mocap"])
    assert len(mocap.channels) == 6
    assert all("Marker" in name for name in mocap.channels.keys())


def test_multistream_import_by_name():
    """Test importing specific streams by name from multi-stream file."""
    emg = EMG.from_file(MULTI_STREAM_XDF_PATH, stream_names=["TestEMG"])
    assert len(emg.channels) == 2


def test_multistream_import_multiple_types():
    """Test importing multiple stream types from multi-stream file."""
    emg = EMG.from_file(MULTI_STREAM_XDF_PATH, stream_types=["EEG", "EMG"])

    # Should have EEG (8) + EMG (2) = 10 channels
    assert len(emg.channels) == 10


def test_multistream_sampling_rates():
    """Test that different sampling rates are handled correctly."""
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)

    # Check expected sampling rates
    eeg_stream = summary.get_stream_by_name("TestEEG")
    assert eeg_stream.nominal_srate == 256.0

    emg_stream = summary.get_stream_by_name("TestEMG")
    assert emg_stream.nominal_srate == 2048.0

    mocap_stream = summary.get_stream_by_name("TestMocap")
    assert mocap_stream.nominal_srate == 120.0

    marker_stream = summary.get_stream_by_name("TestMarkers")
    assert marker_stream.nominal_srate == 0.0  # Irregular


def test_multistream_marker_stream_info():
    """Test that marker (string) streams are summarized correctly."""
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)

    marker_stream = summary.get_stream_by_name("TestMarkers")
    assert marker_stream is not None
    assert marker_stream.stream_type == "Markers"
    assert marker_stream.sample_count == 5
    assert marker_stream.channel_format == "string"


def test_multistream_channel_labels():
    """Test that channel labels are correctly extracted from multi-stream file."""
    summary = summarize_xdf(MULTI_STREAM_XDF_PATH)

    eeg_stream = summary.get_stream_by_name("TestEEG")
    assert "EEG1" in eeg_stream.channel_labels
    assert len(eeg_stream.channel_labels) == 8

    emg_stream = summary.get_stream_by_name("TestEMG")
    assert "EMG_L" in emg_stream.channel_labels
    assert "EMG_R" in emg_stream.channel_labels
