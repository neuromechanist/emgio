import pytest
import os
import tempfile
import warnings
import numpy as np
import pyedflib
import pandas as pd
from ..core.emg import EMG
from ..exporters.edf import EDFExporter, _truncate_value


@pytest.fixture
def sample_emg():
    """Create an EMG object with sample data."""
    emg = EMG()

    # Create sample data
    time = np.linspace(0, 1, 1000)  # 1 second at 1000Hz
    emg_data = np.sin(2 * np.pi * 10 * time)  # 10Hz sine wave
    acc_data = np.cos(2 * np.pi * 5 * time)   # 5Hz cosine wave

    # Add channels
    emg.add_channel('EMG1', emg_data, 1000, 'mV', 'EMG')
    emg.add_channel('ACC1', acc_data, 1000, 'g', 'ACC')

    return emg


def test_edf_export(sample_emg):
    """Test EDF export functionality."""
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
        edf_path = f.name

    try:
        # Export to EDF
        EDFExporter.export(sample_emg, edf_path)

        # Check if EDF file was created
        assert os.path.exists(edf_path)

        # Check if channels.tsv was created
        channels_tsv_path = os.path.splitext(edf_path)[0] + '_channels.tsv'
        assert os.path.exists(channels_tsv_path)

        # Verify EDF content
        with pyedflib.EdfReader(edf_path) as f:
            assert f.signals_in_file == 2  # Two channels

            # Check signal headers
            signal_headers = f.getSignalHeaders()
            assert len(signal_headers) == 2

            # Check first channel (EMG1)
            assert signal_headers[0]['label'] == 'EMG1'
            assert signal_headers[0]['dimension'] == 'mV'
            assert signal_headers[0]['sample_rate'] == 1000

            # Check second channel (ACC1)
            assert signal_headers[1]['label'] == 'ACC1'
            assert signal_headers[1]['dimension'] == 'g'
            assert signal_headers[1]['sample_rate'] == 1000

            # Check signal data
            emg_data = f.readSignal(0)
            acc_data = f.readSignal(1)
            assert len(emg_data) == 1000
            assert len(acc_data) == 1000

        # Verify channels.tsv content
        channels_df = pd.read_csv(channels_tsv_path, sep='\t')
        assert len(channels_df) == 2
        assert list(channels_df['name']) == ['EMG1', 'ACC1']
        assert list(channels_df['type']) == ['EMG', 'ACC']
        assert list(channels_df['units']) == ['mV', 'g']
        assert list(channels_df['sampling_frequency']) == [1000, 1000]

    finally:
        # Cleanup
        if os.path.exists(edf_path):
            os.unlink(edf_path)
        if os.path.exists(channels_tsv_path):
            os.unlink(channels_tsv_path)


def test_edf_export_no_signals():
    """Test error handling when exporting empty EMG object."""
    empty_emg = EMG()
    with pytest.raises(ValueError):
        with tempfile.NamedTemporaryFile(suffix='.edf') as f:
            EDFExporter.export(empty_emg, f.name)


def test_edf_export_file_permissions():
    """Test error handling for file permission issues."""
    with pytest.raises(Exception):
        EDFExporter.export(sample_emg(), '/nonexistent/directory/test.edf')


def test_truncate_value():
    """Test value truncation functionality."""
    # Test value that needs truncation
    with warnings.catch_warnings(record=True) as w:
        result = _truncate_value(0.123456789, "test_channel", True)
        assert result == 0.1234567
        assert len(w) == 1
        assert "truncated" in str(w[0].message)

    # Test value that doesn't need truncation
    with warnings.catch_warnings(record=True) as w:
        result = _truncate_value(0.1234, "test_channel", True)
        assert result == 0.1234
        assert len(w) == 0


def test_voltage_conversion():
    """Test automatic voltage conversion from V to mV."""
    emg = EMG()
    # Create sample data in Volts
    time = np.linspace(0, 1, 1000)
    signal = 0.001 * np.sin(2 * np.pi * 10 * time)  # 1mV amplitude in Volts
    emg.add_channel('EMG1', signal, 1000, 'V', 'EMG')

    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
        edf_path = f.name

    try:
        EDFExporter.export(emg, edf_path)
        with pyedflib.EdfReader(edf_path) as f:
            signal_headers = f.getSignalHeaders()
            # Check if unit was converted to mV
            assert signal_headers[0]['dimension'] == 'mV'
            # Check if values were scaled properly
            exported_signal = f.readSignal(0)
            assert np.max(np.abs(exported_signal)) >= 0.9  # Should be ~1mV
    finally:
        if os.path.exists(edf_path):
            os.unlink(edf_path)


def test_bdf_format_selection():
    """Test automatic BDF format selection for high precision data."""
    emg = EMG()
    # Create high precision data that would need BDF
    time = np.linspace(0, 1, 1000)
    signal = 0.0000001234567890 * np.sin(2 * np.pi * 10 * time)
    emg.add_channel('EMG1', signal, 1000, 'V', 'EMG')

    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
        edf_path = f.name
        bdf_path = os.path.splitext(edf_path)[0] + '.bdf'

    try:
        with warnings.catch_warnings(record=True) as w:
            EDFExporter.export(emg, edf_path)
            # Should use BDF and warn about it
            assert os.path.exists(bdf_path)
            assert any("Using BDF format" in str(warn.message) for warn in w)

            # Verify BDF content
            with pyedflib.EdfReader(bdf_path) as f:
                signal_headers = f.getSignalHeaders()
                # Check if using 24-bit resolution
                assert signal_headers[0]['digital_max'] == 8388607
    finally:
        if os.path.exists(edf_path):
            os.unlink(edf_path)
        if os.path.exists(bdf_path):
            os.unlink(bdf_path)
