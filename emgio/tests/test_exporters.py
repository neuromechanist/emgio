import pytest
import os
import tempfile
import warnings
import numpy as np
import pyedflib
import pandas as pd
from ..core.emg import EMG
from ..exporters.edf import EDFExporter, _determine_scaling_factors, _calculate_precision_loss


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


def test_determine_scaling_factors():
    """Test scaling factor calculation."""
    # Test normal case
    phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(-1.0, 1.0)
    assert dig_min == -32768
    assert dig_max == 32767
    assert scaling == 32767.0  # Full range mapping

    # Test BDF mode
    phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(-1.0, 1.0, use_bdf=True)
    assert dig_min == -8388608
    assert dig_max == 8388607
    assert scaling == 8388607.0  # Full range mapping

    # Test constant signal
    phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(1.0, 1.0)
    assert phys_min < phys_max  # Should create a small range
    assert abs(abs(phys_max - phys_min) - 0.002) < 1e-4  # 0.1% margin on each side

    # Test zero signal
    phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(0.0, 0.0)
    assert phys_min == -1.0
    assert phys_max == 1.0


def test_calculate_precision_loss():
    """Test precision loss calculation."""
    # Create test signal
    signal = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    # Test with perfect scaling (no loss)
    scaling = 32767.0  # Maps [-1, 1] to full 16-bit range
    loss = _calculate_precision_loss(signal, scaling, -32768, 32767)
    assert loss < 0.01  # Should be minimal loss

    # Test with reduced scaling (some loss)
    scaling = 16383.5  # Maps [-1, 1] to half the range
    loss = _calculate_precision_loss(signal, scaling, -32768, 32767)
    assert loss > 0.0  # Should have some loss

    # Test with zero signal
    signal = np.zeros(5)
    loss = _calculate_precision_loss(signal, scaling, -32768, 32767)
    assert loss == 0.0


def test_edf_export(sample_emg):
    """Test EDF export functionality."""
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
        edf_path = f.name

    try:
        # Export to EDF
        EDFExporter.export(sample_emg, edf_path, precision_threshold=1)

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
            assert signal_headers[0]['sample_frequency'] == 1000

            # Verify scaling is correct
            assert signal_headers[0]['digital_min'] == -32768
            assert signal_headers[0]['digital_max'] == 32767
            assert signal_headers[0]['physical_min'] < signal_headers[0]['physical_max']

            # Check second channel (ACC1)
            assert signal_headers[1]['label'] == 'ACC1'
            assert signal_headers[1]['dimension'] == 'g'
            assert signal_headers[1]['sample_frequency'] == 1000

            # Verify scaling is correct
            assert signal_headers[1]['digital_min'] == -32768
            assert signal_headers[1]['digital_max'] == 32767
            assert signal_headers[1]['physical_min'] < signal_headers[1]['physical_max']

            # Check signal data and verify values are within digital range
            emg_data = f.readSignal(0)
            acc_data = f.readSignal(1)
            assert len(emg_data) == 1000
            assert len(acc_data) == 1000
            assert np.all(emg_data >= signal_headers[0]['physical_min'])
            assert np.all(emg_data <= signal_headers[0]['physical_max'])
            assert np.all(acc_data >= signal_headers[1]['physical_min'])
            assert np.all(acc_data <= signal_headers[1]['physical_max'])

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


def test_edf_export_file_permissions(sample_emg):
    """Test error handling for file permission issues."""
    with pytest.raises(Exception):
        EDFExporter.export(sample_emg, '/nonexistent/directory/test.edf')


def test_precision_threshold():
    """Test precision threshold parameter."""
    emg = EMG()
    time = np.linspace(0, 1, 1000)
    # Create signal that would have ~0.1% precision loss in EDF
    signal = np.sin(2 * np.pi * 10 * time) * 32800  # Just above 16-bit range
    emg.add_channel('EMG1', signal, 1000, 'uV', 'EMG')

    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
        edf_path = f.name
        bdf_path = os.path.splitext(edf_path)[0] + '.bdf'

    try:
        # Should use EDF with higher threshold
        EDFExporter.export(emg, edf_path, precision_threshold=0.2)
        assert os.path.exists(edf_path)
        assert not os.path.exists(bdf_path)

        # Should use BDF with lower threshold
        EDFExporter.export(emg, edf_path, precision_threshold=0.01)
        assert os.path.exists(bdf_path)

    finally:
        if os.path.exists(edf_path):
            os.unlink(edf_path)
        if os.path.exists(bdf_path):
            os.unlink(bdf_path)


def test_format_reproducibility():
    """Test signal reproducibility for both EDF and BDF formats."""
    time = np.linspace(0, 1, 1000)

    # Test BDF format with large amplitude signal
    emg = EMG()
    bdf_signal = np.sin(2 * np.pi * 10 * time) * 1e6  # Large amplitude
    emg.add_channel('EMG1', bdf_signal, 1000, 'uV', 'EMG')

    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
        edf_path = f.name
        bdf_path = os.path.splitext(edf_path)[0] + '.bdf'

    try:
        # Test BDF reproducibility
        EDFExporter.export(emg, edf_path)
        assert os.path.exists(bdf_path)  # Should use BDF for large signal
        with pyedflib.EdfReader(bdf_path) as f:
            bdf_data = f.readSignal(0)
            bdf_correlation = np.corrcoef(bdf_signal, bdf_data)[0, 1]
            assert bdf_correlation > 0.99, f"BDF correlation ({bdf_correlation}) below threshold"

        # Test EDF reproducibility with smaller signal
        emg = EMG()
        edf_signal = np.sin(2 * np.pi * 10 * time) * 1000  # Smaller amplitude
        emg.add_channel('EMG1', edf_signal, 1000, 'uV', 'EMG')

        EDFExporter.export(emg, edf_path, precision_threshold=0.1)
        assert os.path.exists(edf_path)  # Should use EDF for smaller signal
        with pyedflib.EdfReader(edf_path) as f:
            edf_data = f.readSignal(0)
            edf_correlation = np.corrcoef(edf_signal, edf_data)[0, 1]
            assert edf_correlation > 0.99, f"EDF correlation ({edf_correlation}) below threshold"

    finally:
        if os.path.exists(edf_path):
            os.unlink(edf_path)
        if os.path.exists(bdf_path):
            os.unlink(bdf_path)


def test_bdf_format_selection():
    """Test automatic BDF format selection for high precision data."""
    emg = EMG()
    time = np.linspace(0, 1, 1000)
    # Create signal that requires 24-bit resolution
    signal = np.sin(2 * np.pi * 10 * time) * 1e6  # Large amplitude
    emg.add_channel('EMG1', signal, 1000, 'uV', 'EMG')

    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
        edf_path = f.name
        bdf_path = os.path.splitext(edf_path)[0] + '.bdf'

    try:
        with warnings.catch_warnings(record=True) as w:
            EDFExporter.export(emg, edf_path)
            # Should use BDF and warn about it
            assert os.path.exists(bdf_path)
            assert any("Using BDF format" in str(warn.message) for warn in w)

            # Verify BDF content and scaling
            with pyedflib.EdfReader(bdf_path) as f:
                signal_headers = f.getSignalHeaders()
                assert signal_headers[0]['digital_max'] == 8388607
                assert signal_headers[0]['digital_min'] == -8388608

                # Read signal and verify values are within physical range
                data = f.readSignal(0)
                assert np.all(data >= signal_headers[0]['physical_min'] - 0.001)  # Allow margin for rounding errors
                assert np.all(data <= signal_headers[0]['physical_max'] + 0.001)

                # Verify signal shape is preserved
                correlation = np.corrcoef(signal, data)[0, 1]
                assert correlation > 0.99  # High correlation with original
    finally:
        if os.path.exists(edf_path):
            os.unlink(edf_path)
        if os.path.exists(bdf_path):
            os.unlink(bdf_path)
