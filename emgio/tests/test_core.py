import pytest
import numpy as np
from ..core.emg import EMG


@pytest.fixture
def empty_emg():
    """Create an empty EMG object."""
    return EMG()


@pytest.fixture
def sample_emg():
    """Create an EMG object with sample data."""
    emg = EMG()

    # Add sample channels
    time = np.linspace(0, 1, 1000)  # 1 second at 1000Hz
    emg_data = np.sin(2 * np.pi * 10 * time)  # 10Hz sine wave
    acc_data = np.cos(2 * np.pi * 5 * time)   # 5Hz cosine wave

    emg.add_channel('EMG1', emg_data, 1000, 'mV', 'EMG')
    emg.add_channel('ACC1', acc_data, 1000, 'g', 'ACC')

    return emg


def test_emg_initialization(empty_emg):
    """Test EMG object initialization."""
    assert empty_emg.signals is None
    assert empty_emg.metadata == {}
    assert empty_emg.channels == {}


def test_add_channel(empty_emg):
    """Test adding a channel to EMG object."""
    data = np.array([1, 2, 3, 4, 5])
    empty_emg.add_channel('EMG1', data, 1000, 'mV', 'EMG')

    assert 'EMG1' in empty_emg.signals.columns
    assert 'EMG1' in empty_emg.channels
    assert empty_emg.channels['EMG1']['sampling_freq'] == 1000
    assert empty_emg.channels['EMG1']['unit'] == 'mV'
    assert empty_emg.channels['EMG1']['type'] == 'EMG'


def test_select_channels(sample_emg):
    """Test channel selection."""
    # Select multiple channels
    emg_multi = sample_emg.select_channels(['EMG1', 'ACC1'])
    assert list(emg_multi.signals.columns) == ['EMG1', 'ACC1']
    assert list(emg_multi.channels.keys()) == ['EMG1', 'ACC1']

    # Select single channel
    emg_single = sample_emg.select_channels('EMG1')
    assert list(emg_single.signals.columns) == ['EMG1']
    assert list(emg_single.channels.keys()) == ['EMG1']


def test_metadata(empty_emg):
    """Test metadata handling."""
    empty_emg.set_metadata('subject', 'S001')
    assert empty_emg.get_metadata('subject') == 'S001'

    # Test non-existent key
    assert empty_emg.get_metadata('nonexistent') is None


def test_invalid_channel_selection(sample_emg):
    """Test error handling for invalid channel selection."""
    with pytest.raises(ValueError):
        sample_emg.select_channels('NonexistentChannel')


def test_plot_signals_validation(empty_emg):
    """Test plot_signals input validation."""
    with pytest.raises(ValueError):
        empty_emg.plot_signals()  # Should raise error when no signals are loaded
