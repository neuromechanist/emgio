import os
import builtins
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
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

    emg.add_channel('EMG1', emg_data, 1000, 'mV', ch_type='EMG')
    emg.add_channel('ACC1', acc_data, 1000, 'g', ch_type='ACC')

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


def test_get_channel_types(sample_emg):
    """Test getting unique channel types."""
    types = sample_emg.get_channel_types()
    assert set(types) == {'EMG', 'ACC'}


def test_get_channels_by_type(sample_emg):
    """Test getting channels of specific type."""
    emg_channels = sample_emg.get_channels_by_type('EMG')
    acc_channels = sample_emg.get_channels_by_type('ACC')
    
    assert emg_channels == ['EMG1']
    assert acc_channels == ['ACC1']
    assert sample_emg.get_channels_by_type('NONEXISTENT') == []


# This will be implemented after #3 is resolved
# def test_select_channels_by_type(sample_emg):
#     """Test channel selection by type."""
#     # Select all EMG channels
#     emg_only = sample_emg.select_channels(channel_type='EMG')
#     assert list(emg_only.signals.columns) == ['EMG1']
#     assert all(info['type'] == 'EMG' for info in emg_only.channels.values())

#     # Select all ACC channels
#     acc_only = sample_emg.select_channels(channel_type='ACC')
#     assert list(acc_only.signals.columns) == ['ACC1']
#     assert all(info['type'] == 'ACC' for info in acc_only.channels.values())

#     # Test with non-existent type
#     with pytest.raises(ValueError):
#         sample_emg.select_channels(channel_type='NONEXISTENT')


def test_select_channels_with_type_filter(sample_emg):
    """Test channel selection with type filtering."""
    # Select specific channels with type filter
    result = sample_emg.select_channels(['EMG1', 'ACC1'], channel_type='EMG')
    assert list(result.signals.columns) == ['EMG1']
    assert all(info['type'] == 'EMG' for info in result.channels.values())

    # Test when no channels match type
    with pytest.raises(ValueError):
        sample_emg.select_channels(['EMG1', 'ACC1'], channel_type='GYRO')


def test_add_channel_validation(empty_emg):
    """Test add_channel with various data types and validation."""
    # Test with different numpy data types
    data_int = np.array([1, 2, 3], dtype=np.int32)
    empty_emg.add_channel('INT', data_int, 1000, 'count', ch_type='OTHER')
    assert np.array_equal(empty_emg.signals['INT'].values, data_int)

    # Test with float data
    data_float = np.array([1.1, 2.2, 3.3])
    empty_emg.add_channel('FLOAT', data_float, 1000, 'mV')
    assert np.array_equal(empty_emg.signals['FLOAT'].values, data_float)

    # Test channel info storage
    assert empty_emg.channels['INT']['type'] == 'OTHER'
    assert empty_emg.channels['FLOAT']['type'] == 'EMG'  # default type
    assert empty_emg.channels['INT']['sampling_freq'] == 1000
    assert empty_emg.channels['INT']['unit'] == 'count'


@pytest.fixture
def mock_importers(monkeypatch):
    """Mock importers for testing from_file method."""
    class MockBaseImporter:
        """Base class for mock importers to ensure consistent interface."""
        def load(self, filepath):
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            return self._load(filepath)

    class MockTrignoImporter(MockBaseImporter):
        def _load(self, filepath):
            emg = EMG()
            emg.add_channel('TEST', np.array([1, 2, 3]), 1000, 'mV', ch_type='EMG')
            emg.set_metadata('device', 'Delsys Trigno')
            emg.set_metadata('source_file', filepath)
            return emg

    class MockOTBImporter(MockBaseImporter):
        def _load(self, filepath):
            emg = EMG()
            emg.add_channel('OTB', np.array([4, 5, 6]), 2000, 'mV', ch_type='EMG')
            emg.set_metadata('device', 'OT Bioelettronica')
            emg.set_metadata('source_file', filepath)
            return emg

    def mock_import(name, *args):
        # Only intercept our specific importer paths
        if any(x in name for x in ['emgio.importers.trigno', 'emgio.importers.otb']):
            if 'trigno' in name:
                return type('Module', (), {'TrignoImporter': MockTrignoImporter})
            elif 'otb' in name:
                return type('Module', (), {'OTBImporter': MockOTBImporter})
        # Let all other imports pass through to the original __import__
        return original_import(name, *args)
    
    original_import = builtins.__import__

    monkeypatch.setattr('builtins.__import__', mock_import)


def test_from_file(mock_importers, tmp_path):
    """Test factory method with different importers."""
    # Create temporary test files
    trigno_file = tmp_path / "test.csv"
    trigno_file.write_text("")  # Empty file is sufficient for testing
    
    otb_file = tmp_path / "test.otb"
    otb_file.write_text("")
    
    # Test Trigno importer
    emg_trigno = EMG.from_file(str(trigno_file), importer='trigno')
    assert 'TEST' in emg_trigno.signals.columns
    assert emg_trigno.channels['TEST']['sampling_freq'] == 1000

    # Test OTB importer
    emg_otb = EMG.from_file(str(otb_file), importer='otb')
    assert 'OTB' in emg_otb.signals.columns
    assert emg_otb.channels['OTB']['sampling_freq'] == 2000

    # Test invalid importer
    with pytest.raises(ValueError) as exc_info:
        EMG.from_file(str(trigno_file), importer='invalid')
    assert "Unsupported importer" in str(exc_info.value)


@pytest.fixture
def mock_plt(monkeypatch):
    """Mock matplotlib.pyplot for testing plot functions."""
    with patch('matplotlib.pyplot') as mock:
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]  # Create two mock axes by default
        
        # Configure subplots to return fig, axes
        mock.subplots.return_value = (mock_fig, mock_axes)
        
        yield mock


def test_plot_signals_basic(sample_emg, mock_plt):
    """Test basic plotting functionality."""
    sample_emg.plot_signals()
    
    # Verify figure creation
    mock_plt.subplots.assert_called_once()
    
    # Verify plot calls
    fig, axes = mock_plt.subplots.return_value
    assert len(axes) == 2  # Should have 2 axes for EMG1 and ACC1


def test_plot_signals_style_options(sample_emg, mock_plt):
    """Test different plot styles."""
    # Test dots style
    sample_emg.plot_signals(style='dots')
    fig, axes = mock_plt.subplots.return_value
    for ax in axes:
        ax.scatter.assert_called_once()
        ax.plot.assert_not_called()

    mock_plt.reset_mock()

    # Test line style
    sample_emg.plot_signals(style='line')
    fig, axes = mock_plt.subplots.return_value
    for ax in axes:
        ax.plot.assert_called_once()
        ax.scatter.assert_not_called()


def test_plot_signals_customization(sample_emg, mock_plt):
    """Test plot customization options."""
    title = "Test Plot"
    sample_emg.plot_signals(
        channels=['EMG1'],
        grid=True,
        title=title
    )
    
    fig, axes = mock_plt.subplots.return_value
    if not isinstance(axes, list):
        axes = [axes]
    
    # Verify title
    fig.suptitle.assert_called_with(title, fontsize=14, y=1.02)
    
    # Verify grid
    for ax in axes:
        ax.grid.assert_called_with(True, linestyle='--', alpha=0.7)


def test_plot_signals_channel_selection(sample_emg, mock_plt):
    """Test plotting with channel selection."""
    # Test single channel
    sample_emg.plot_signals(channels=['EMG1'])
    fig, axes = mock_plt.subplots.return_value
    if not isinstance(axes, list):
        axes = [axes]
    assert len(axes) == 1

    mock_plt.reset_mock()

    # Test invalid channel
    with pytest.raises(ValueError) as exc_info:
        sample_emg.plot_signals(channels=['NonexistentChannel'])
    assert "Channels not found" in str(exc_info.value)


def test_plot_signals_time_range(sample_emg, mock_plt):
    """Test plotting with time range selection."""
    time_range = (0.2, 0.8)
    sample_emg.plot_signals(time_range=time_range)
    
    # Verify data selection
    fig, axes = mock_plt.subplots.return_value
    for ax in axes:
        data = ax.plot.call_args[0][1]  # Get y-values from plot call
        assert len(data) < len(sample_emg.signals)  # Should be subset of data


@pytest.fixture
def mock_edf_exporter(monkeypatch):
    """Mock EDF exporter for testing export functionality."""
    class MockEDFExporter:
        @staticmethod
        def export(emg_obj, filepath, **kwargs):
            if not filepath.endswith('.edf'):
                raise ValueError("File must have .edf extension")
            # Store export parameters for verification
            MockEDFExporter.last_export = {
                'filepath': filepath,
                'channels': list(emg_obj.channels.keys()),
                'kwargs': kwargs
            }
    
    def mock_import(*args):
        return type('Module', (), {'EDFExporter': MockEDFExporter})
    
    monkeypatch.setattr('builtins.__import__', mock_import)
    return MockEDFExporter


def test_to_edf_export(sample_emg, mock_edf_exporter):
    """Test EDF export functionality."""
    # Test basic export
    filepath = 'test.edf'
    sample_emg.to_edf(filepath)
    
    assert mock_edf_exporter.last_export['filepath'] == filepath
    assert set(mock_edf_exporter.last_export['channels']) == {'EMG1', 'ACC1'}
    
    # Test with additional kwargs
    custom_kwargs = {'patient_id': 'TEST001'}
    sample_emg.to_edf(filepath, **custom_kwargs)
    assert mock_edf_exporter.last_export['kwargs'] == custom_kwargs
    
    # Test invalid file extension
    with pytest.raises(ValueError):
        sample_emg.to_edf('test.txt')


def test_to_edf_empty(empty_emg, mock_edf_exporter):
    """Test EDF export with empty EMG object."""
    with pytest.raises(ValueError):
        empty_emg.to_edf('test.edf')


def test_add_channel_with_prefilter(empty_emg):
    """Test adding channel with prefilter specification."""
    data = np.array([1, 2, 3])
    prefilter = "HP 20Hz"
    empty_emg.add_channel('EMG1', data, 1000, 'mV', prefilter=prefilter)
    
    assert empty_emg.channels['EMG1']['prefilter'] == prefilter


def test_select_channels_none_with_type(sample_emg):
    """Test selecting all channels of a type when channels=None."""
    # Add another EMG channel for testing
    data = np.linspace(0, 1, 1000)
    sample_emg.add_channel('EMG2', data, 1000, 'mV', ch_type='EMG')
    
    # Select all EMG channels
    result = sample_emg.select_channels(channels=None, channel_type='EMG')
    assert set(result.signals.columns) == {'EMG1', 'EMG2'}
    assert all(info['type'] == 'EMG' for info in result.channels.values())


def test_plot_signals_invalid_style(sample_emg, mock_plt):
    """Test plot_signals with invalid style."""
    # Test with dots style
    sample_emg.plot_signals(style='invalid_style')  # Should default to line
    fig, axes = mock_plt.subplots.return_value
    for ax in axes:
        ax.plot.assert_called_once()
        ax.scatter.assert_not_called()


def test_plot_signals_single_channel_array(sample_emg, mock_plt):
    """Test plotting single channel returns correct axes array."""
    sample_emg.plot_signals(channels=['EMG1'])
    fig, axes = mock_plt.subplots.return_value
    
    # Verify axes is properly handled when single channel
    if not isinstance(axes, list):
        axes = [axes]
    assert len(axes) == 1
    
    # Verify channel info in title
    ch_info = sample_emg.channels['EMG1']
    expected_title = f"EMG1 ({ch_info['type']} - {ch_info['sampling_freq']} Hz)"
    axes[0].set_title.assert_called_with(expected_title)


def test_select_channels_empty_result(sample_emg):
    """Test selecting channels with type filter resulting in empty selection."""
    with pytest.raises(ValueError) as exc_info:
        sample_emg.select_channels(['EMG1'], channel_type='GYRO')
    assert "None of the selected channels are of type" in str(exc_info.value)


def test_add_multiple_channels(empty_emg):
    """Test adding multiple channels with different properties."""
    # Add first channel
    data1 = np.array([1, 2, 3])
    empty_emg.add_channel('CH1', data1, 1000, 'mV', ch_type='EMG')
    
    # Add second channel with different properties
    data2 = np.array([4, 5, 6])
    empty_emg.add_channel('CH2', data2, 2000, 'g', ch_type='ACC')
    
    # Verify both channels exist with correct properties
    assert set(empty_emg.signals.columns) == {'CH1', 'CH2'}
    assert empty_emg.channels['CH1']['sampling_freq'] == 1000
    assert empty_emg.channels['CH2']['sampling_freq'] == 2000
    assert empty_emg.channels['CH1']['type'] == 'EMG'
    assert empty_emg.channels['CH2']['type'] == 'ACC'
