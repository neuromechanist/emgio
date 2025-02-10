import pytest
import os
import tempfile
import numpy as np
import pyedflib
from ..importers.trigno import TrignoImporter
from ..importers.otb import OTBImporter
from ..importers.edf import EDFImporter


@pytest.fixture
def sample_trigno_csv():
    """Create a sample Trigno CSV file."""
    content = '''Label: EMG1 Sampling frequency: 1000 Unit: mV Domain: Time
Label: EMG2 Sampling frequency: 1000 Unit: mV Domain: Time
Label: ACC1 Sampling frequency: 1000 Unit: g Domain: Time

X[s],"EMG1","EMG2","ACC1"
0.000,0.1,0.2,0.3
0.001,0.2,0.3,0.4
0.002,0.3,0.4,0.5
0.003,0.4,0.5,0.6
0.004,0.5,0.6,0.7'''

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


def test_trigno_importer(sample_trigno_csv):
    """Test Trigno importer with sample data."""
    importer = TrignoImporter()
    emg = importer.load(sample_trigno_csv)

    # Check if all channels were loaded
    assert 'EMG1' in emg.channels
    assert 'EMG2' in emg.channels
    assert 'ACC1' in emg.channels

    # Check channel properties
    assert emg.channels['EMG1']['sampling_freq'] == 1000
    assert emg.channels['EMG1']['unit'] == 'mV'
    assert emg.channels['EMG1']['type'] == 'EMG'

    assert emg.channels['ACC1']['unit'] == 'g'
    assert emg.channels['ACC1']['type'] == 'ACC'

    # Check data shape
    assert len(emg.signals) == 5  # 5 samples
    assert len(emg.signals.columns) == 3  # 3 channels

    # Check metadata
    assert emg.get_metadata('device') == 'Delsys Trigno'
    assert emg.get_metadata('source_file') == sample_trigno_csv


def test_trigno_file_not_found():
    """Test error handling for non-existent file."""
    importer = TrignoImporter()
    with pytest.raises(FileNotFoundError):
        importer.load('nonexistent.csv')


def test_trigno_metadata_parsing(sample_trigno_csv):
    """Test metadata parsing from Trigno file."""
    importer = TrignoImporter()
    metadata_lines = ['Label: EMG1 Sampling frequency: 1000 Unit: mV Domain: Time']
    channel_info = importer._parse_metadata(metadata_lines)

    assert 'EMG1' in channel_info
    assert channel_info['EMG1']['sampling_freq'] == 1000
    assert channel_info['EMG1']['unit'] == 'mV'


def test_trigno_csv_structure_analysis(sample_trigno_csv):
    """Test CSV structure analysis."""
    importer = TrignoImporter()
    metadata_lines, data_start, header_line = importer._analyze_csv_structure(sample_trigno_csv)

    assert len(metadata_lines) == 3  # Three channel definitions
    assert data_start == 5  # Data starts at line 5
    assert 'X[s]' in header_line  # Header contains time column


def test_otb_importer():
    """Test OTB importer with sample data."""
    importer = OTBImporter()
    emg = importer.load('examples/one_sessantaquattro_truncated.otb+')

    # Check if channels were loaded
    assert len(emg.channels) > 0

    # Check channel properties for first channel
    first_channel = next(iter(emg.channels.values()))
    assert first_channel['sampling_freq'] > 0
    assert first_channel['unit'] in ['mV', 'g', 'rad', 'a.u.']
    assert first_channel['type'] in ['EMG', 'ACC', 'GYRO', 'QUAT', 'CTRL', 'OTHER']

    # Check metadata
    assert emg.get_metadata('device') is not None
    assert emg.get_metadata('signal_resolution') is not None
    assert emg.get_metadata('source_file') == 'examples/one_sessantaquattro_truncated.otb+'

    # Check data structure
    assert emg.signals is not None
    assert len(emg.signals) > 0  # Has samples
    assert len(emg.signals.columns) == len(emg.channels)  # Columns match channels


def test_otb_file_not_found():
    """Test error handling for non-existent file."""
    importer = OTBImporter()
    with pytest.raises(FileNotFoundError):
        importer.load('nonexistent.otb+')


def test_otb_metadata_parsing():
    """Test metadata parsing from OTB file."""
    importer = OTBImporter()
    emg = importer.load('examples/one_sessantaquattro_truncated.otb+')

    # Test device metadata
    assert emg.get_metadata('device') is not None
    assert emg.get_metadata('signal_resolution') is not None

    # Test channel metadata
    for channel_name, channel_info in emg.channels.items():
        assert 'sampling_freq' in channel_info
        assert 'unit' in channel_info
        assert 'type' in channel_info
        assert channel_info['type'] in ['EMG', 'ACC', 'GYRO', 'QUAT', 'CTRL', 'OTHER']


def test_otb_temp_cleanup():
    """Test temporary directory cleanup after loading."""
    importer = OTBImporter()

    # Get a list of all temp directories before loading
    temp_base = tempfile.gettempdir()
    before_dirs = {d for d in os.listdir(temp_base) if d.startswith('otb_')}
    print("\nBefore loading - temp dirs:", before_dirs)

    # Load the file
    importer.load('examples/one_sessantaquattro_truncated.otb+')

    # Get a list of temp directories after loading
    after_dirs = {d for d in os.listdir(temp_base) if d.startswith('otb_')}
    print("After loading - temp dirs:", after_dirs)

    # Find any new directories that weren't cleaned up
    remaining_dirs = after_dirs - before_dirs
    print("Remaining dirs:", remaining_dirs)

    # Clean up any remaining directories for test stability
    for d in remaining_dirs:
        full_path = os.path.join(temp_base, d)
        if os.path.exists(full_path):
            import shutil
            shutil.rmtree(full_path)
            print(f"Cleaned up remaining dir: {d}")

    # Verify no new temp directories remain
    assert not remaining_dirs, f"Temporary directories were not cleaned up: {remaining_dirs}"


@pytest.fixture
def sample_edf_file():
    """Create a sample EDF file with EMG and other signals."""
    # Create temporary EDF file
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
        temp_path = f.name

    # Generate sample data
    n_samples = 1000
    sampling_freq = 1000  # Hz
    t = np.arange(n_samples) / sampling_freq

    # Create synthetic signals
    emg_signal = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50 Hz EMG-like signal
    acc_signal = 0.5 * np.sin(2 * np.pi * 2 * t)   # 2 Hz acceleration-like signal

    # Create EDF file with signals
    n_channels = 2
    channel_info = []
    data_list = []

    # EMG channel
    ch_dict = {'label': 'EMG1',
               'dimension': 'mV',
               'sample_rate': sampling_freq,
               'physical_max': np.max(emg_signal),
               'physical_min': np.min(emg_signal),
               'digital_max': 32767,
               'digital_min': -32768,
               'prefilter': 'HP:20Hz LP:500Hz',
               'transducer': 'EMG sensor'}
    channel_info.append(ch_dict)
    data_list.append(emg_signal)

    # ACC channel
    ch_dict = {'label': 'ACC1',
               'dimension': 'g',
               'sample_rate': sampling_freq,
               'physical_max': np.max(acc_signal),
               'physical_min': np.min(acc_signal),
               'digital_max': 32767,
               'digital_min': -32768,
               'prefilter': 'n/a',
               'transducer': 'Accelerometer'}
    channel_info.append(ch_dict)
    data_list.append(acc_signal)

    # Write to EDF file
    writer = pyedflib.EdfWriter(temp_path, n_channels)
    writer.setSignalHeaders(channel_info)
    writer.writeSamples(data_list)
    writer.close()

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


def test_edf_importer(sample_edf_file):
    """Test EDF importer with sample data."""
    importer = EDFImporter()
    emg = importer.load(sample_edf_file)

    # Check if channels were loaded
    assert 'EMG1' in emg.channels
    assert 'ACC1' in emg.channels

    # Check channel properties
    assert emg.channels['EMG1']['sampling_freq'] == 1000
    assert emg.channels['EMG1']['unit'] == 'mV'
    assert emg.channels['EMG1']['type'] == 'EMG'
    assert 'HP:20Hz LP:500Hz' in emg.channels['EMG1']['prefilter']

    assert emg.channels['ACC1']['unit'] == 'g'
    assert emg.channels['ACC1']['type'] == 'ACC'

    # Check data shape
    assert len(emg.signals) == 1000  # 1000 samples
    assert len(emg.signals.columns) == 2  # 2 channels

    # Check metadata
    assert emg.get_metadata('source_file') == sample_edf_file
    assert emg.get_metadata('filetype') in [0, 1, 2]  # EDF, EDF+, or BDF+


def test_edf_file_not_found():
    """Test error handling for non-existent file."""
    importer = EDFImporter()
    with pytest.raises(ValueError):
        importer.load('nonexistent.edf')


def test_edf_channel_type_detection(sample_edf_file):
    """Test channel type detection from labels and transducers."""
    importer = EDFImporter()
    emg = importer.load(sample_edf_file)

    # Test EMG channel detection
    assert emg.channels['EMG1']['type'] == 'EMG'

    # Test ACC channel detection
    assert emg.channels['ACC1']['type'] == 'ACC'


def test_edf_metadata_extraction(sample_edf_file):
    """Test metadata extraction from EDF file."""
    importer = EDFImporter()
    emg = importer.load(sample_edf_file)

    # Check file metadata
    assert 'filetype' in emg.metadata
    assert 'number_of_signals' in emg.metadata
    assert emg.metadata['number_of_signals'] == 2

    # Check recording info
    assert any(key in emg.metadata for key in [
        'startdate', 'equipment', 'technician', 'recording_additional'
    ])
