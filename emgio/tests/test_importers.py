import pytest
import os
import tempfile
from ..importers.trigno import TrignoImporter


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
