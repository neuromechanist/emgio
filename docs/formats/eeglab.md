# EEGLAB Format

EEGLAB is a MATLAB toolbox for processing EEG, MEG, and other electrophysiological data. biosigIO can import EEGLAB `.set` files to work with EMG data stored in this format.

## Format Description

EEGLAB `.set` files are MATLAB files that contain:

- Signal data in a matrix format
- Channel information (names, locations, types)
- Event markers
- Metadata about the recording
- Processing history

### Structure

A typical EEGLAB `.set` file contains these fields:

| Field | Description |
|-------|-------------|
| `data` | Signal data matrix (channels × time points) |
| `chanlocs` | Channel information (names, types, locations) |
| `srate` | Sampling rate in Hz |
| `xmin` | Time of first data point (seconds) |
| `xmax` | Time of last data point (seconds) |
| `times` | Time points vector |
| `event` | Event markers |
| `epoch` | Epoch information (if epoched) |
| `subject` | Subject identifier |
| `condition` | Condition name or description |

## Importer Implementation

The EEGLAB importer in biosigIO (`biosigio.importers.eeglab`) works by:

1. Loading the `.set` file using Python's `scipy.io.loadmat` (for MATLAB files)
2. Extracting the EEG structure with signal data and metadata
3. Converting channel information to biosigIO's channel format
4. Creating appropriate metadata dictionary
5. Handling both continuous and epoched data

## Code Example

```python
from biosigio import Recording

# Load data from EEGLAB .set file
emg = Recording.from_file('data.set', importer='eeglab')

# Print metadata
print(f"Subject: {emg.get_metadata('subject')}")
print(f"Condition: {emg.get_metadata('condition')}")
print(f"Sampling rate: {emg.get_metadata('srate')} Hz")

# Print channel information
print(f"Number of channels: {emg.get_n_channels()}")
channel_types = emg.get_channel_types()
print(f"Channel types: {channel_types}")

# Plot data
emg.plot_signals(time_range=(0, 5))
```

## Channel Type Mapping

EEGLAB doesn't always explicitly designate channel types. biosigIO's EEGLAB importer uses the following rules to assign channel types:

1. Channels with 'EMG' in the name are assigned type 'EMG'
2. Channels with 'EEG' in the name are assigned type 'EEG'
3. Channels with 'ACC' in the name are assigned type 'ACC'
4. Other channels are assigned type 'OTHER'

## Notes and Limitations

- biosigIO supports both MATLAB v7.3 and older format .set files
- Event markers are preserved in the metadata
- Channel locations (if available) are preserved in the channel information
- Some EEGLAB-specific information may not be fully preserved in the conversion
- Time information is properly handled to maintain accurate timing in the imported data 