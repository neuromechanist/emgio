# EEGLAB Format

EEGLAB is a MATLAB toolbox for processing EEG, MEG, and other electrophysiological data. biosigIO can import EEGLAB `.set` files to work with biosignal data stored in this format.

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

1. Loading the `.set` file using `scipy.io.loadmat` (pre-v7.3, non-HDF5 MAT-files only)
2. Extracting the EEG structure with signal data and metadata. When the matrix is
   stored in a separate float32 `.fdt` file (EEGLAB's default for large
   recordings, where `EEG.data` holds the `.fdt` filename), the sibling `.fdt`
   next to the `.set` is loaded and reshaped to `(nbchan, pnts * trials)`
3. Converting channel information to biosigIO's channel format
4. Creating appropriate metadata dictionary
5. Loading event markers into the recording's `events` table (onset/duration in seconds)

## Code Example

```python
from biosigio import Recording

# Load data from EEGLAB .set file
rec = Recording.from_file('data.set', importer='eeglab')

# Print metadata
print(f"Subject: {rec.get_metadata('subject')}")
print(f"Condition: {rec.get_metadata('condition')}")
print(f"Sampling rate: {rec.get_metadata('srate')} Hz")

# Print channel information
print(f"Number of channels: {rec.get_n_channels()}")
channel_types = rec.get_channel_types()
print(f"Channel types: {channel_types}")

# Plot data
rec.plot_signals(time_range=(0, 5))
```

## Channel Type Mapping

EEGLAB doesn't always explicitly designate channel types. biosigIO's EEGLAB importer uses the following rules to assign channel types:

1. Channels with 'EMG' in the name are assigned type 'EMG'
2. Channels with 'EEG' in the name are assigned type 'EEG'
3. Channels with 'ACC' in the name are assigned type 'ACC'
4. Other channels are assigned type 'OTHER'

## Notes and Limitations

- Only pre-v7.3 (non-HDF5) `.set` files are supported, via `scipy.io.loadmat`; MATLAB v7.3/HDF5 `.set` files are not supported
- Signal data stored inline in the `.set` or in a separate float32 `.fdt` file is supported; the sibling `.fdt` is resolved by the `.set` path (so BIDS-renamed files load even though `EEG.data` keeps the original `.fdt` name)
- Event markers are loaded into the `events` table (`rec.events`): EEGLAB event latency/duration (samples) are converted to onset/duration in seconds and the event `type` becomes the description
- Channel locations (if available) are preserved in the channel information
- Some EEGLAB-specific information may not be fully preserved in the conversion
- Time information is properly handled to maintain accurate timing in the imported data 