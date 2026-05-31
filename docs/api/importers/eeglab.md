# EEGLAB Importer

The `EEGLABImporter` class is responsible for importing EMG (and other biopotential) data from EEGLAB `.set` files.

## Class Documentation

::: biosigio.importers.eeglab
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording
from biosigio.importers.eeglab import EEGLABImporter

# Method 1: Using Recording.from_file (recommended)
emg = Recording.from_file('data.set', importer='eeglab')

# Method 2: Using the importer directly
importer = EEGLABImporter('data.set')
signals, channels, metadata = importer.load()
emg = Recording(signals, channels, metadata)
```

## File Format Support

The EEGLAB importer supports:

1. MATLAB `.set` files (version 7.3 and earlier)
2. Both continuous and epoched data
3. Multiple channel types (EMG, EEG, ACC, etc.)
4. Event markers and annotations

## Channel Type Detection

The EEGLAB importer attempts to detect channel types based on:

1. Channel labels in the EEGLAB `chanlocs` structure
2. Channel type information if available
3. Naming conventions (e.g., channels with 'EMG' in the name are classified as 'EMG')

## Parameters

- **file_path (str)**: Path to the EEGLAB .set file
- **kwargs (dict)**: Additional keyword arguments
  - **load_data (bool, optional)**: Whether to load the data or just metadata. Default is True.
  - **channel_types (dict, optional)**: Manual mapping of channel names to types.

## Return Values

The `load()` method returns a tuple of:

1. **signals (pandas.DataFrame)**: Signal data with channels as columns
2. **channels (dict)**: Dictionary of channel information including:
   - channel_type: Type of channel (EMG, EEG, etc.)
   - physical_dimension: Physical unit (e.g., 'µV')
   - sample_frequency: Sampling rate in Hz
   - coordinates: Channel coordinates if available

3. **metadata (dict)**: Dictionary containing metadata from the EEGLAB file, including:
   - subject: Subject identifier
   - session: Session identifier
   - condition: Condition/task information
   - srate: Sampling rate
   - xmin/xmax: Time limits
   - event: Event markers
   - epoch: Epoch information (if epoched)

## Notes

- The importer automatically handles both continuous and epoched data
- For epoched data, epochs are concatenated in the time dimension
- Event markers are preserved in the metadata
- Channel locations are preserved in the channel information when available 