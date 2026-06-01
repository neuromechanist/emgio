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
emg = EEGLABImporter().load('data.set')
```

## File Format Support

The EEGLAB importer reads `.set` files with `scipy.io.loadmat`, so it supports:

1. Pre-v7.3 (non-HDF5) MATLAB `.set` files. MATLAB v7.3 / HDF5-format `.set`
   files are not supported.
2. Multiple channel types (EMG, EEG, ACC, etc.)
3. Event markers (stored in metadata)

## Channel Type Detection

The EEGLAB importer attempts to detect channel types based on:

1. Channel labels in the EEGLAB `chanlocs` structure
2. Channel type information if available
3. Naming conventions (e.g., channels with 'EMG' in the name are classified as 'EMG')

## Parameters

`EEGLABImporter().load(filepath)` takes:

- **filepath (str)**: Path to the EEGLAB `.set` file.

## Return Value

The `load()` method returns a single `Recording` object with:

1. **signals**: signal data with channels as columns.
2. **channels**: per-channel information including:
   - `channel_type`: type of channel (EMG, EEG, etc.)
   - `physical_dimension`: physical unit (defaults to `'uV'`)
   - `sample_frequency`: sampling rate in Hz
   - `X`/`Y`/`Z`: channel coordinates when available in `chanlocs`
3. **metadata**: fields parsed from the EEGLAB file, which may include:
   - `setname`, `filename`, `filepath`
   - `subject`, `group`, `condition`, `session`, `comments`
   - `srate`: sampling rate
   - `nbchan`, `trials`, `pnts`
   - `xmin`/`xmax`: time limits
   - `events`: list of event markers (stored under the `events` key)
   - `device`: set to `'EEGLAB'`

## Notes

- Event markers are preserved in metadata under the `events` key.
- Channel coordinates are preserved in the channel information when available.