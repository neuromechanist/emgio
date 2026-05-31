# Trigno Importer

The `TrignoImporter` class is responsible for importing EMG data from Delsys Trigno CSV files.

## Class Documentation

::: emgio.importers.trigno
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from emgio import Recording
from emgio.importers.trigno import TrignoImporter

# Method 1: Using Recording.from_file (recommended)
emg = Recording.from_file('data.csv', importer='trigno')

# Method 2: Using the importer directly
importer = TrignoImporter('data.csv')
signals, channels, metadata = importer.load()
emg = Recording(signals, channels, metadata)
```

## File Format Requirements

The importer expects a CSV file with:

1. A header section (optional) containing metadata lines starting with '#'
2. A data section with columns:
   - First column: Time in seconds
   - Remaining columns: Channel data

Example:
```
# Delsys Trigno EMG Data
# Recording Date: 2023-01-01
# Subject: S001
Time(s),EMG1,EMG2,EMG3,EMG4,ACC1_X,ACC1_Y,ACC1_Z
0.000,0.01,0.02,0.03,0.04,0.05,0.06,0.07
0.001,0.02,0.03,0.04,0.05,0.06,0.07,0.08
...
```

## Channel Type Detection

The Trigno importer automatically detects channel types based on channel names:

- Channels containing 'EMG' or 'emg' are classified as 'EMG'
- Channels containing 'ACC' or 'acc' are classified as 'ACC'
- Other channels are classified as 'OTHER'

## Parameters

- **file_path (str)**: Path to the Trigno CSV file
- **kwargs (dict)**: Additional keyword arguments
  - **header_rows (int, optional)**: Number of header rows to skip. If not provided, automatically detected.
  - **delimiter (str, optional)**: Column delimiter. Default is ','.
  - **channel_types (dict, optional)**: Manual mapping of channel names to types.

## Return Values

The `load()` method returns a tuple of:

1. **signals (pandas.DataFrame)**: Signal data with channels as columns
2. **channels (dict)**: Dictionary of channel information
3. **metadata (dict)**: Dictionary containing metadata from the header 