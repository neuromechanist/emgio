# CSV Importer

The `CSVImporter` class is responsible for importing EMG and other physiological data from generic CSV files with flexible format detection and configuration options.

## Class Documentation

::: biosigio.importers.csv
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording
from biosigio.importers.csv import CSVImporter

# Method 1: Using Recording.from_file (recommended)
rec = Recording.from_file('data.csv', importer='csv')

# Method 2: Using the importer directly
rec = CSVImporter().load('data.csv', has_header=True, delimiter=',')
```

## Auto-Detection Features

The CSV importer includes several auto-detection capabilities:

- **Format Detection**: Recognizes specialized formats like Trigno CSV files
- **Delimiter Detection**: Identifies the most common delimiter (comma, tab, semicolon)
- **Header Detection**: Determines if the first row is a header based on content
- **Time Column Detection**: Looks for columns that might represent time

## Parameters

`CSVImporter().load(filepath, force_generic=False, **kwargs)` takes:

- **filepath (str)**: Path to the CSV file.
- **force_generic (bool, optional)**: Force using the generic CSV importer even
  if a specialized format (e.g., Trigno) is detected. When loading through
  `Recording.from_file(..., importer='csv', force_csv=True)`, the `force_csv`
  argument is forwarded as this `force_generic` flag.
- **kwargs**: Additional keyword arguments:
  - **sample_frequency (float, optional)**: Sampling frequency in Hz (required if no time column)
  - **has_header (bool, optional)**: Whether file has a header row (auto-detected if not specified)
  - **skiprows (int, optional)**: Number of rows to skip at beginning (auto-detected if not specified)
  - **delimiter (str, optional)**: Column delimiter (auto-detected if not specified)
  - **time_column (str or int, optional)**: Name or index of column to use as time index (auto-detected if not specified)
  - **columns (list, optional)**: List of column names or indices to include
  - **channel_names (list, optional)**: Custom names for channels
  - **channel_types (dict, optional)**: Dict mapping column names to channel types ('EMG', 'ACC', etc.)
  - **physical_dimensions (dict, optional)**: Dict mapping column names to physical dimensions
  - **metadata (dict, optional)**: Dict of additional metadata to include

## Return Value

The `load()` method returns a single `Recording` object with:

1. **signals**: signal data with channels as columns and time as index.
2. **channels**: per-channel information including:
   - `channel_type`: type of channel (EMG, ACC, GYRO, MISC, OTHER), inferred from the column name when not provided
   - `physical_dimension`: physical unit (e.g., 'µV', 'g')
   - `sample_frequency`: sampling rate in Hz
3. **metadata**: includes `source_file`, `file_format` ('CSV'), and any
   additional metadata passed via the `metadata` keyword argument.

## Implementation Details

The CSV importer uses pandas to:

1. Detect the format and structure of the CSV file
2. Extract time information if available or generate a time index based on sample frequency
3. Convert column data to appropriate formats
4. Apply channel labeling and typing based on provided information
5. Construct a pandas DataFrame with the signal data

## Examples

### Basic CSV with Headers

```python
# Load CSV with automatic format detection
rec = Recording.from_file('data.csv', importer='csv')
```

### Headerless CSV with Custom Names

```python
# Load headerless CSV with custom channel names
rec = Recording.from_file('data.csv', importer='csv',
                   has_header=False,
                   sample_frequency=1000,  # Required since no time column
                   channel_names=['EMG_L', 'EMG_R', 'ACC_X'])
```

### Setting Channel Types and Units

```python
# Specify channel types and physical dimensions
rec = Recording.from_file('data.csv', importer='csv',
                   channel_types={
                       'EMG1': 'EMG',
                       'EMG2': 'EMG',
                       'ACC1': 'ACC'
                   },
                   physical_dimensions={
                       'EMG1': 'mV',
                       'EMG2': 'mV',
                       'ACC1': 'g'
                   })
```
