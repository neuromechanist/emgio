# CSV Importer

The `CSVImporter` class is responsible for importing EMG and other physiological data from generic CSV files with flexible format detection and configuration options.

## Class Documentation

::: emgio.importers.csv
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from emgio import EMG
from emgio.importers.csv import CSVImporter

# Method 1: Using EMG.from_file (recommended)
emg = EMG.from_file('data.csv', importer='csv')

# Method 2: Using the importer directly
importer = CSVImporter('data.csv', has_header=True, delimiter=',')
signals, channels, metadata = importer.load()
emg = EMG(signals, channels, metadata)
```

## Auto-Detection Features

The CSV importer includes several auto-detection capabilities:

- **Format Detection**: Recognizes specialized formats like Trigno CSV files
- **Delimiter Detection**: Identifies the most common delimiter (comma, tab, semicolon)
- **Header Detection**: Determines if the first row is a header based on content
- **Time Column Detection**: Looks for columns that might represent time

## Parameters

- **file_path (str)**: Path to the CSV file
- **kwargs (dict)**: Additional keyword arguments
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
  - **force_csv (bool, optional)**: Force using generic CSV importer even if specialized format is detected

## Return Values

The `load()` method returns a tuple of:

1. **signals (pandas.DataFrame)**: Signal data with channels as columns and time as index
2. **channels (dict)**: Dictionary of channel information including:
   - channel_type: Type of channel (EMG, EEG, etc.)
   - physical_dimension: Physical unit (e.g., 'mV', 'g')
   - sample_frequency: Sampling rate in Hz

3. **metadata (dict)**: Dictionary containing metadata from the file and any additional provided metadata

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
emg = EMG.from_file('data.csv', importer='csv')
```

### Headerless CSV with Custom Names

```python
# Load headerless CSV with custom channel names
emg = EMG.from_file('data.csv', importer='csv',
                   has_header=False,
                   sample_frequency=1000,  # Required since no time column
                   channel_names=['EMG_L', 'EMG_R', 'ACC_X'])
```

### Setting Channel Types and Units

```python
# Specify channel types and physical dimensions
emg = EMG.from_file('data.csv', importer='csv',
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
