# EDF Importer

The `EDFImporter` class is responsible for importing EMG and other physiological data from EDF (European Data Format) and BDF (BioSemi Data Format) files.

## Class Documentation

::: emgio.importers.edf
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from emgio import Recording
from emgio.importers.edf import EDFImporter

# Method 1: Using Recording.from_file (recommended)
emg = Recording.from_file('data.edf', importer='edf')  # Works for both .edf and .bdf

# Method 2: Using the importer directly
importer = EDFImporter('data.edf')
signals, channels, metadata = importer.load()
emg = Recording(signals, channels, metadata)
```

## File Format Support

The EDF importer supports:

1. EDF files (16-bit resolution)
2. BDF files (24-bit resolution)
3. EDF+ files with annotations
4. Different sampling rates for different channels
5. Time-stamped data

## Channel Type Detection

The EDF importer attempts to identify channel types based on:

1. Channel labels in the EDF header
2. Signal characteristics 
3. Common naming conventions (e.g., channels with 'EMG' in the name are classified as 'EMG')

## Parameters

- **file_path (str)**: Path to the EDF/BDF file
- **kwargs (dict)**: Additional keyword arguments
  - **load_data (bool, optional)**: Whether to load the data or just metadata. Default is True.
  - **channel_types (dict, optional)**: Manual mapping of channel names to types.
  - **start_time (float, optional)**: Start time in seconds for loading a subset of the data.
  - **end_time (float, optional)**: End time in seconds for loading a subset of the data.

## Return Values

The `load()` method returns a tuple of:

1. **signals (pandas.DataFrame)**: Signal data with channels as columns
2. **channels (dict)**: Dictionary of channel information including:
   - channel_type: Type of channel (EMG, EEG, etc.)
   - physical_dimension: Physical unit (e.g., 'µV', 'mV')
   - sample_frequency: Sampling rate in Hz
   - physical_min: Minimum physical value
   - physical_max: Maximum physical value
   - digital_min: Minimum digital value
   - digital_max: Maximum digital value

3. **metadata (dict)**: Dictionary containing metadata from the EDF file, including:
   - subject: Subject identifier (from EDF patient field)
   - recording_date: Recording date
   - start_time: Start time of the recording
   - duration: Duration of the recording in seconds
   - file_type: 'EDF' or 'BDF'
   - annotations: Any annotations found in the file

## Implementation Details

The EDF importer uses the `pyedflib` package to:

1. Read the EDF/BDF file header to extract metadata
2. Extract channel information and convert to EMGIO's format
3. Load the signal data, applying appropriate scaling
4. Handle annotations if present
5. Convert the data to a pandas DataFrame for use with EMGIO

## Working with Annotations

EDF+ files can contain annotations that mark specific events or segments in the data. The importer preserves these annotations in the metadata dictionary:

```python
# Load EDF+ file with annotations
emg = Recording.from_file('data.edf+', importer='edf')

# Access annotations
annotations = emg.get_metadata('annotations')
if annotations:
    for annotation in annotations:
        onset, duration, description = annotation
        print(f"Event: {description} at {onset}s, duration: {duration}s")
``` 