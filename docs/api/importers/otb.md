# OTB Importer

The `OTBImporter` class is responsible for importing EMG and other electrophysiological data from OT Bioelettronica's OTB+ files.

## Class Documentation

::: emgio.importers.otb
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from emgio import Recording
from emgio.importers.otb import OTBImporter

# Method 1: Using Recording.from_file (recommended)
emg = Recording.from_file('data.otb+', importer='otb')

# Method 2: Using the importer directly
importer = OTBImporter('data.otb+')
signals, channels, metadata = importer.load()
emg = Recording(signals, channels, metadata)
```

## File Format Support

The OTB importer supports:
1. Binary OTB+ files from OT Bioelettronica devices (SESSANTAQUATTRO, MUOVI, etc.)
2. Multiple channel types (EMG, ACC, TRIG, AUX, IMUX)
3. Different sampling rates for different channel types
4. Various bit resolutions (8-bit, 12-bit, 16-bit, etc.)

## Channel Type Mapping

The OTB importer identifies channel types based on the OTB+ file header information. Common channel types include:

- **EMG**: Electromyography channels
- **ACC**: Accelerometer channels
- **TRIG**: Trigger channels
- **AUX**: Auxiliary channels 
- **IMUX**: Input multiplexer channels

## Parameters

- **file_path (str)**: Path to the OTB+ file
- **kwargs (dict)**: Additional keyword arguments
  - **use_matlab (bool, optional)**: Whether to use MATLAB for import (requires MATLAB runtime). Default is False.
  - **channel_types (dict, optional)**: Manual mapping of channel names to types.

## Return Values

The `load()` method returns a tuple of:

1. **signals (pandas.DataFrame)**: Signal data with channels as columns
2. **channels (dict)**: Dictionary of channel information including:
   - channel_type: Type of channel (EMG, ACC, etc.)
   - physical_dimension: Physical unit (e.g., 'µV' for EMG, 'g' for ACC)
   - sample_frequency: Sampling rate in Hz
   - bit_resolution: Bit resolution of the channel
   - calibration_factor: Calibration factor for converting raw to physical values

3. **metadata (dict)**: Dictionary containing metadata from the OTB+ file, including:
   - device: Device name (e.g., 'SESSANTAQUATTRO')
   - recording_date: Recording date if available
   - signal_resolution: Bit resolution of the signals
   - device_settings: Additional device-specific settings

## Implementation Details

The OTB importer works by:

1. Reading the binary header information from the OTB+ file
2. Parsing the header to extract metadata about channels and device settings
3. Loading the raw signal data from the binary file
4. Applying calibration factors to convert raw values to physical units
5. Organizing the data into channel types and creating the appropriate metadata

## Dependencies

- The OTB importer includes a pure Python implementation for parsing OTB+ files
- It can optionally use the OT Bioelettronica MATLAB functions for import if MATLAB is available 