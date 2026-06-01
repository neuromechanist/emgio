# OTB Importer

The `OTBImporter` class is responsible for importing EMG and other electrophysiological data from OT Bioelettronica's OTB+ files.

## Class Documentation

::: biosigio.importers.otb
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording
from biosigio.importers.otb import OTBImporter

# Method 1: Using Recording.from_file (recommended)
rec = Recording.from_file('data.otb+', importer='otb')

# Method 2: Using the importer directly
rec = OTBImporter().load('data.otb+')
```

## File Format Support

The OTB importer supports:
1. OTB/OTB+ archives from OT Bioelettronica devices (Sessantaquattro, Muovi, Quattrocento, etc.)
2. Multiple channel types (EMG, ACC, GYRO, QUAT, CTRL, OTHER)
3. 16-bit and 24-bit signal resolutions

The archive carries one device-wide sampling frequency, which is assigned to
every channel.

## Channel Type Mapping

The OTB importer identifies channel types from the OTB+ XML metadata (adapter
model, channel description/ID). The mapped types are:

- **EMG**: Electromyography channels (Due, Muovi, Sessantaquattro, Novecento, Quattro, Quattrocento adapters)
- **ACC**: Accelerometer channels
- **GYRO**: Gyroscope channels
- **QUAT**: Quaternion channels
- **CTRL**: Control channels
- **OTHER**: Anything that does not match the above

## Parameters

`OTBImporter().load(filepath)` takes:

- **filepath (str)**: Path to the OTB/OTB+ file.

## Return Value

The `load()` method returns a single `Recording` object with:

1. **signals**: signal data with channels as columns.
2. **channels**: per-channel information including:
   - `channel_type`: type of channel (EMG, ACC, GYRO, QUAT, CTRL, OTHER)
   - `physical_dimension`: physical unit (e.g., 'mV' for EMG, 'g' for ACC/GYRO, 'rad' for QUAT)
   - `sample_frequency`: device sampling rate in Hz
   - `prefilter`: pre-filtering string built from the adapter's HP/LP filter settings
3. **metadata**: fields parsed from the OTB+ archive, including:
   - `source_file`: input path
   - `device`: device name parsed from the XML header
   - `signal_resolution`: bit resolution of the signals (e.g., 16 or 24)

## Implementation Details

The OTB importer works by:

1. Extracting the OTB/OTB+ archive (a tar container) to a temporary directory.
2. Parsing the XML metadata file to extract device and per-channel information.
3. Reading the raw binary `.sig` signal data (reconstructing 24-bit samples when applicable).
4. Applying the device gain and reference-voltage scaling to convert raw values to physical units.
5. Adding each channel to the Recording with its inferred type and unit.