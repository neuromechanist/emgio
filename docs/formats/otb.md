# OTB Format

OTB/OTB+ is a data format developed by OT Bioelettronica for storing multi-channel electrophysiological data. EMGIO provides support for importing data from OTB+ files.

## Format Description

OTB files typically have a `.otb+` extension and are binary files containing:

- Header information with metadata
- Multi-channel signal data
- Potentially multiple signals types (EMG, ACC, etc.)
- Optional event markers

### Structure

The OTB+ format consists of:

1. A header section containing information about:
   - Device type and configuration
   - Number of channels
   - Sampling frequency
   - Bit resolution
   - Calibration values
   
2. A data section containing:
   - Raw signal data for all channels
   - Potentially interleaved channel types (EMG, accelerometer, etc.)

## Importer Implementation

The OTB importer in EMGIO (`emgio.importers.otb`) works by:

1. Reading the binary OTB+ file header to extract metadata
2. Parsing the data section and arranging it into a structured format
3. Identifying different channel types based on the header information
4. Setting appropriate units and calibration values
5. Creating a properly formatted channel dictionary

## Code Example

```python
from emgio import Recording
import matplotlib.pyplot as plt

# Load data from OTB file
emg = Recording.from_file('data.otb+', importer='otb')

# Print device information
print(f"Device: {emg.get_metadata('device')}")
print(f"Resolution: {emg.get_metadata('signal_resolution')} bits")

# Summarize channel types
channel_types = emg.get_channel_types()
print(f"Channel types: {channel_types}")

for ch_type in channel_types:
    channels = emg.get_channels_by_type(ch_type)
    print(f"{ch_type} channels: {len(channels)}")

# Select only EMG channels
emg_only = emg.select_channels(channel_type='EMG')

# Plot EMG channels
emg_only.plot_signals(time_range=(0, 5))
plt.tight_layout()
plt.show()
```

## Channel Types

OTB files can contain various channel types:

- **EMG** - Electromyography channels
- **ACC** - Accelerometer channels 
- **IMUX** - Input multiplexer channels (for OT Bioelettronica devices)
- **AUX** - Auxiliary input channels
- **TRIG** - Trigger or event marker channels

## Notes and Limitations

- OTB files can come from different devices (SESSANTAQUATTRO, MUOVI, etc.)
- Channel naming conventions may vary between devices
- Sampling rates can be different for different channel types
- Some OTB+ files may contain additional proprietary information
- The EMGIO importer preserves as much metadata as possible from the original file

## References

- OT Bioelettronica: [https://otbioelettronica.it/](https://otbioelettronica.it/)
- The OTB importer in EMGIO is based on the MATLAB import functions provided by OT Bioelettronica 