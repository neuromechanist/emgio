# Trigno Format

The Delsys Trigno wireless EMG system exports data in a CSV format. This page describes the format and how EMGIO works with it.

## Format Description

Trigno CSV files typically have the following structure:

- A header section with metadata
- A data section with columns for each channel
- EMG and accelerometer data in separate columns

### Example File Structure

```
# Trigno Data File
# Date: 2023-05-01
# Time: 10:30:00
# Device: Trigno Wireless System
# ...additional metadata...

Time(s),EMG1,EMG2,EMG3,EMG4,ACC1_X,ACC1_Y,ACC1_Z,...
0.000,0.01,0.02,0.03,0.04,0.1,0.2,0.3,...
0.001,0.02,0.03,0.04,0.05,0.1,0.2,0.3,...
0.002,0.03,0.04,0.05,0.06,0.1,0.2,0.3,...
```

## Channel Types

EMGIO recognizes the following channel types in Trigno CSV files:

- **EMG** - Electromyography channels (typically in μV or mV)
- **ACC** - Accelerometer channels (typically in g)

## Importer Implementation

The Trigno importer in EMGIO (`emgio.importers.trigno`) processes these files by:

1. Reading the header section to extract metadata
2. Parsing the data section as a pandas DataFrame
3. Identifying channel types based on naming conventions
4. Setting appropriate units and sampling frequencies for each channel

## Code Example

```python
from emgio import EMG

# Load data from Trigno CSV file
emg = EMG.from_file('data.csv', importer='trigno')

# Print identified channel types
channel_types = emg.get_channel_types()
print(f"Identified channel types: {channel_types}")

# Get EMG channels
emg_channels = emg.get_channels_by_type('EMG')
print(f"EMG channels: {emg_channels}")

# Get accelerometer channels
acc_channels = emg.get_channels_by_type('ACC')
print(f"ACC channels: {acc_channels}")
```

## Notes and Limitations

- Trigno systems can have different sampling rates for EMG and accelerometer channels
- The importer automatically detects the sampling rate from the time column
- Column names may vary between different Trigno system versions
- Some metadata may be missing depending on the export settings 