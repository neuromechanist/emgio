# Trigno Examples

This page demonstrates how to work with EMG data from the Delsys Trigno wireless EMG system. Trigno data is exported as CSV files with a metadata preamble followed by a data table. The importer reads per-channel `Label: ... Sampling frequency: ... Unit: ...` lines from the preamble, then locates the data header line (the one containing the `X[s]` time column) and reads the samples below it. Per-channel sampling frequency comes from each channel's `Sampling frequency:` field, so EMG and accelerometer (ACC) channels can carry different rates.

## Basic Trigno Example

```python
import os
from biosigio import Recording
import matplotlib.pyplot as plt

# Load data from Trigno CSV file
data_path = 'path_to_your_trigno_data.csv'
emg = Recording.from_file(data_path, importer='trigno')

# Print information about the loaded data
print(f"Number of channels: {emg.get_n_channels()}")
print(f"Number of samples: {emg.get_n_samples()}")
# Trigno recordings are often mixed-rate (EMG vs ACC/GYRO), so a single overall
# sampling frequency may not exist. get_sampling_frequency() raises ValueError in
# that case; read the per-channel rate from channels[ch]['sample_frequency'] instead.
try:
    print(f"Sampling frequency: {emg.get_sampling_frequency()} Hz")
except ValueError:
    print("Mixed per-channel sampling rates (see per-channel rates below)")
print(f"Recording duration: {emg.get_duration()} seconds")

# List available channels
print("\nAvailable channels:")
for ch_name, ch_info in emg.channels.items():
    print(f"- {ch_name} ({ch_info['channel_type']})")
    print(f"  Sampling rate: {ch_info['sample_frequency']} Hz")
    print(f"  Dimension: {ch_info['physical_dimension']}")

# Plot EMG signals
emg.plot_signals(time_range=(0, 5), title="Raw EMG Signals")
plt.tight_layout()
plt.show()
```

## Separating EMG and ACC Channels

Trigno systems often record both EMG and accelerometer data. You can separate these channels for analysis:

```python
# Get EMG channels
emg_channels = emg.get_channels_by_type('EMG')
print(f"EMG channels: {emg_channels}")

# Create EMG-only object
emg_only = emg.select_channels(emg_channels)

# Get accelerometer channels
acc_channels = emg.get_channels_by_type('ACC')
print(f"ACC channels: {acc_channels}")

# Create ACC-only object
acc_only = emg.select_channels(acc_channels)

# Plot EMG and ACC signals separately (each call manages its own figure)
emg_only.plot_signals(time_range=(0, 5), title="EMG Signals")
acc_only.plot_signals(time_range=(0, 5), title="Accelerometer Signals")
```

## Exporting Trigno Data to EDF/BDF

You can export the Trigno data to EDF or BDF format for use with other software.
EDF/BDF requires a single sampling rate across all channels, so a mixed-rate
Trigno recording (EMG vs ACC/GYRO) must be split by channel type (or resampled to
a common rate) before export; exporting a mixed-rate recording raises a
`ValueError`.

```python
# Export only EMG channels (a single-rate subset)
emg_only = emg.select_channels(channel_type='EMG')
output_path = 'trigno_emg_only'
emg_only.to_edf(output_path)  # Format (EDF/BDF) will be selected automatically
print(f"Exported EMG channels to: {output_path}")

# Accelerometer channels have a different rate, so export them separately
acc_only = emg.select_channels(channel_type='ACC')
acc_only.to_edf('trigno_acc_only')
print("Exported ACC channels to: trigno_acc_only")
```

## Working with Multiple Trigno Recordings

If you have multiple recordings from the same experiment, you can load them separately and compare:

```python
# Load two recordings
session1 = Recording.from_file('session1.csv', importer='trigno')
session2 = Recording.from_file('session2.csv', importer='trigno')

# Add metadata to distinguish them
session1.set_metadata('session', '1')
session1.set_metadata('condition', 'rest')

session2.set_metadata('session', '2')
session2.set_metadata('condition', 'active')

# Plot channels from both sessions (each call manages its own figure)
channel_to_compare = 'EMG1'  # Replace with your channel name

session1.plot_signals([channel_to_compare], time_range=(0, 5),
                     title=f"Session 1: {session1.get_metadata('condition')}")

session2.plot_signals([channel_to_compare], time_range=(0, 5),
                     title=f"Session 2: {session2.get_metadata('condition')}")
```

## Complete Trigno Workflow Example

Here's a complete workflow using Trigno data:

```python
import os
from biosigio import Recording
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
data_path = 'trigno_recording.csv'
emg = Recording.from_file(data_path, importer='trigno')

# 2. Add metadata
emg.set_metadata('subject', 'S001')
emg.set_metadata('condition', 'reaching')

# 3. Select only EMG channels
emg_only = emg.select_channels(channel_type='EMG')

# The EMG-only subset has a single rate, so get_sampling_frequency() is safe here
# (the full mixed-rate recording would raise ValueError).
emg_only.set_metadata('sampling_rate', emg_only.get_sampling_frequency())

# 4. Plot raw signals (plot_signals manages its own figure). Pass show=False so
# the figure stays open for savefig; plot_signals calls plt.show() by default.
emg_only.plot_signals(time_range=(0, 10),
                     title="Raw EMG Signals", show=False)
plt.savefig('raw_emg.png')
plt.show()

# 5. Export to EDF/BDF
output_path = 'processed_emg'
emg_only.to_edf(output_path)

print(f"Processing complete. Data exported to: {output_path}")
```

This example demonstrates a complete workflow from loading Trigno data to selecting channels, visualizing, and exporting to EDF/BDF format. 