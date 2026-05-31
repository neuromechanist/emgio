# Trigno Examples

This page demonstrates how to work with EMG data from the Delsys Trigno wireless EMG system. Trigno data is typically exported as CSV files with a specific format that biosigIO can parse.

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
print(f"Sampling frequency: {emg.get_sampling_frequency()} Hz")
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

# Plot EMG and ACC signals separately
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
emg_only.plot_signals(time_range=(0, 5), title="EMG Signals", ax=plt.gca())

plt.subplot(2, 1, 2)
acc_only.plot_signals(time_range=(0, 5), title="Accelerometer Signals", ax=plt.gca())

plt.tight_layout()
plt.show()
```

## Exporting Trigno Data to EDF/BDF

You can export the Trigno data to EDF or BDF format for use with other software:

```python
# Export all channels
output_path = 'trigno_all_channels'
emg.to_edf(output_path)  # Format (EDF/BDF) will be selected automatically
print(f"Exported to: {output_path}")

# Export only EMG channels
emg_only = emg.select_channels(channel_type='EMG')
output_path = 'trigno_emg_only'
emg_only.to_edf(output_path)
print(f"Exported EMG channels to: {output_path}")
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

# Plot channels from both sessions
channel_to_compare = 'EMG1'  # Replace with your channel name

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
session1.plot_signals([channel_to_compare], time_range=(0, 5), 
                     title=f"Session 1: {session1.get_metadata('condition')}", 
                     ax=plt.gca())

plt.subplot(2, 1, 2)
session2.plot_signals([channel_to_compare], time_range=(0, 5), 
                     title=f"Session 2: {session2.get_metadata('condition')}", 
                     ax=plt.gca())

plt.tight_layout()
plt.show()
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
emg.set_metadata('sampling_rate', emg.get_sampling_frequency())

# 3. Select only EMG channels
emg_only = emg.select_channels(channel_type='EMG')

# 4. Plot raw signals
plt.figure(figsize=(12, 6))
emg_only.plot_signals(time_range=(0, 10), 
                     title="Raw EMG Signals", 
                     ax=plt.gca())
plt.tight_layout()
plt.savefig('raw_emg.png')

# 5. Export to EDF/BDF
output_path = 'processed_emg'
emg_only.to_edf(output_path)

print(f"Processing complete. Data exported to: {output_path}")
```

This example demonstrates a complete workflow from loading Trigno data to selecting channels, visualizing, and exporting to EDF/BDF format. 