# XDF Example

This example demonstrates how to work with XDF files from Lab Streaming Layer (LSL) recordings, including exploring multi-stream files and selective loading.

## Exploring XDF Contents

XDF files often contain multiple streams. Before loading, explore what's available:

```python
from emgio.importers.xdf import summarize_xdf

# Summarize all streams in the file
summary = summarize_xdf('examples/multi_stream_test.xdf')
print(summary)
```

Output:
```
XDF File: examples/multi_stream_test.xdf
----------------------------------------
Stream 1: TestEEG (EEG)
  Channels: 8, Rate: 256.0 Hz
  Samples: 1280, Duration: 5.0s
  Labels: EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8
Stream 2: TestEMG (EMG)
  Channels: 2, Rate: 2048.0 Hz
  Samples: 10240, Duration: 5.0s
  Labels: EMG_L, EMG_R
Stream 3: TestMocap (Mocap)
  Channels: 6, Rate: 120.0 Hz
  Samples: 600, Duration: 5.0s
  Labels: Marker1_X, Marker1_Y, Marker1_Z, Marker2_X, Marker2_Y, Marker2_Z
Stream 4: TestMarkers (Markers)
  Channels: 1, Rate: 0.0 Hz (irregular)
  Samples: 5
```

## Finding Specific Streams

```python
# Find all EMG streams
emg_streams = summary.get_streams_by_type('EMG')
for stream in emg_streams:
    print(f"Found EMG stream: {stream.name} with {stream.channel_count} channels")

# Find a specific stream by name
mocap = summary.get_stream_by_name('TestMocap')
if mocap:
    print(f"Mocap rate: {mocap.nominal_srate} Hz")
    print(f"Mocap channels: {mocap.channel_labels}")
```

## Loading All Numeric Data

```python
from emgio import EMG

# Load all numeric streams (EEG, EMG, Mocap - excludes Markers)
emg = EMG.from_file('examples/multi_stream_test.xdf')

print(f"Total channels: {len(emg.channels)}")
print(f"Channel names: {list(emg.channels.keys())}")
```

## Selective Stream Loading

### Load by Stream Type

```python
# Load only EMG data
emg_data = EMG.from_file('examples/multi_stream_test.xdf', stream_types=['EMG'])
print(f"EMG channels: {list(emg_data.channels.keys())}")
# Output: ['EMG_L', 'EMG_R']

# Load EEG and EMG together
combined = EMG.from_file('examples/multi_stream_test.xdf', stream_types=['EEG', 'EMG'])
print(f"Combined channels: {len(combined.channels)}")
# Output: 10 (8 EEG + 2 EMG)
```

### Load by Stream Name

```python
# Load specific streams by name
emg_data = EMG.from_file('examples/multi_stream_test.xdf', stream_names=['TestEMG'])
```

## Working with Multi-Rate Data

When loading streams with different sampling rates, they're resampled to a common time base:

```python
# Load EEG (256 Hz) and EMG (2048 Hz)
combined = EMG.from_file('examples/multi_stream_test.xdf', stream_types=['EEG', 'EMG'])

# Check the resulting sample rate (will be the highest: 2048 Hz)
first_channel = list(combined.channels.keys())[0]
print(f"Sample rate: {combined.channels[first_channel]['sample_frequency']} Hz")
```

## Exporting to EDF

After loading, export to EDF/BDF format:

```python
# Load EMG streams
emg = EMG.from_file('examples/multi_stream_test.xdf', stream_types=['EMG'])

# Export to EDF
emg.to_edf('output_emg.edf')

# Verify the export
emg_reloaded = EMG.from_file('output_emg.edf')
print(f"Exported channels: {list(emg_reloaded.channels.keys())}")
```

## Complete Workflow Example

```python
from emgio import EMG
from emgio.importers.xdf import summarize_xdf

# 1. Explore the file
summary = summarize_xdf('recording.xdf')
print(summary)

# 2. Identify streams of interest
emg_streams = summary.get_streams_by_type('EMG')
print(f"Found {len(emg_streams)} EMG streams")

# 3. Load selected data
emg = EMG.from_file('recording.xdf', stream_types=['EMG'])

# 4. Check loaded data
print(f"Channels: {list(emg.channels.keys())}")
print(f"Duration: {emg.signals.index[-1]:.1f}s")
print(f"Sample rate: {emg.channels[list(emg.channels.keys())[0]]['sample_frequency']} Hz")

# 5. Plot signals
emg.plot_signals(time_range=(0, 5))

# 6. Export
emg.to_edf('emg_export.edf', verify=True)
```

## Notes

- Marker streams (string data) are not loaded as signal channels
- When multiple streams are loaded, channels are prefixed with stream names
- Time indices are normalized to start at 0
- The `pyxdf` package is used internally for reading XDF files
