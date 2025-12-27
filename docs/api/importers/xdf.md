# XDF Importer

The `XDFImporter` class handles importing data from XDF (Extensible Data Format) files, the native format for Lab Streaming Layer (LSL) recordings. It supports multi-stream files with different sampling rates and data types.

## Class Documentation

::: emgio.importers.xdf
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Examples

### Basic Loading

```python
from emgio import EMG
from emgio.importers.xdf import XDFImporter

# Method 1: Using EMG.from_file (recommended)
emg = EMG.from_file('recording.xdf')

# Method 2: Using the importer directly
importer = XDFImporter()
emg = importer.load('recording.xdf')
```

### Exploring File Contents

Before loading, explore what streams are available:

```python
from emgio.importers.xdf import summarize_xdf

summary = summarize_xdf('recording.xdf')
print(summary)

# Output example:
# XDF File: recording.xdf
# ----------------------------------------
# Stream 1: MyEEG (EEG)
#   Channels: 8, Rate: 256.0 Hz
#   Samples: 15360, Duration: 60.0s
# Stream 2: MyEMG (EMG)
#   Channels: 2, Rate: 2048.0 Hz
#   Samples: 122880, Duration: 60.0s
# Stream 3: Markers (Markers)
#   Channels: 1, Rate: 0.0 Hz (irregular)
#   Samples: 10
```

### Selective Stream Loading

```python
# Load only specific stream types
emg = EMG.from_file('recording.xdf', stream_types=['EMG'])

# Load multiple types
emg = EMG.from_file('recording.xdf', stream_types=['EMG', 'EEG'])

# Load by stream name
emg = EMG.from_file('recording.xdf', stream_names=['MyEMGDevice'])

# Load by stream ID
emg = EMG.from_file('recording.xdf', stream_ids=[2])
```

### Setting Default Channel Type

```python
# For streams without explicit channel type metadata
emg = EMG.from_file('recording.xdf', default_channel_type='EMG')
```

## File Format Support

The XDF importer supports:

1. Single-stream and multi-stream XDF files
2. Compressed XDF files (.xdfz)
3. Numeric data types: float32, float64, int8, int16, int32, int64
4. Different sampling rates across streams (with resampling)
5. Channel labels from stream descriptors

## Stream Selection Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `stream_names` | `list[str]` | Filter by stream names (case-insensitive) |
| `stream_types` | `list[str]` | Filter by stream types (e.g., "EMG", "EEG") |
| `stream_ids` | `list[int]` | Filter by stream IDs |
| `default_channel_type` | `str` | Default type for channels without explicit type |

## Return Values

The `load()` method returns an `EMG` object with:

### Signals (pandas.DataFrame)
- Time-indexed signal data
- Channels as columns
- Resampled to common time base if multiple streams

### Channels (dict)
For each channel:
- `channel_type`: Inferred or default type
- `physical_dimension`: Unit (default "a.u.")
- `sample_frequency`: Effective sampling rate
- `stream_name`: Original stream name
- `stream_id`: Original stream ID

### Metadata (dict)
- `device`: "XDF"
- `source_file`: Path to the XDF file
- `stream_count`: Number of streams in file
- `stream_names`: List of all stream names
- `stream_types`: List of all stream types

## Helper Classes

### XDFSummary

Provides an overview of the XDF file:

```python
summary = summarize_xdf('recording.xdf')

# Access all streams
for stream in summary.streams:
    print(f"{stream.name}: {stream.channel_count} channels")

# Find streams by type
emg_streams = summary.get_streams_by_type('EMG')

# Find stream by name
stream = summary.get_stream_by_name('MyDevice')
```

### XDFStreamInfo

Contains metadata for a single stream:

- `stream_id`: Unique stream identifier
- `name`: Stream name
- `stream_type`: Stream type (EEG, EMG, etc.)
- `channel_count`: Number of channels
- `nominal_srate`: Declared sampling rate
- `effective_srate`: Actual measured sampling rate
- `channel_format`: Data format (float32, string, etc.)
- `source_id`: Source identifier
- `hostname`: Recording machine hostname
- `sample_count`: Number of samples
- `duration_seconds`: Recording duration
- `channel_labels`: List of channel names

## Implementation Notes

1. **String/Marker Streams:** Streams with `channel_format='string'` are excluded from signal loading but appear in summaries.

2. **Time Alignment:** When loading multiple streams, timestamps are aligned to start at 0.

3. **Resampling:** Multiple streams with different rates are resampled using linear interpolation to the highest rate.

4. **Channel Naming:** Channels are prefixed with stream name to avoid conflicts (e.g., "StreamName_ChannelLabel").
