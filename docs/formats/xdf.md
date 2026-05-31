# XDF Format

EMGIO supports loading data stored in the Extensible Data Format (XDF), the native format for Lab Streaming Layer (LSL) recordings. XDF files can contain multiple synchronized data streams from various devices, making them common in multi-modal experiments.

## File Structure

XDF files are self-contained binary files that can hold:

- **Multiple Streams:** Each stream represents a data source (e.g., EEG, EMG, motion capture, markers)
- **Stream Headers:** Metadata for each stream including name, type, channel count, sampling rate, and channel labels
- **Time Series Data:** The actual samples with timestamps
- **Stream Footers:** Summary information like sample counts and time ranges

Common file extensions:

- `.xdf`: Standard XDF file
- `.xdfz`: Compressed XDF file (gzip)

## Exploring XDF Files

Before loading, you can explore the contents of an XDF file using `summarize_xdf`:

```python
from emgio.importers.xdf import summarize_xdf

# Get a summary of all streams in the file
summary = summarize_xdf('recording.xdf')
print(summary)

# Find streams by type
emg_streams = summary.get_streams_by_type('EMG')
eeg_streams = summary.get_streams_by_type('EEG')

# Find a specific stream by name
stream = summary.get_stream_by_name('MyEMGDevice')
if stream:
    print(f"Channels: {stream.channel_count}")
    print(f"Sample rate: {stream.nominal_srate} Hz")
    print(f"Duration: {stream.duration_seconds:.1f}s")
```

## Loading Data

### Basic Loading

Load all numeric streams from an XDF file:

```python
from emgio.core.emg import Recording

# Load all numeric streams (excludes marker/string streams)
emg = Recording.from_file('recording.xdf')
```

### Selective Loading

XDF files often contain many streams. You can select specific streams:

```python
# Load only EMG streams
emg = Recording.from_file('recording.xdf', stream_types=['EMG'])

# Load specific stream types
emg = Recording.from_file('recording.xdf', stream_types=['EMG', 'EXG'])

# Load by stream name
emg = Recording.from_file('recording.xdf', stream_names=['MyEMGDevice'])

# Load by stream ID
emg = Recording.from_file('recording.xdf', stream_ids=[1, 3])
```

## Multi-Stream Handling

When loading multiple streams with different sampling rates, EMGIO:

1. Resamples all streams to a common time base
2. Uses the highest sampling rate among selected streams
3. Preserves channel labels with stream name prefixes for disambiguation

```python
# Load EEG and EMG together (different sample rates)
emg = Recording.from_file('recording.xdf', stream_types=['EEG', 'EMG'])

# Channels will be named like: "StreamName_ChannelLabel"
print(list(emg.channels.keys()))
# ['MyEEG_Fp1', 'MyEEG_Fp2', ..., 'MyEMG_Bicep', 'MyEMG_Tricep']
```

## Stream Types

Common LSL stream types you might encounter:

| Type | Description |
|------|-------------|
| `EEG` | Electroencephalography |
| `EMG` | Electromyography |
| `EXG` | Generic ExG (bio-electric signals) |
| `ECG` | Electrocardiography |
| `EOG` | Electrooculography |
| `Mocap` | Motion capture |
| `Markers` | Event markers (string data) |
| `Gaze` | Eye tracking |
| `Audio` | Audio signals |

## Channel Type Detection

The XDF importer attempts to infer channel types from:

1. Stream type in the XDF header
2. Channel labels (e.g., "EMG_Bicep" -> EMG)
3. A default type you can specify

```python
# Set default channel type for streams without explicit type info
emg = Recording.from_file('recording.xdf', default_channel_type='EMG')
```

## Metadata

Loaded XDF files include metadata such as:

- `device`: Set to "XDF"
- `source_file`: Path to the XDF file
- `stream_count`: Number of streams in the file
- `stream_names`: Names of all streams
- `stream_types`: Types of all streams

```python
emg = Recording.from_file('recording.xdf')
print(emg.get_metadata('stream_names'))
print(emg.get_metadata('stream_types'))
```

## Preserving LSL Timestamps

XDF files store per-sample timestamps from the Lab Streaming Layer clock. When exporting to formats like EDF that require regular sampling, these timestamps are normally lost during resampling.

To preserve the original LSL timestamps, use the `include_timestamps` option:

```python
# Load with timestamp preservation
emg = Recording.from_file('recording.xdf', include_timestamps=True)

# Each stream gets a timestamp channel named "{stream_name}_LSL_timestamps"
print(list(emg.channels.keys()))
# ['MyEMG_Ch1', 'MyEMG_Ch2', 'MyEMG_LSL_timestamps']
```

Timestamp channels:

- Contain the original LSL timestamps in seconds
- Are marked with `channel_type='MISC'` and `physical_dimension='s'`
- Are resampled along with the data when multiple streams have different rates
- Survive export to EDF/BDF for later synchronization analysis

This is useful for:

- Synchronizing with other data sources recorded with LSL
- Analyzing timing jitter in the original recording
- Post-hoc alignment with marker streams

## Marker Streams

Marker/event streams contain string data and are not loaded as signal channels. Use `summarize_xdf()` to inspect marker streams:

```python
summary = summarize_xdf('recording.xdf')
marker_stream = summary.get_stream_by_name('EventMarkers')
if marker_stream:
    print(f"Format: {marker_stream.channel_format}")  # 'string'
    print(f"Events: {marker_stream.sample_count}")
```

## Requirements

The XDF importer requires the `pyxdf` package. It is included as a core
dependency when installing EMGIO, so no additional installation is required.
