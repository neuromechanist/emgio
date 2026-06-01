# Recording Class API

The `Recording` class is the main class in biosigIO for working with biosignal data. It encapsulates signals, channel information, and metadata, and provides methods for data manipulation and export.

## Class Documentation

::: biosigio.core.emg.Recording
    handler: python
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - from_file
        - add_channel
        - add_event
        - select_channels
        - get_channel_types
        - get_channels_by_type
        - set_metadata
        - get_metadata
        - to_edf
        - plot_signals
      members_order: source
      show_object_full_path: false
      heading_level: 2
      show_bases: false
      docstring_options:
        ignore_init_summary: true

## Attributes

Details about the main attributes:

### `signals`

`pandas.DataFrame`: Contains the raw signal data. The index is typically time in seconds (if available from the source file or calculated), and columns represent the different channels.

### `metadata`

`dict`: A dictionary holding metadata about the recording session (e.g., subject ID, recording date, device info). Keys and values depend on the source file.

### `channels`

`dict`: A dictionary where keys are channel labels (strings) and values are dictionaries containing channel-specific information (e.g., `sample_frequency`, `physical_dimension`, `channel_type`, `prefilter`).

### `events`

`pandas.DataFrame`: Contains time-stamped annotations or events loaded from the file (e.g., from EDF+ or WFDB annotations) or added manually. It has the columns `onset`, `duration`, and `description`.

## Key Methods Summary

### Data Loading

- `from_file()`: Load biosignal data from a file (class method)
- `add_channel()`: Append a channel (data + metadata) to a Recording

### Data Access

- `get_sampling_frequency()`: Get the sampling frequency of the data
- `get_n_samples()`: Get the number of samples
- `get_n_channels()`: Get the number of channels
- `get_duration()`: Get the duration of the recording in seconds
- `get_channel_types()`: Get the unique channel types
- `get_channels_by_type()`: Get channel names of a specific type

### Data Manipulation

- `select_channels()`: Create a new Recording object with selected channels
- `set_metadata()`: Set a single metadata field
- `get_metadata()`: Get a single metadata field
- `has_metadata()`: Check if a metadata field exists

### Visualization

- `plot_signals()`: Plot EMG signals with customizable options

### Export

- `to_edf()`: Export data to EDF/BDF format with optional verification

## Usage Examples

### Loading Data

```python
from biosigio import Recording

# Load from file with automatic importer selection
emg = Recording.from_file("data.otb+")

# Load with explicit importer
emg = Recording.from_file("data.otb+", importer="otb")

# Generic CSV/TXT supports automatic importer selection, but a Delsys Trigno
# export is best loaded with its dedicated importer
emg = Recording.from_file("data.csv", importer='trigno')
```

### Building a Recording Programmatically

```python
import numpy as np
from biosigio import Recording

# Start from an empty Recording and add channels
emg = Recording()
emg.add_channel(
    label='EMG1',
    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    sample_frequency=1000,
    physical_dimension='µV',
    channel_type='EMG',
)
emg.add_channel(
    label='EMG2',
    data=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
    sample_frequency=1000,
    physical_dimension='µV',
    channel_type='EMG',
)
```

### Selecting Channels

```python
# Select by channel names
subset = emg.select_channels(['EMG1', 'EMG2'])

# Select by channel type
emg_only = emg.select_channels(channel_type='EMG')
```

### Metadata Handling

```python
# Set metadata
emg.set_metadata('subject', 'S001')

# Get metadata
subject = emg.get_metadata('subject')

# Check if metadata exists
if emg.has_metadata('condition'):
    condition = emg.get_metadata('condition')
```

### Plotting

```python
# Plot all channels
emg.plot_signals()

# Plot specific channels with time range
emg.plot_signals(['EMG1', 'EMG2'], time_range=(0, 5))

# Customize plot
emg.plot_signals(
    channels=['EMG1', 'EMG2'],
    time_range=(0, 5),
    title='EMG Signals',
    grid=True,
    detrend=False,
    offset_scale=0.8
)
```

### Verification

```python
from biosigio.analysis.verification import compare_signals, report_verification_results
from biosigio.visualization.static import plot_comparison
import matplotlib.pyplot as plt

# Export with built-in verification
emg_original.to_edf('output', verify=True, verify_tolerance=0.001)

# Export and verify with custom channel mapping
channel_map = {'EMG1': 'CH1', 'EMG2': 'CH2'}
emg_original.to_edf('output', verify=True, verify_channel_map=channel_map)

# Export, verify, and generate verification plot
emg_original.to_edf('output', verify=True, verify_plot=True)

# Manual verification (alternative approach). With the default format='auto',
# to_edf may write either output.edf or output.bdf; reload the path it wrote.
emg_original.to_edf('output')
emg_reloaded = Recording.from_file('output.edf')  # or 'output.bdf' if BDF was selected

# Compare signals
results = compare_signals(emg_original, emg_reloaded, tolerance=0.001)
is_identical = report_verification_results(results, verify_tolerance=0.001)

# Plot comparison for visual verification
plot_comparison(emg_original, emg_reloaded, channels=['EMG1', 'EMG2'])
plt.show()
```

### Exporting

```python
# Export with automatic format selection
emg.to_edf('output')

# Force EDF format
emg.to_edf('output', format='edf')

# Control format selection method
emg.to_edf('output', method='svd')
```
