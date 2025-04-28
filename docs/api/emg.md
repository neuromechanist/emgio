# EMG Class API

The `EMG` class is the main class in EMGIO for working with EMG data. It encapsulates signals, channel information, and metadata, and provides methods for data manipulation and export.

## Class Documentation

::: emgio.core.emg.EMG
    options:
      show_root_heading: true
      show_source: true
      members: true
      show_submodules: true

## Key Methods Summary

### Data Loading

- `from_file()`: Load EMG data from a file (class method)
- `from_dataframe()`: Create EMG object from a pandas DataFrame (class method)

### Data Access

- `get_sampling_frequency()`: Get the sampling frequency of the data
- `get_n_samples()`: Get the number of samples
- `get_n_channels()`: Get the number of channels
- `get_duration()`: Get the duration of the recording in seconds
- `get_channel_types()`: Get the unique channel types
- `get_channels_by_type()`: Get channel names of a specific type

### Data Manipulation

- `select_channels()`: Create a new EMG object with selected channels
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
from emgio import EMG

# Load from file with automatic importer selection
emg = EMG.from_file("data.otb+")

# Load with explicit importer
emg = EMG.from_file("data.otb+", importer="otb")

# Load with explicit importer (.csv does not support automatic importer selection)
emg = EMG.from_file("data.csv", importer='trigno')
```

### Creating from DataFrame

```python
import pandas as pd
from emgio import EMG

# Create a DataFrame with EMG data
data = pd.DataFrame({
    'EMG1': [1, 2, 3, 4, 5],
    'EMG2': [5, 4, 3, 2, 1]
})

# Create channels dictionary
channels = {
    'EMG1': {
        'channel_type': 'EMG',
        'physical_dimension': 'µV',
        'sample_frequency': 1000
    },
    'EMG2': {
        'channel_type': 'EMG',
        'physical_dimension': 'µV',
        'sample_frequency': 1000
    }
}

# Create EMG object
emg = EMG.from_dataframe(data, channels=channels)
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
from emgio.analysis.verification import compare_signals, report_verification_results
from emgio.visualization.static import plot_comparison
import matplotlib.pyplot as plt

# Export with built-in verification
emg_original.to_edf('output', verify=True, verify_tolerance=0.001)

# Export and verify with custom channel mapping
channel_map = {'EMG1': 'CH1', 'EMG2': 'CH2'}
emg_original.to_edf('output', verify=True, verify_channel_map=channel_map)

# Export, verify, and generate verification plot
emg_original.to_edf('output', verify=True, verify_plot=True)

# Manual verification (alternative approach)
emg_original.to_edf('output')
emg_reloaded = EMG.from_file('output.edf')

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
