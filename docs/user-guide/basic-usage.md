# Basic Usage

EMGIO provides a unified interface for working with EMG data from various systems. This page covers the core functionality that's common across all supported data formats.

## Loading Data

The main entry point for loading data is the `EMG.from_file()` method. This method automatically determines the correct importer based on the file extension, or you can specify the importer explicitly:

```python
from emgio import EMG

# Automatic importer selection based on file extension
emg = EMG.from_file('data.csv')  # Will use Trigno importer for CSV files

# Explicit importer selection
emg = EMG.from_file('data.csv', importer='trigno')
emg = EMG.from_file('data.set', importer='eeglab')
emg = EMG.from_file('data.otb+', importer='otb')
emg = EMG.from_file('data.edf', importer='edf')
```

## Accessing Data

Once you've loaded data, you can access the signals and metadata:

```python
# Access raw signal data (returns a pandas DataFrame)
signals = emg.signals

# Get information about channels
channels = emg.channels

# Get sampling frequency
fs = emg.get_sampling_frequency()

# Get the number of samples
n_samples = emg.get_n_samples()

# Get the number of channels
n_channels = emg.get_n_channels()
```

## Plotting Signals

EMGIO provides methods to visualize the EMG signals:

```python
# Plot all channels
emg.plot_signals()

# Plot specific channels
emg.plot_signals(['EMG1', 'EMG2'])

# Plot with time range (in seconds)
emg.plot_signals(time_range=(0, 5))

# Customize plot
emg.plot_signals(
    channels=['EMG1', 'EMG2'],
    time_range=(0, 5),
    title='EMG Signals',
    grid=True,
    figsize=(12, 6)
)
```

## Exporting Data

EMGIO can export data to EDF/BDF formats:

```python
# Export to EDF or BDF (format selected automatically)
emg.to_edf('output')  # Extension (.edf/.bdf) will be added automatically

# Force EDF format
emg.to_edf('output', force_format='edf')

# Force BDF format
emg.to_edf('output', force_format='bdf')

# Control the analysis method for format selection
emg.to_edf('output', method='svd')  # Use SVD analysis only
emg.to_edf('output', method='fft')  # Use FFT analysis only
emg.to_edf('output', method='both')  # Use both methods (default)
```

## Next Steps

After mastering these basics, you might want to explore:

- [Channel Selection](channel-selection.md) - Learn how to select and manipulate channels
- [EDF/BDF Format Selection](edf-bdf-selection.md) - Understanding how EMGIO selects the appropriate format
- [Metadata Handling](metadata.md) - Working with metadata in EMGIO 