# EEGLAB Examples

This page provides examples for working with EEGLAB `.set` files using biosigIO. The importer reads `.set` files with `scipy.io.loadmat`, so only pre-v7.3 (non-HDF5) MATLAB `.set` files are supported; v7.3/HDF5 `.set` files are not.

## Basic EEGLAB Example

```python
import os
from biosigio import Recording
import matplotlib.pyplot as plt

# Load data from an EEGLAB .set file
data_path = 'path_to_your_eeglab_file.set'
emg = Recording.from_file(data_path, importer='eeglab')

# Print metadata
print("\nMetadata:")
print("-" * 50)
for key in ['subject', 'session', 'condition', 'srate', 'nbchan', 'pnts']:
    if key in emg.metadata:
        print(f"{key}: {emg.get_metadata(key)}")

# Print available channels
print("\nAvailable channels:")
print("-" * 50)
channel_types = emg.get_channel_types()
for ch_type in channel_types:
    channels = emg.get_channels_by_type(ch_type)
    print(f"{ch_type} channels ({len(channels)}):")
    for i, ch_name in enumerate(channels[:5]):  # Print first 5 channels of each type
        ch_info = emg.channels[ch_name]
        print(f"  - {ch_name} (Sampling rate: {ch_info['sample_frequency']} Hz, "
              f"Unit: {ch_info['physical_dimension']})")
    if len(channels) > 5:
        print(f"  ... and {len(channels) - 5} more {ch_type} channels")

# Plot EMG channels
emg_channels = emg.get_channels_by_type('EMG')
if emg_channels:
    # Select EMG channels only
    emg_only = emg.select_channels(emg_channels)
    
    # Plot the first 5 seconds
    plt.figure(figsize=(12, 8))
    emg_only.plot_signals(time_range=(0, 5), title="EMG Signals from EEGLAB")
    plt.tight_layout()
    plt.show()
```

## Working with Events

EEGLAB files often contain event markers. Here's how to access and work with them:

```python
from biosigio import Recording
import numpy as np
import matplotlib.pyplot as plt

# Load EEGLAB data with events
emg = Recording.from_file('data_with_events.set', importer='eeglab')

# EEGLAB events are stored under the metadata key 'events' (plural) as a list
# of dicts with 'type', 'latency', and (optionally) 'duration'/'trial_type'.
if emg.has_metadata('events'):
    events = emg.get_metadata('events')
    print(f"Found {len(events)} events")
    
    # Print the first 5 events
    for i, event in enumerate(events[:5]):
        print(f"Event {i+1}: Type={event.get('type')}, Latency={event.get('latency')}")
    
    # Extract specific event types
    movement_events = [e for e in events if e.get('type') == 'movement']
    print(f"Found {len(movement_events)} movement events")
    
    # Plot signals around an event
    if movement_events:
        # Get the timestamp of the first movement event (convert from samples to seconds)
        event_sample = movement_events[0].get('latency')
        fs = emg.get_sampling_frequency()
        event_time = event_sample / fs
        
        # Plot 2 seconds before and after the event. Pass show=False so the
        # event marker can be overlaid before displaying.
        window = 2  # seconds
        emg.plot_signals(
            time_range=(event_time - window, event_time + window),
            title=f"EMG around movement event at {event_time:.2f}s",
            show=False
        )

        # Add a vertical line at the event time
        plt.axvline(x=event_time, color='r', linestyle='--', label='Movement Event')
        plt.legend()
        plt.show()
```

## Inspecting Epoch Metadata

EEGLAB stores the number of epochs in the `trials` field and the samples per
epoch in `pnts`. These are read into the recording metadata, so you can inspect
them after loading. `get_metadata` returns `None` when a key is absent, so guard
for that before comparing:

```python
from biosigio import Recording
import matplotlib.pyplot as plt

# Load EEGLAB data
emg = Recording.from_file('epoched_data.set', importer='eeglab')

# trials > 1 indicates epoched data; get_metadata returns None if absent
trials = emg.get_metadata('trials')
is_epoched = trials is not None and trials > 1
print(f"Data is {'epoched' if is_epoched else 'continuous'}")

if is_epoched:
    n_epochs = emg.get_metadata('trials')
    epoch_length = emg.get_metadata('pnts')
    fs = emg.get_sampling_frequency()
    epoch_duration = epoch_length / fs

    print(f"Number of epochs: {n_epochs}")
    print(f"Epoch length: {epoch_length} samples ({epoch_duration:.2f} seconds)")

# Plot the first few seconds (plot_signals manages its own figure)
emg.plot_signals(time_range=(0, 5), title="EEGLAB Signals")
```

## Exporting EEGLAB Data to EDF/BDF

Converting EEGLAB data to EDF/BDF format:

```python
from biosigio import Recording

# Load EEGLAB data
emg = Recording.from_file('data.set', importer='eeglab')

# Export all channels to EDF/BDF
output_path = 'eeglab_all_channels'
emg.to_edf(output_path)  # Format (EDF/BDF) selected automatically

# Export only EMG channels
emg_only = emg.select_channels(channel_type='EMG')
output_path = 'eeglab_emg_only'
emg_only.to_edf(output_path)

print("Conversion complete!")
```

This example demonstrates loading EEGLAB `.set` files, exploring their structure, working with events, handling epoched data, and exporting to EDF/BDF format. 