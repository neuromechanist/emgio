# EEGLAB Examples

This page provides examples for working with EEGLAB `.set` files using EMGIO.

## Basic EEGLAB Example

```python
import os
from emgio import EMG
import matplotlib.pyplot as plt

# Load data from an EEGLAB .set file
data_path = 'path_to_your_eeglab_file.set'
emg = EMG.from_file(data_path, importer='eeglab')

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
from emgio import EMG
import numpy as np
import matplotlib.pyplot as plt

# Load EEGLAB data with events
emg = EMG.from_file('data_with_events.set', importer='eeglab')

# Check if events exist in the metadata
if 'event' in emg.metadata:
    events = emg.get_metadata('event')
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
        
        # Plot 2 seconds before and after the event
        window = 2  # seconds
        plt.figure(figsize=(12, 8))
        emg.plot_signals(
            time_range=(event_time - window, event_time + window),
            title=f"EMG around movement event at {event_time:.2f}s"
        )
        
        # Add a vertical line at the event time
        plt.axvline(x=event_time, color='r', linestyle='--', label='Movement Event')
        plt.legend()
        plt.tight_layout()
        plt.show()
```

## Converting Epoched Data

EEGLAB can store continuous or epoched data. EMGIO can work with both:

```python
from emgio import EMG
import matplotlib.pyplot as plt

# Load epoched EEGLAB data
emg = EMG.from_file('epoched_data.set', importer='eeglab')

# Check if data is epoched
is_epoched = emg.get_metadata('trials', 1) > 1
print(f"Data is {'epoched' if is_epoched else 'continuous'}")

if is_epoched:
    # Get epoch information
    n_epochs = emg.get_metadata('trials')
    epoch_length = emg.get_metadata('pnts')
    fs = emg.get_sampling_frequency()
    epoch_duration = epoch_length / fs
    
    print(f"Number of epochs: {n_epochs}")
    print(f"Epoch length: {epoch_length} samples ({epoch_duration:.2f} seconds)")
    
    # EMGIO automatically concatenates epochs, so the data is handled as continuous
    # The total duration is epochs * epoch_duration
    total_duration = emg.get_duration()
    print(f"Total duration: {total_duration:.2f} seconds")
    
    # Plot signals from different epochs
    plt.figure(figsize=(15, 10))
    
    # Plot first epoch
    plt.subplot(3, 1, 1)
    emg.plot_signals(
        time_range=(0, epoch_duration),
        title="First Epoch",
        ax=plt.gca()
    )
    
    # Plot middle epoch
    middle_epoch = n_epochs // 2
    middle_start = middle_epoch * epoch_duration
    plt.subplot(3, 1, 2)
    emg.plot_signals(
        time_range=(middle_start, middle_start + epoch_duration),
        title=f"Middle Epoch (Epoch {middle_epoch+1})",
        ax=plt.gca()
    )
    
    # Plot last epoch
    last_start = (n_epochs - 1) * epoch_duration
    plt.subplot(3, 1, 3)
    emg.plot_signals(
        time_range=(last_start, last_start + epoch_duration),
        title=f"Last Epoch (Epoch {n_epochs})",
        ax=plt.gca()
    )
    
    plt.tight_layout()
    plt.show()
```

## Exporting EEGLAB Data to EDF/BDF

Converting EEGLAB data to EDF/BDF format:

```python
from emgio import EMG

# Load EEGLAB data
emg = EMG.from_file('data.set', importer='eeglab')

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