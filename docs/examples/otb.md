# OTB Examples

This page provides examples for working with OTB+ files from OT Bioelettronica devices using EMGIO.

## Basic OTB Example

```python
import matplotlib.pyplot as plt
from emgio import EMG

# Load OTB data
data_path = 'path_to_your_otb_file.otb+'
emg = EMG.from_file(data_path, importer='otb')

# Print device information
print("\nDevice Information:")
print("-" * 50)
print(f"Device: {emg.get_metadata('device')}")
print(f"Resolution: {emg.get_metadata('signal_resolution')} bits")

# Print channel type summary
print("\nChannel Type Summary:")
print("-" * 50)
channel_types = emg.get_channel_types()
for ch_type in channel_types:
    channels = emg.get_channels_by_type(ch_type)
    print(f"{ch_type}: {len(channels)} channels")

# Plot one channel of each type
plt.figure(figsize=(12, 8))
subplot_count = len(channel_types)
for i, ch_type in enumerate(channel_types, 1):
    channels = emg.get_channels_by_type(ch_type)
    if channels:
        plt.subplot(subplot_count, 1, i)
        # Select the first channel of this type
        sample_channel = emg.select_channels([channels[0]])
        sample_channel.plot_signals(
            time_range=(0, 5),
            title=f"{ch_type} Sample: {channels[0]}",
            ax=plt.gca()
        )

plt.tight_layout()
plt.show()
```

## Working with EMG Channels

OTB files often contain multiple channel types. Here's how to work specifically with EMG channels:

```python
from emgio import EMG
import matplotlib.pyplot as plt
import numpy as np

# Load OTB data
emg = EMG.from_file('otb_data.otb+', importer='otb')

# Create a new EMG object with only EMG channels
emg_channels = emg.get_channels_by_type('EMG')

if emg_channels:
    print(f"Found {len(emg_channels)} EMG channels")
    
    # Select EMG channels
    emg_data = emg.select_channels(emg_channels)
    
    # Plot EMG data
    plt.figure(figsize=(15, 10))
    
    # If there are many channels, select a subset
    channels_to_plot = emg_channels[:8] if len(emg_channels) > 8 else emg_channels
    
    emg_data.plot_signals(
        channels=channels_to_plot,
        time_range=(0, 5),
        title='EMG Channels from OTB File',
        grid=True
    )
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Calculate and plot RMS of EMG signals
    plt.figure(figsize=(15, 6))
    
    # Calculate RMS in 100ms windows
    fs = emg_data.get_sampling_frequency()
    window_size = int(0.1 * fs)  # 100ms window
    
    for i, channel in enumerate(channels_to_plot):
        # Get signal data
        signal = emg_data.signals[channel].values
        
        # Calculate RMS in windows
        n_windows = len(signal) // window_size
        rms = np.zeros(n_windows)
        
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window_data = signal[start:end]
            rms[w] = np.sqrt(np.mean(np.square(window_data)))
        
        # Create time vector for RMS
        time = np.arange(n_windows) * (window_size / fs)
        
        # Plot RMS
        plt.plot(time, rms, label=channel)
    
    plt.title('RMS of EMG Signals (100ms windows)')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No EMG channels found in the file")
```

## Working with Multiple OTB Files

When working with multiple recordings from the same experiment:

```python
from emgio import EMG
import matplotlib.pyplot as plt

# Load two OTB files
file1 = 'session1.otb+'
file2 = 'session2.otb+'

emg1 = EMG.from_file(file1, importer='otb')
emg2 = EMG.from_file(file2, importer='otb')

# Add metadata to distinguish them
emg1.set_metadata('session', '1')
emg1.set_metadata('condition', 'rest')
emg2.set_metadata('session', '2')
emg2.set_metadata('condition', 'contraction')

# Check that they have the same channel structure
channels1 = set(emg1.channels.keys())
channels2 = set(emg2.channels.keys())
common_channels = channels1.intersection(channels2)

print(f"File 1 has {len(channels1)} channels")
print(f"File 2 has {len(channels2)} channels")
print(f"Common channels: {len(common_channels)}")

# Compare EMG channels between sessions
emg_channels = [ch for ch in common_channels 
                if emg1.channels[ch]['channel_type'] == 'EMG']

if emg_channels:
    # Select a channel to compare
    channel_to_compare = emg_channels[0]
    
    plt.figure(figsize=(12, 8))
    
    # Plot channel from first session
    plt.subplot(2, 1, 1)
    emg1.plot_signals(
        [channel_to_compare], 
        time_range=(0, 5),
        title=f"Session 1 ({emg1.get_metadata('condition')}): {channel_to_compare}",
        ax=plt.gca()
    )
    
    # Plot same channel from second session
    plt.subplot(2, 1, 2)
    emg2.plot_signals(
        [channel_to_compare], 
        time_range=(0, 5),
        title=f"Session 2 ({emg2.get_metadata('condition')}): {channel_to_compare}",
        ax=plt.gca()
    )
    
    plt.tight_layout()
    plt.show()
```

## Exporting OTB Data to EDF/BDF

Converting OTB data to EDF/BDF format:

```python
from emgio import EMG

# Load OTB data
emg = EMG.from_file('data.otb+', importer='otb')

# Add metadata for export
emg.set_metadata('subject', 'S001')
emg.set_metadata('recording_date', '2023-01-15')
emg.set_metadata('experimenter', 'Researcher')

# Export all channels
output_path = 'otb_all_channels'
emg.to_edf(output_path)  # Format (EDF/BDF) selected automatically
print(f"Exported all channels to: {output_path}")

# Export only EMG channels for clarity
emg_channels = emg.get_channels_by_type('EMG')
if emg_channels:
    emg_only = emg.select_channels(emg_channels)
    output_path = 'otb_emg_only'
    emg_only.to_edf(output_path)
    print(f"Exported EMG channels to: {output_path}")

# If there are too many channels for EDF, you might need to split
# EDF has limitations on the number of channels it can handle
if len(emg.channels) > 100:  # EDF typically handles fewer channels
    print("Large number of channels detected, splitting export")
    
    # Split into groups of 50 channels
    channel_lists = [list(emg.channels.keys())[i:i+50] 
                    for i in range(0, len(emg.channels), 50)]
    
    for i, channel_list in enumerate(channel_lists):
        subset_emg = emg.select_channels(channel_list)
        output_path = f'otb_split_{i+1}'
        subset_emg.to_edf(output_path)
        print(f"Exported channel group {i+1} to: {output_path}")
```

These examples demonstrate how to load OTB data, explore channel information, work with specific channel types, compare multiple recordings, and export to EDF/BDF format. 