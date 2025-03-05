"""
Example script demonstrating how to use the EMGIO package with EEGLAB .set files.
This example shows how to:
1. Load data from an EEGLAB .set file
2. Access metadata and channel information
3. Select specific channels
4. Plot the signals
5. Export to EDF/BDF format with automatic format selection and precision handling
"""

import os
from emgio import EMG


def main():
    # Sample data path - replace with your actual data path
    data_path = (
        'examples/wristbandEMG_truncated.set'  # Update this path to your EEGLAB .set file
    )

    if not os.path.exists(data_path):
        print(f"Sample file not found: {data_path}")
        print("Please update the data_path variable with your EEGLAB .set file path.")
        return

    # Load the data using EEGLAB importer
    print("Loading EMG data from EEGLAB .set file...")
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

    # Print event information if available
    if 'events' in emg.metadata and emg.metadata['events']:
        print("\nEvents:")
        print("-" * 50)
        events = emg.metadata['events']
        print(f"Number of events: {len(events)}")
        for i, event in enumerate(events[:5]):  # Print first 5 events
            print(f"  Event {i + 1}: Type={event['type']}, "
                  f"Latency={event['latency']:.1f}")
        if len(events) > 5:
            print(f"  ... and {len(events) - 5} more events")

    # Select EMG channels only and create a new EMG object
    emg_channels = emg.get_channels_by_type('EMG')
    if emg_channels:
        emg_only = emg.select_channels(emg_channels)  # Creates a new EMG object with only EMG channels
        
        # Plot the first 5 seconds of data with different configurations
        print("\nPlotting EMG signals...")
        
        # Get the maximum time value to determine plot range
        max_time = emg_only.signals.index[-1]
        plot_end = min(5.0, max_time)  # Plot up to 5 seconds or the end of the data
        
        # Default plot with uniform scaling
        emg_only.plot_signals(
            channels=emg_only.signals.columns[:8],  # Plot first 8 channels
            time_range=(0, plot_end),
            title="EMG Signals - Uniform Scale"
        )
        
        # Export to EDF/BDF (format will be automatically selected)
        output_path = 'examples/eeglab_emg'  # Extension will be added by the exporter (.edf or .bdf)
        print("\nExporting EMG data...")
        print("Note: The exporter will automatically:")
        print("- Choose between EDF/BDF based on precision requirements")
        print("- Handle value truncation with appropriate warnings")
        
        emg.to_edf(output_path)
        
        print("\nExport complete! Files created:")
        print(f"- {output_path}.edf or {output_path}.bdf (depending on precision requirements)")
        print(f"- {output_path}_channels.tsv (channel metadata)")
    else:
        print("\nNo EMG channels found in the data")


if __name__ == "__main__":
    main()
