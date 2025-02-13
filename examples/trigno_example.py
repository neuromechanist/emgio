"""
Example script demonstrating how to use the EMGIO package with Trigno EMG data.
This example shows how to:
1. Load data from a Trigno CSV file
2. Select specific channels
3. Plot the signals
4. Export to EDF/BDF format with automatic format selection and precision handling
"""

import os
from emgio import EMG


def main():
    # Sample data path - replace with your actual data path
    data_path = (
        'examples/truncated_trigno_sample.csv'  # Update this path to your Trigno CSV file
    )

    if not os.path.exists(data_path):
        print(f"Sample file not found: {data_path}")
        print("Please update the data_path variable with your Trigno CSV file path.")
        return

    # Load the data using Trigno importer
    print("Loading EMG data...")
    emg = EMG.from_file(data_path, importer='trigno')

    # Print available channels
    print("\nAvailable channels:")
    for ch_name, ch_info in emg.channels.items():
        print(f"- {ch_name} ({ch_info['type']})")
        print(f"  Sampling rate: {ch_info['sampling_freq']} Hz")
        print(f"  Unit: {ch_info['unit']}")

    # Select EMG channels only
    emg_channels = [ch for ch, info in emg.channels.items() if info['type'] == 'EMG']
    # emg.select_channels(emg_channels)  # TODO: #3 This removes all other channels in place, behavior should change.

    # Plot the first 5 seconds of data with different configurations
    print("\nPlotting EMG signals...")

    # Default plot with uniform scaling
    emg.plot_signals(
        time_range=(0, 5),
        channels=emg_channels,
        title="EMG Signals - Uniform Scale"
    )

    # Plot with detrending
    emg.plot_signals(
        time_range=(0, 5),
        channels=emg_channels,
        detrend=True,
        title="EMG Signals - Detrended"
    )

    # Plot with individual scaling
    emg.plot_signals(
        time_range=(0, 5),
        channels=emg_channels,
        uniform_scale=False,
        title="EMG Signals - Individual Scaling"
    )

    # Export to EDF/BDF (format will be automatically selected)
    output_path = 'examples/trigno_emg'  # Extension will be added by the exporter (.edf or .bdf)
    print("\nExporting EMG data...")
    print("Note: The exporter will automatically:")
    print("- Convert voltage units to mV if needed")
    print("- Choose between EDF/BDF based on precision requirements")
    print("- Handle value truncation with appropriate warnings")

    emg.to_edf(output_path)

    print("\nExport complete!")


if __name__ == "__main__":
    main()
