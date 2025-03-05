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
        print(f"- {ch_name} ({ch_info['channel_type']})")
        print(f"  Sampling rate: {ch_info['sample_frequency']} Hz")
        print(f"  Dimension: {ch_info['physical_dimension']}")

    # Select EMG channels only and create a new EMG object
    emg_channels = [ch for ch, info in emg.channels.items() if info['channel_type'] == 'EMG']
    emg_only = emg.select_channels(emg_channels)  # Creates a new EMG object with only EMG channels
    # Original emg object remains unchanged with all channels

    # Plot the first 5 seconds of data with different configurations
    print("\nPlotting EMG signals from EMG-only channels...")

    # Default plot with uniform scaling
    emg_only.plot_signals(
        time_range=(0, 5),
        title="EMG Signals - Uniform Scale"
    )

    # Export to EDF/BDF (format will be automatically selected)
    output_path = 'examples/trigno_emg'  # Extension will be added by the exporter (.edf or .bdf)
    print("\nExporting EMG data...")
    print("Note: The exporter will automatically:")
    print("- Convert voltage units to mV if needed")
    print("- Choose between EDF/BDF based on precision requirements")
    print("- Handle value truncation with appropriate warnings")
    print("- Use the specified signal analysis method")

    # Example of using the exposed parameters for signal analysis
    # method: 'svd', 'fft', or 'both' (default)
    # svd_rank: Optional manual rank cutoff for SVD method
    # fft_noise_range: Optional tuple (min_freq, max_freq) for FFT method
    # Since the non-emg channels have different sampling frequency, the output will become bdf to handle 
    # NaNs for the missing values in the non-emg channels. This is the default behavior for mixed sampling frequencies.
    emg_only.to_edf(output_path, method='both', svd_rank=None, fft_noise_range=None)

    # Alternative examples:
    # Using only FFT method:
    # emg.to_edf(output_path, method='fft')
    
    # Using only SVD method with custom rank:
    # emg.to_edf(output_path, method='svd', svd_rank=5)
    
    # Using both methods with custom FFT noise range:
    # emg.to_edf(output_path, method='both', fft_noise_range=(0.1, 10))

    print("\nExport complete!")


if __name__ == "__main__":
    main()
