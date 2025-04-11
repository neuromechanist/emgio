"""
Example script demonstrating how to use the EMGIO package with CSV EMG data.
This example shows how to:
1. Load data from a generic CSV/Text file
2. Define channel information for unlabeled data
3. Plot the signals
4. Export to EDF/BDF format

This example uses a sample EMG file from Zenodo dataset 7668251.
The file contains unlabeled columns of EMG data with no header.
"""

import os
from emgio import EMG


def main():
    # Sample data path - replace with your actual data path
    data_path = 'examples/truncated_emg_sample_zenodo7668251.txt'

    if not os.path.exists(data_path):
        print(f"Sample file not found: {data_path}")
        print("Please update the data_path variable with your CSV file path.")
        return

    # Set up channel naming and metadata for the unlabeled data file
    channel_names = ['EMG_CH1', 'EMG_CH2', 'EMG_CH3', 'EMG_CH4']

    # Define channel types (all are EMG channels in this case)
    channel_types = {name: 'EMG' for name in channel_names}

    # Define physical dimensions (units) for each channel
    physical_dimensions = {name: 'mV' for name in channel_names}

    # Additional metadata for the recording
    metadata = {
        'subject': 'Sample Subject',
        'device': 'Sample EMG Device',
        'recording_date': '2023-01-01',
        'experiment': 'Sample EMG Recording',
        'source': 'Zenodo Dataset 7668251'
    }

    # Load the data using CSV importer
    print("Loading EMG data...")

    try:
        # Import the CSVImporter directly
        from emgio.importers.csv import CSVImporter

        # Prepare parameters for the CSV importer
        csv_params = {
            'columns': [0, 1, 2, 3],  # Select columns by index (0-based)
            'has_header': False,       # No header in this file
            'delimiter': None,         # Auto-detect delimiter
            'sample_frequency': 1000.0,  # 1kHz sampling rate
            'metadata': metadata
        }

        # Create importer and load the data
        importer = CSVImporter()
        emg = importer.load(data_path, **csv_params)

        # Rename channels and set their types and units
        print("Renaming channels and setting metadata...")

        # Create a new EMG object with renamed channels
        new_emg = EMG()

        # Copy metadata
        for key, value in emg.metadata.items():
            new_emg.set_metadata(key, value)

        # Copy data with new channel names and metadata
        for i, (old_name, new_name) in enumerate(zip(emg.channels.keys(), channel_names)):
            ch_info = emg.channels[old_name]
            new_emg.add_channel(
                label=new_name,
                data=emg.signals[old_name].values,
                sample_frequency=ch_info['sample_frequency'],
                physical_dimension=physical_dimensions[new_name],
                channel_type=channel_types[new_name]
            )

        # Replace our reference to emg with the new object
        emg = new_emg
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Print available channels
    print("\nAvailable channels:")
    for ch_name, ch_info in emg.channels.items():
        print(f"- {ch_name} ({ch_info['channel_type']})")
        print(f"  Sampling rate: {ch_info['sample_frequency']} Hz")
        print(f"  Dimension: {ch_info['physical_dimension']}")

    # Print metadata
    print("\nRecording Information:")
    print("-" * 50)
    for key, value in emg.metadata.items():
        if key != 'source_file':  # Skip the file path
            print(f"{key}: {value}")

    # Plot the first 3 seconds of data
    print("\nPlotting EMG signals...")
    try:
        # Default plot
        emg.plot_signals(
            time_range=(0, 3),
            title="EMG Signals - First 3 Seconds",
            grid=True
        )

        # Plot with detrending (removes mean)
        emg.plot_signals(
            time_range=(0, 3),
            title="EMG Signals - Detrended",
            detrend=True,
            grid=True
        )
    except Exception as e:
        print(f"Error plotting signals: {str(e)}")

    # Export to EDF/BDF
    output_path = 'examples/csv_emg'  # Extension will be added by the exporter (.edf or .bdf)
    print("\nExporting EMG data...")

    try:
        # Export to EDF/BDF (format will be automatically selected based on data)
        emg.to_edf(output_path)
        print("\nExport complete!")
    except Exception as e:
        print(f"Error exporting to EDF: {str(e)}")


if __name__ == "__main__":
    main()
