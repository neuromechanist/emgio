import os

import matplotlib.pyplot as plt

from biosigio.core.emg import Recording


def main():
    # Define the path to the example WFDB file
    # This example uses record 100 from PhysioNet's MIT-BIH Arrhythmia Database
    # https://physionet.org/content/mitdb/1.0.0/
    file_path = "examples/100.hea"

    if not os.path.exists(file_path):
        print(f"Error: Example file not found at {file_path}")
        print(
            "Please ensure the MIT-BIH Arrhythmia database record 100 files "
            "(100.hea, 100.dat, 100.atr) are in the examples/ directory."
        )
        return

    # --- Loading WFDB Data with Annotations ---
    print(f"Loading WFDB data from: {file_path}")
    # The importer automatically looks for a corresponding .atr file (e.g., 100.atr)
    # and loads annotations into the emg.events DataFrame if found.
    try:
        emg = Recording.from_file(file_path, importer="wfdb")
    except ValueError as e:
        print(f"Error loading file: {e}")
        print("Ensure the 'wfdb' package is installed ('pip install wfdb')")
        return
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both the .hea and .dat files exist.")
        return

    print("Data loaded successfully.")

    # --- Displaying Metadata and Channel Info ---
    print("\nMetadata:")
    print("-" * 50)
    print(f"Record Name: {emg.get_metadata('record_name')}")
    print(f"Sampling Frequency: {emg.get_metadata('sampling_frequency')} Hz")
    print(f"Source File: {emg.get_metadata('source_file')}")
    start_date = emg.get_metadata("startdate")
    start_time = emg.get_metadata("starttime")
    if start_date or start_time:
        print(f"Recording Start: {start_date} {start_time}")
    if emg.get_metadata("comments"):
        print(f"Comments:\n{emg.get_metadata('comments')}")

    print("\nChannels:")
    print("-" * 50)
    for name, info in emg.channels.items():
        print(
            f"- {name}: Type={info.get('channel_type', 'N/A')}, Unit={info.get('physical_dimension', 'N/A')}"
        )

    # --- Accessing Loaded Annotations ---
    print("\nAnnotations (Events):")
    print("-" * 50)
    if emg.events is not None and not emg.events.empty:
        print(f"Found {len(emg.events)} annotations.")
        print("First 5 annotations:")
        print(emg.events.head())
        print("\nLast 5 annotations:")
        print(emg.events.tail())
        annotations_present = True
    else:
        print("No annotations found or loaded.")
        # Check metadata for reasons if annotations were expected
        if emg.get_metadata("annotation_status"):
            print(f"Status: {emg.get_metadata('annotation_status')}")
        if emg.get_metadata("annotation_error"):
            print(f"Error: {emg.get_metadata('annotation_error')}")
        annotations_present = False

    # --- Plotting Signals (optional) ---
    # Plotting the first 10 seconds of the first channel (MLII)
    print("\nPlotting first 10 seconds of the first channel...")
    try:
        first_channel = list(emg.channels.keys())[0]
        emg.plot_signals(
            channels=[first_channel], time_range=(0, 10), title=f"{first_channel} (First 10s)"
        )
        plt.show()
    except Exception as e:
        print(f"Could not plot signals: {e}")

    # --- Exporting to EDF/BDF (demonstrating annotation preservation) ---
    print("\nExporting data to EDF and BDF formats...")
    output_base = "examples/100_exported"

    try:
        # Export to EDF
        print("Exporting to EDF...")
        edf_path = f"{output_base}.edf"
        emg.to_edf(edf_path, format="edf")
        print(f"Exported to {edf_path}")
        if annotations_present:
            print("(Annotations included in EDF+)")

        # Export to BDF
        print("\nExporting to BDF...")
        bdf_path = f"{output_base}.bdf"
        emg.to_edf(bdf_path, format="bdf")
        print(f"Exported to {bdf_path}")
        if annotations_present:
            print("(Annotations included in BDF+)")

    except Exception as e:
        print(f"Error during export: {e}")

    print("\nExample script finished.")


if __name__ == "__main__":
    main()
