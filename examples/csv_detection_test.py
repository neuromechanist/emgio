"""
Example script demonstrating the CSV format detection feature.

This example shows:
1. How EMGIO detects specialized CSV formats
2. How the detection mechanism improves the user experience
3. The benefits of specialized importers for structured formats
"""

import os

from emgio import Recording


def test_format_detection():
    """Test CSV format detection on a sample file."""
    # Sample Trigno CSV file
    trigno_csv = "examples/truncated_trigno_sample.csv"
    # Sample generic CSV file (our EMG sample file)
    generic_csv = "examples/truncated_emg_sample_zenodo7668251.txt"

    print("\n=== Testing Automatic Format Detection ===\n")

    # Test Trigno file detection
    if os.path.exists(trigno_csv):
        print("Testing with Trigno CSV file...")
        try:
            # This should detect Trigno format and suggest using the Trigno importer
            print("Attempting to load Trigno CSV with generic CSV importer...")
            Recording.from_file(trigno_csv, importer="csv")
            print("File loaded successfully (you shouldn't see this)")
        except ValueError as e:
            print(f"Format detection message: {str(e)}")

        # Load with the specialized importer
        print("\nLoading with specialized Trigno importer...")
        try:
            trigno_emg = Recording.from_file(trigno_csv, importer="trigno")

            # Print channel information from Trigno importer
            print("\nChannel Information (Trigno Importer):")
            print("-" * 50)
            ch_types = {}
            for ch_info in trigno_emg.channels.values():
                ch_type = ch_info["channel_type"]
                if ch_type not in ch_types:
                    ch_types[ch_type] = 1
                else:
                    ch_types[ch_type] += 1

            for ch_type, count in ch_types.items():
                print(f"- {ch_type}: {count} channels")

            # Print sample of channels
            print("\nSample channels:")
            for ch_name in list(trigno_emg.channels.keys())[:3]:
                ch_info = trigno_emg.channels[ch_name]
                print(f"- {ch_name} ({ch_info['channel_type']})")
                print(f"  Sampling rate: {ch_info['sample_frequency']} Hz")
                print(f"  Dimension: {ch_info['physical_dimension']}")

            print("\nMetadata from Trigno importer:")
            for key, value in trigno_emg.metadata.items():
                if key != "source_file":
                    print(f"- {key}: {value}")
        except Exception as e:
            print(f"Error loading with Trigno importer: {str(e)}")
    else:
        print(f"Sample Trigno file not found: {trigno_csv}")

    # Test generic CSV file
    print("\n=== Testing with Generic CSV File ===\n")

    if os.path.exists(generic_csv):
        print("Testing with generic CSV/text file...")
        try:
            # This should work fine with the CSV importer
            print("Loading generic CSV with CSV importer...")

            # We need to provide additional parameters for the generic CSV
            csv_params = {
                "has_header": False,
                "sample_frequency": 1000.0,
                "channel_types": {
                    "Channel_0": "EMG",
                    "Channel_1": "EMG",
                    "Channel_2": "EMG",
                    "Channel_3": "EMG",
                },
                "physical_dimensions": {
                    "Channel_0": "mV",
                    "Channel_1": "mV",
                    "Channel_2": "mV",
                    "Channel_3": "mV",
                },
            }

            generic_emg = Recording.from_file(generic_csv, importer="csv", **csv_params)

            # Print channel information
            print("\nChannel Information (Generic CSV):")
            print("-" * 50)
            for ch_name, ch_info in generic_emg.channels.items():
                print(f"- {ch_name} ({ch_info['channel_type']})")
                print(f"  Sampling rate: {ch_info['sample_frequency']} Hz")
                print(f"  Dimension: {ch_info['physical_dimension']}")

            print("\nWith the generic CSV importer, we need to manually specify:")
            print("- Channel types")
            print("- Physical dimensions")
            print("- Sampling frequency")
            print("- Header/delimiter configuration")
        except Exception as e:
            print(f"Error loading generic CSV: {str(e)}")
    else:
        print(f"Sample generic CSV file not found: {generic_csv}")

    print("\n=== Conclusion ===\n")
    print("The format detection feature helps users by:")
    print("1. Identifying specialized formats like Trigno CSV exports")
    print("2. Suggesting the appropriate importer for better metadata and channel detection")
    print("3. Allowing users to override the detection if needed")
    print("\nThis ensures a better user experience while maintaining flexibility.")


if __name__ == "__main__":
    test_format_detection()
