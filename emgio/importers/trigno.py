from typing import List, Tuple

import pandas as pd

from ..core.emg import EMG
from .base import BaseImporter


class TrignoImporter(BaseImporter):
    """Importer for Delsys Trigno EMG system data."""

    def _analyze_csv_structure(self, csv_path: str) -> Tuple[List[str], int, str]:
        """
        Analyze the CSV file structure to identify metadata and data sections.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Tuple containing:
                - List of metadata lines
                - Line number where data starts
                - Header line
        """
        metadata_lines = []
        data_start_line = 0
        header_line = None

        with open(csv_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                if "X[s]" in line:  # This is the header line
                    header_line = line
                    data_start_line = i + 1
                    break

                metadata_lines.append(line)

        return metadata_lines, data_start_line, header_line

    def _parse_metadata(self, metadata_lines: List[str]) -> dict:
        """
        Parse metadata lines to extract channel information.

        Args:
            metadata_lines: List of metadata lines from the file

        Returns:
            Dictionary containing channel information
        """
        channel_info = {}

        for line in metadata_lines:
            if line.startswith("Label:"):
                # Extract channel name
                name_part = line[line.find("Label:") + 6 : line.find("Sampling")].strip()

                # Extract sampling frequency
                freq_str = line[line.find("frequency:") + 10 :].split()[0]
                sampling_freq = float(freq_str)

                # Extract unit
                unit = line[line.find("Unit:") + 5 : line.find("Domain")].strip()

                channel_info[name_part] = {
                    "sample_frequency": sampling_freq,
                    "physical_dimension": unit,
                }

        return channel_info

    def load(self, filepath: str) -> EMG:
        """
        Load EMG data from Trigno CSV file.

        Args:
            filepath: Path to the Trigno CSV file

        Returns:
            EMG: EMG object containing the loaded data
        """
        # Create EMG object
        emg = EMG()

        # Analyze file structure
        metadata_lines, data_start, header_line = self._analyze_csv_structure(filepath)

        # Parse metadata
        channel_info = self._parse_metadata(metadata_lines)

        # Read data section
        df = pd.read_csv(filepath, skiprows=data_start - 1)

        # Clean up column names
        df.columns = [col.replace("X[s]", "").strip('"') for col in df.columns]

        # Get valid channel names (excluding time columns and extra columns)
        channel_labels = [col for col in df.columns if col and not col.startswith(".")]

        # Create time index
        time_col = df.columns[0]  # First column is time
        df.set_index(time_col, inplace=True)

        # Add channels to EMG object
        for label in channel_labels:
            if label in channel_info:
                info = channel_info[label]

                # Determine channel type
                if "EMG" in label:
                    ch_type = "EMG"
                elif "ACC" in label:
                    ch_type = "ACC"
                elif "GYRO" in label:
                    ch_type = "GYRO"
                else:
                    ch_type = "OTHER"

                emg.add_channel(
                    label=label,
                    data=df[label].values,
                    sample_frequency=info["sample_frequency"],
                    physical_dimension=info["physical_dimension"],
                    channel_type=ch_type,
                )

        # Add file metadata
        emg.set_metadata("source_file", filepath)
        emg.set_metadata("device", "Delsys Trigno")

        return emg
