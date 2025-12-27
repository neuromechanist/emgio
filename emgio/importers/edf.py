from typing import Dict, Tuple

import numpy as np
import pyedflib

from ..core.emg import EMG
from .base import BaseImporter


class EDFImporter(BaseImporter):
    """Importer for EDF/EDF+/BDF format files."""

    def _determine_channel_type(self, label: str, transducer: str) -> str:
        """
        Determine channel type based on label and transducer info.

        Args:
            label: Channel label from EDF header
            transducer: Transducer type from EDF header

        Returns:
            str: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        label = label.upper()
        transducer = transducer.upper()

        if "EMG" in label or "EMG" in transducer:
            return "EMG"
        elif "ACC" in label or "ACCELEROMETER" in transducer:
            return "ACC"
        elif "GYRO" in label or "GYROSCOPE" in transducer:
            return "GYRO"
        elif "TRIG" in label or "TRIGGER" in transducer:
            return "TRIG"
        else:
            return "OTHER"

    def _extract_metadata(self, edf_reader: pyedflib.EdfReader) -> Dict:
        """
        Extract metadata from EDF file header.

        Args:
            edf_reader: pyedflib EdfReader instance

        Returns:
            dict: Dictionary containing file metadata
        """
        header = edf_reader.getHeader()
        signals_headers = edf_reader.getSignalHeaders()

        metadata = {
            "recording_info": {
                "startdate": header.get("startdate", None),
                "patientcode": header.get("patientcode", ""),
                "gender": header.get("gender", ""),
                "birthdate": header.get("birthdate", ""),
                "patient_name": header.get("patient_name", ""),
                "patient_additional": header.get("patient_additional", ""),
                "admincode": header.get("admincode", ""),
                "technician": header.get("technician", ""),
                "equipment": header.get("equipment", ""),
                "recording_additional": header.get("recording_additional", ""),
            },
            "file_info": {
                "filetype": header.get("filetype", 0),  # 0: EDF, 1: EDF+, 2: BDF+
                "number_of_signals": len(signals_headers),
                "file_duration": header.get("file_duration", 0),
                "datarecord_duration": header.get("datarecord_duration", 0),
            },
        }

        return metadata

    def _read_signal_data(
        self, edf_reader: pyedflib.EdfReader, signal_idx: int
    ) -> Tuple[np.ndarray, Dict]:
        """
        Read signal data and header information for a specific channel.

        Args:
            edf_reader: pyedflib EdfReader instance
            signal_idx: Index of the signal to read

        Returns:
            tuple: (signal_data, signal_info)
        """
        # Get signal header
        signal_header = edf_reader.getSignalHeaders()[signal_idx]

        # Read the signal data
        signal_data = edf_reader.readSignal(signal_idx)

        # Extract signal information
        signal_info = {
            "label": signal_header["label"].strip(),
            "transducer": signal_header.get("transducer", "").strip(),
            "physical_dimension": signal_header.get("dimension", "").strip() or "n/a",
            "physical_min": signal_header["physical_min"],
            "physical_max": signal_header["physical_max"],
            "digital_min": signal_header["digital_min"],
            "digital_max": signal_header["digital_max"],
            "prefilter": signal_header.get("prefilter", "").strip() or "n/a",
            "sample_frequency": signal_header["sample_frequency"],
        }

        return signal_data, signal_info

    def load(self, filepath: str) -> EMG:
        """
        Load EMG data from EDF/EDF+/BDF file.

        Args:
            filepath: Path to the EDF file

        Returns:
            EMG: EMG object containing the loaded data
        """
        try:
            edf_reader = pyedflib.EdfReader(filepath)

            # Create EMG object
            emg = EMG()

            # Extract and store metadata
            metadata = self._extract_metadata(edf_reader)
            for key, value in metadata["recording_info"].items():
                if value:  # Only store non-empty values
                    emg.set_metadata(key, value)
            for key, value in metadata["file_info"].items():
                emg.set_metadata(key, value)

            # Store source file information
            emg.set_metadata("source_file", filepath)

            # Read signals
            for i in range(edf_reader.signals_in_file):
                signal_data, signal_info = self._read_signal_data(edf_reader, i)

                # Determine channel type
                channel_type = self._determine_channel_type(
                    signal_info["label"], signal_info["transducer"]
                )

                # Add channel to EMG object
                emg.add_channel(
                    label=signal_info["label"],
                    data=signal_data,
                    sample_frequency=signal_info["sample_frequency"],
                    physical_dimension=signal_info["physical_dimension"],
                    prefilter=signal_info["prefilter"],
                    channel_type=channel_type,
                )

                # Store additional channel-specific metadata
                channel_metadata = {
                    "physical_min": signal_info["physical_min"],
                    "physical_max": signal_info["physical_max"],
                    "digital_min": signal_info["digital_min"],
                    "digital_max": signal_info["digital_max"],
                    "transducer": signal_info["transducer"],
                }
                emg.channels[signal_info["label"]].update(channel_metadata)

            return emg

        except Exception as e:
            raise ValueError(f"Error reading EDF file: {str(e)}") from e

        finally:
            if "edf_reader" in locals():
                edf_reader.close()
