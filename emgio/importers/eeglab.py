from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..core.emg import EMG
from .base import BaseImporter


class EEGLABImporter(BaseImporter):
    """Importer for EEGLAB .set files containing EMG data."""

    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from EEGLAB .set file.

        Args:
            data: Dictionary containing EEGLAB .set file data

        Returns:
            dict: Dictionary containing extracted metadata
        """
        metadata = {}

        # Extract basic recording information
        if "setname" in data:
            metadata["setname"] = str(data["setname"][0]) if data["setname"].size > 0 else ""

        if "filename" in data:
            metadata["filename"] = str(data["filename"][0]) if data["filename"].size > 0 else ""

        if "filepath" in data:
            metadata["filepath"] = str(data["filepath"][0]) if data["filepath"].size > 0 else ""

        # Extract subject information
        if "subject" in data:
            metadata["subject"] = str(data["subject"][0]) if data["subject"].size > 0 else ""

        if "group" in data:
            metadata["group"] = str(data["group"][0]) if data["group"].size > 0 else ""

        if "condition" in data:
            metadata["condition"] = str(data["condition"][0]) if data["condition"].size > 0 else ""

        if "session" in data:
            metadata["session"] = str(data["session"][0]) if data["session"].size > 0 else ""

        if "comments" in data:
            metadata["comments"] = str(data["comments"][0]) if data["comments"].size > 0 else ""

        # Extract recording parameters
        if "srate" in data:
            metadata["srate"] = float(data["srate"][0][0]) if data["srate"].size > 0 else 0

        if "nbchan" in data:
            metadata["nbchan"] = int(data["nbchan"][0][0]) if data["nbchan"].size > 0 else 0

        if "trials" in data:
            metadata["trials"] = int(data["trials"][0][0]) if data["trials"].size > 0 else 0

        if "pnts" in data:
            metadata["pnts"] = int(data["pnts"][0][0]) if data["pnts"].size > 0 else 0

        if "xmin" in data and data["xmin"].size > 0:
            metadata["xmin"] = float(data["xmin"][0][0])

        if "xmax" in data and data["xmax"].size > 0:
            metadata["xmax"] = float(data["xmax"][0][0])

        # Add device information
        metadata["device"] = "EEGLAB"

        return metadata

    def _determine_channel_type(self, channel_info: Dict[str, Any]) -> str:
        """
        Determine channel type based on channel information.

        Args:
            channel_info: Dictionary containing channel information

        Returns:
            str: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        # Check if type is explicitly specified
        if "type" in channel_info and len(channel_info["type"]) > 0:
            ch_type = str(channel_info["type"][0])

            # Map EEGLAB channel types to emgio channel types
            if ch_type.upper() == "EMG":
                return "EMG"
            elif ch_type.upper() in ["ACC", "ACCELEROMETER"]:
                return "ACC"
            elif ch_type.upper() in ["GYRO", "GYROSCOPE"]:
                return "GYRO"
            elif ch_type.upper() in ["TRIG", "TRIGGER"]:
                return "TRIG"

        # If type is not specified or not recognized, try to determine from label
        if "labels" in channel_info and channel_info["labels"].size > 0:
            label = str(channel_info["labels"][0])
            label_upper = label.upper()

            if "EMG" in label_upper:
                return "EMG"
            elif "ACC" in label_upper:
                return "ACC"
            elif "GYRO" in label_upper:
                return "GYRO"
            elif "TRIG" in label_upper:
                return "TRIG"

        # Default to EMG if we can't determine the type
        return "EMG"

    def _process_channel_info(self, chanlocs: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process channel location information.

        Args:
            chanlocs: Array containing channel location information

        Returns:
            list: List of dictionaries containing channel information
        """
        channel_info_list = []

        # Process each channel
        for i in range(len(chanlocs[0])):
            channel_info = {}

            # Extract channel fields
            for field in chanlocs.dtype.names:
                # Get the field value for this channel
                field_value = chanlocs[0][i][field]

                # Process based on field name
                if field == "labels" and field_value.size > 0:
                    channel_info["label"] = str(field_value[0])
                elif field == "type" and field_value.size > 0:
                    channel_info["type"] = str(field_value[0])
                elif field == "X" and field_value.size > 0:
                    channel_info["X"] = float(field_value[0])
                elif field == "Y" and field_value.size > 0:
                    channel_info["Y"] = float(field_value[0])
                elif field == "Z" and field_value.size > 0:
                    channel_info["Z"] = float(field_value[0])

            # Determine channel type
            channel_info["channel_type"] = self._determine_channel_type(channel_info)

            # Add to list
            channel_info_list.append(channel_info)

        return channel_info_list

    def _process_events(self, events: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process event information.

        Args:
            events: Array containing event information

        Returns:
            list: List of dictionaries containing event information
        """
        event_list = []

        # Check if events exist
        if events.size == 0:
            return event_list

        # Process each event
        for i in range(len(events[0])):
            event_info = {}

            # Extract event fields
            for field in events.dtype.names:
                # Get the field value for this event
                field_value = events[0][i][field]

                # Process based on field name
                if field == "latency" and field_value.size > 0:
                    event_info["latency"] = float(field_value[0][0])
                elif field == "type" and field_value.size > 0:
                    event_info["type"] = str(field_value[0])
                elif field == "duration" and field_value.size > 0:
                    event_info["duration"] = (
                        float(field_value[0][0]) if field_value[0].size > 0 else 0
                    )
                elif field == "trial_type" and field_value.size > 0:
                    event_info["trial_type"] = (
                        str(field_value[0]) if field_value[0].size > 0 else ""
                    )

            # Add to list if it has required fields
            if "latency" in event_info and "type" in event_info:
                event_list.append(event_info)

        return event_list

    def load(self, filepath: str) -> EMG:
        """
        Load EMG data from EEGLAB .set file.

        Args:
            filepath: Path to the EEGLAB .set file

        Returns:
            EMG: EMG object containing the loaded data
        """
        try:
            # Load the .set file
            data = loadmat(filepath)

            # Create EMG object
            emg = EMG()

            # Extract and store metadata
            metadata = self._extract_metadata(data)
            for key, value in metadata.items():
                emg.set_metadata(key, value)

            # Store source file information
            emg.set_metadata("source_file", filepath)

            # Process channel information
            if "chanlocs" in data and data["chanlocs"].size > 0:
                channel_info_list = self._process_channel_info(data["chanlocs"])
            else:
                # If no channel locations, create default channel info
                channel_info_list = []
                for i in range(metadata.get("nbchan", 0)):
                    channel_info_list.append({"label": f"Channel{i + 1}", "channel_type": "EMG"})

            # Process event information
            if "event" in data and data["event"].size > 0:
                event_list = self._process_events(data["event"])
                emg.set_metadata("events", event_list)

            # Extract signal data
            if "data" in data and data["data"].size > 0:
                # Get data array
                signal_data = data["data"]

                # Get sampling rate
                srate = metadata.get("srate", 1000)

                # Create time array for index
                if "times" in data and data["times"].size > 0:
                    time_index = data["times"][0] / srate
                else:
                    # Create time array based on number of points and sampling rate
                    pnts = metadata.get("pnts", signal_data.shape[1])
                    time_index = np.arange(pnts) / srate

                # Create DataFrame with time index
                df = pd.DataFrame(index=time_index)

                # Add channels to EMG object
                for i, channel_info in enumerate(channel_info_list):
                    if i < signal_data.shape[0]:  # Make sure we have data for this channel
                        # Get channel data
                        channel_data = signal_data[i, :]

                        # Add to DataFrame
                        channel_label = channel_info.get("label", f"Channel{i + 1}")
                        df[channel_label] = channel_data

                # Set signals DataFrame
                emg.signals = df

                # Add channel information
                for i, channel_info in enumerate(channel_info_list):
                    if i < signal_data.shape[0]:  # Make sure we have data for this channel
                        channel_label = channel_info.get("label", f"Channel{i + 1}")

                        # Add channel info
                        emg.channels[channel_label] = {
                            "sample_frequency": srate,
                            "physical_dimension": "uV",  # Default unit for EEG/EMG
                            "prefilter": "n/a",
                            "channel_type": channel_info.get("channel_type", "EMG"),
                        }

                        # Add additional channel metadata
                        if "X" in channel_info:
                            emg.channels[channel_label]["X"] = channel_info["X"]
                        if "Y" in channel_info:
                            emg.channels[channel_label]["Y"] = channel_info["Y"]
                        if "Z" in channel_info:
                            emg.channels[channel_label]["Z"] = channel_info["Z"]

            return emg

        except Exception as e:
            raise ValueError(f"Error reading EEGLAB .set file: {str(e)}") from e
