import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

import numpy as np

from ..core.emg import EMG
from .base import BaseImporter


class OTBImporter(BaseImporter):
    """Importer for OTB/OTB+ EMG system data."""

    def _extract_otb(self, filepath: str) -> str:
        """
        Extract OTB/OTB+ file to a temporary directory.

        Args:
            filepath: Path to the OTB/OTB+ file

        Returns:
            Path to the temporary directory containing extracted files
        """
        temp_dir = tempfile.mkdtemp(prefix="otb_")

        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"OTB file not found: {filepath}")

        print(f"Processing file: {filepath}")
        print(f"File exists: {os.path.exists(filepath)}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        print(f"Temp directory: {temp_dir}")

        try:
            # Use system tar command
            print(f"Extracting {filepath} to {temp_dir}")
            result = subprocess.run(
                ["tar", "xf", filepath, "-C", temp_dir], capture_output=True, text=True
            )

            if result.returncode != 0:
                raise ValueError(f"tar command failed: {result.stderr}")

            print(f"Contents of {temp_dir}:")
            for item in os.listdir(temp_dir):
                print(f"- {item}")

            return temp_dir

        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            raise ValueError(f"Could not extract OTB file: {str(e)}") from e

    def _parse_xml_metadata(self, xml_path: str) -> Dict:
        """
        Parse XML metadata file to extract device and channel information.

        Args:
            xml_path: Path to the XML metadata file

        Returns:
            Dictionary containing device and channel metadata
        """
        print("\nParsing XML file:", xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        print("\nRoot tag:", root.tag)
        print("Root attributes:", root.attrib)

        # Initialize metadata structure
        metadata = {"device": {}, "channels": {}}

        # Try different possible XML structures
        device = root.find(".//Device")  # Search recursively
        if device is None:
            device = root  # Try root element if no Device tag found

        print("\nDevice element:", device)
        print("Device attributes:", device.attrib)

        # Parse device attributes with various possible names
        attrs = device.attrib
        name = (
            attrs.get("Name")
            or attrs.get("name")
            or attrs.get("DeviceName")
            or attrs.get("deviceName")
            or ""
        )
        sampling_freq = float(
            attrs.get("SampleFrequency")
            or attrs.get("sampleFrequency")
            or attrs.get("SamplingFrequency")
            or attrs.get("samplingFrequency")
            or 0
        )
        ad_bits = int(
            attrs.get("ad_bits")
            or attrs.get("AD_bits")
            or attrs.get("AdBits")
            or attrs.get("adBits")
            or 16
        )

        metadata["device"] = {"name": name, "sampling_frequency": sampling_freq, "ad_bits": ad_bits}
        print("\nParsed device metadata:", metadata["device"])
        # Parse channels
        channels = device.find(".//Channels")  # Search recursively
        if channels is not None:
            print("\nFound Channels element")
            for adapter in channels.findall(".//Adapter"):
                print("\nAdapter:", adapter.attrib)
                adapter_id = adapter.attrib.get("ID", "")
                adapter_gain = float(adapter.attrib.get("Gain", 1.0))
                start_index = int(adapter.attrib.get("ChannelStartIndex", 0))

                for channel in adapter.findall(".//Channel"):
                    print("Channel:", channel.attrib)
                    idx = int(channel.attrib.get("Index", 0))
                    channel_num = start_index + idx + 1

                    # Determine channel type based on adapter and channel info
                    EMG_adapter_models = [
                        "Due",
                        "Muovi",
                        "Sessantaquattro",
                        "Novecento",
                        "Quattro",
                        "Quattrocento",
                    ]
                    # check if one of the EMG adapter models is in the adapter ID
                    if any(model in adapter_id for model in EMG_adapter_models):
                        ch_type = "EMG"
                    elif "ACC" in adapter_id or "Acceleration" in channel.attrib.get(
                        "Description", ""
                    ):
                        ch_type = "ACC"
                    elif "GYRO" in adapter_id or "Gyroscope" in channel.attrib.get(
                        "Description", ""
                    ):
                        ch_type = "GYRO"
                    elif "Quaternion" in channel.attrib.get("ID", ""):
                        ch_type = "QUAT"
                    elif "Control" in adapter_id:
                        ch_type = "CTRL"
                    else:
                        ch_type = "OTHER"

                    # Determine unit based on channel type
                    if ch_type == "EMG":
                        unit = "mV"
                    elif ch_type in ["ACC", "GYRO"]:
                        unit = "g"
                    elif ch_type == "QUAT":
                        unit = "rad"
                    else:
                        unit = "a.u."  # arbitrary units

                    # Construct prefiltering string if filter info is available
                    hp_filter = adapter.attrib.get("HighPassFilter")
                    lp_filter = adapter.attrib.get("LowPassFilter")
                    prefilter = "n/a"
                    if hp_filter or lp_filter:
                        filters = []
                        if hp_filter:
                            filters.append(f"HP:{hp_filter}Hz")
                        if lp_filter:
                            filters.append(f"LP:{lp_filter}Hz")
                        prefilter = " ".join(filters)

                    metadata["channels"][f"CH{channel_num}"] = {
                        "channel_type": ch_type,
                        "adapter": adapter_id,
                        "sample_frequency": metadata["device"]["sampling_frequency"],
                        "physical_dimension": unit,
                        "gain": adapter_gain,
                        "description": channel.attrib.get("Description", ""),
                        "muscle": channel.attrib.get("Muscle", ""),
                        "prefilter": prefilter,
                    }
                    print(
                        f"Added channel CH{channel_num}:", metadata["channels"][f"CH{channel_num}"]
                    )
        else:
            print("\nNo Channels element found")

        return metadata

    def _read_signal_data(self, sig_path: str, metadata: Dict) -> Tuple[np.ndarray, float]:
        """
        Read binary signal data and apply appropriate scaling.

        Args:
            sig_path: Path to the signal file
            metadata: Dictionary containing device and channel metadata

        Returns:
            Tuple containing:
                - numpy array of scaled signal data
                - sampling frequency
        """
        ad_bits = metadata["device"]["ad_bits"]
        sampling_freq = metadata["device"]["sampling_frequency"]
        num_channels = len(metadata["channels"])

        # Always read as 16-bit initially (like MATLAB's 'short')
        data = np.fromfile(sig_path, dtype=np.int16)
        data = data.reshape(-1, num_channels).T

        # For 24-bit data, reconstruct from two 16-bit values
        if ad_bits == 24:
            # Each 24-bit value is stored as two 16-bit values
            # Reshape to get pairs of 16-bit values
            data = data.reshape(num_channels, -1, 2)
            # Combine the pairs into 24-bit values (stored in 32-bit int)
            data = (data[:, :, 0].astype(np.int32) << 8) | (data[:, :, 1] & 0xFF)
            # Sign extend from 24 to 32 bits
            data = np.where(data & 0x800000, data | ~0xFFFFFF, data)

        # Apply scaling according to MATLAB reference
        power_supply = 3.3  # Voltage reference is 3.3V
        scaled_data = np.zeros_like(data, dtype=np.float64)
        for ch_num, ch_info in metadata["channels"].items():
            ch_idx = int(ch_num[2:]) - 1  # Extract channel number from 'CHx'
            gain = ch_info["gain"]

            # Apply scaling formula: data * PowerSupply / (2^ad_bits) * 1000 / gain
            # 1000 factor converts to mV
            if ch_info["channel_type"] == "EMG":
                scale = power_supply / (2**ad_bits) * 1000 / gain
            else:
                # For non-EMG channels (like control signals), use no scaling
                scale = 1.0

            scaled_data[ch_idx] = data[ch_idx] * scale

        return scaled_data, sampling_freq

    def load(self, filepath: str) -> EMG:
        """
        Load EMG data from OTB/OTB+ file.

        Args:
            filepath: Path to the OTB/OTB+ file

        Returns:
            EMG: EMG object containing the loaded data
        """
        # Create EMG object
        emg = EMG()
        temp_dir = None

        try:
            # Extract OTB file
            temp_dir = self._extract_otb(filepath)

            # Find signal file first
            sig_files = [f for f in os.listdir(temp_dir) if f.endswith(".sig")]
            if not sig_files:
                raise ValueError("No signal file found in OTB archive")

            # Find corresponding XML file (same name but .xml extension)
            sig_base = os.path.splitext(sig_files[0])[0]
            xml_file = sig_base + ".xml"
            xml_path = os.path.join(temp_dir, xml_file)

            if not os.path.exists(xml_path):
                raise ValueError(f"Metadata file not found: {xml_file}")

            print(f"Using signal file: {sig_files[0]}")
            print(f"Parsing XML file: {xml_path}")
            metadata = self._parse_xml_metadata(xml_path)
            print("Device metadata:", metadata["device"])

            data, sampling_freq = self._read_signal_data(
                os.path.join(temp_dir, sig_files[0]), metadata
            )

            # Add channels to EMG object
            for ch_name, ch_info in metadata["channels"].items():
                ch_idx = int(ch_name[2:]) - 1  # Extract channel number from 'CHx'
                emg.add_channel(
                    label=ch_name,
                    data=data[ch_idx],
                    sample_frequency=ch_info["sample_frequency"],
                    physical_dimension=ch_info["physical_dimension"],
                    prefilter=ch_info["prefilter"],
                    channel_type=ch_info["channel_type"],
                )

            # Add metadata
            emg.set_metadata("source_file", filepath)
            emg.set_metadata("device", metadata["device"]["name"])
            emg.set_metadata("signal_resolution", metadata["device"]["ad_bits"])

        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                import shutil

                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp directory {temp_dir}: {str(e)}")

        return emg
