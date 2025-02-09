import os
import tempfile
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, Tuple
from .base import BaseImporter
from ..core.emg import EMG


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
        temp_dir = tempfile.mkdtemp(prefix='otb_')
        
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
            result = subprocess.run(['tar', 'xf', filepath, '-C', temp_dir], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                raise ValueError(f"tar command failed: {result.stderr}")
            
            print(f"Contents of {temp_dir}:")
            for item in os.listdir(temp_dir):
                print(f"- {item}")
            
            return temp_dir
            
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            raise ValueError(f"Could not extract OTB file: {str(e)}")

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
        metadata = {
            'device': {},
            'channels': {}
        }

        # Try different possible XML structures
        device = root.find('.//Device')  # Search recursively
        if device is None:
            device = root  # Try root element if no Device tag found

        print("\nDevice element:", device)
        print("Device attributes:", device.attrib)
        
        # Parse device attributes with various possible names
        attrs = device.attrib
        name = (attrs.get('Name') or attrs.get('name') or 
                attrs.get('DeviceName') or attrs.get('deviceName') or '')
        sampling_freq = float(attrs.get('SampleFrequency') or 
                            attrs.get('sampleFrequency') or 
                            attrs.get('SamplingFrequency') or 
                            attrs.get('samplingFrequency') or 0)
        ad_bits = int(attrs.get('ad_bits') or 
                     attrs.get('AD_bits') or 
                     attrs.get('AdBits') or 
                     attrs.get('adBits') or 16)
        
        metadata['device'] = {
            'name': name,
            'sampling_frequency': sampling_freq,
            'ad_bits': ad_bits
        }
        print("\nParsed device metadata:", metadata['device'])
        # Parse channels
        channels = device.find('.//Channels')  # Search recursively
        if channels is not None:
            print("\nFound Channels element")
            for adapter in channels.findall('.//Adapter'):
                print("\nAdapter:", adapter.attrib)
                adapter_id = adapter.attrib.get('ID', '')
                adapter_gain = float(adapter.attrib.get('Gain', 1.0))
                start_index = int(adapter.attrib.get('ChannelStartIndex', 0))

                for channel in adapter.findall('.//Channel'):
                    print("Channel:", channel.attrib)
                    idx = int(channel.attrib.get('Index', 0))
                    channel_num = start_index + idx + 1
                    
                    # Determine channel type based on adapter and channel info
                    EMG_adapter_models = ['Due', 'Muovi', 'Sessantaquattro', 'Novecento', 'Quattro', 'Quattrocento']
                    # check if one of the EMG adapter models is in the adapter ID
                    if any(model in adapter_id for model in EMG_adapter_models):
                        ch_type = 'EMG'
                    elif ('ACC' in adapter_id or 
                          'Acceleration' in channel.attrib.get('Description', '')):
                        ch_type = 'ACC'
                    elif ('GYRO' in adapter_id or 
                          'Gyroscope' in channel.attrib.get('Description', '')):
                        ch_type = 'GYRO'
                    elif 'Quaternion' in channel.attrib.get('ID', ''):
                        ch_type = 'QUAT'
                    elif 'Control' in adapter_id:
                        ch_type = 'CTRL'
                    else:
                        ch_type = 'OTHER'

                    # Determine unit based on channel type
                    if ch_type == 'EMG':
                        unit = 'mV'
                    elif ch_type in ['ACC', 'GYRO']:
                        unit = 'g'
                    elif ch_type == 'QUAT':
                        unit = 'rad'
                    else:
                        unit = 'a.u.'  # arbitrary units

                    metadata['channels'][f'CH{channel_num}'] = {
                        'type': ch_type,
                        'adapter': adapter_id,
                        'sampling_freq': metadata['device']['sampling_frequency'],
                        'unit': unit,
                        'gain': adapter_gain,
                        'description': channel.attrib.get('Description', ''),
                        'muscle': channel.attrib.get('Muscle', '')
                    }
                    print(f"Added channel CH{channel_num}:", metadata['channels'][f'CH{channel_num}'])
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
        device_name = metadata['device']['name']
        ad_bits = metadata['device']['ad_bits']
        sampling_freq = metadata['device']['sampling_frequency']
        num_channels = len(metadata['channels'])

        # Read binary data
        dtype = np.int16 if ad_bits == 16 else np.int32
        data = np.fromfile(sig_path, dtype=dtype)
        data = data.reshape(-1, num_channels).T

        # Apply device-specific scaling
        scaled_data = np.zeros_like(data, dtype=np.float64)
        for ch_num, ch_info in metadata['channels'].items():
            ch_idx = int(ch_num[2:]) - 1  # Extract channel number from 'CHx'
            
            # Get scaling factor based on device and adapter type
            if ch_info['adapter'] == 'Direct connection to Syncstation Input':
                scale = 0.1526  # PowerSupply=5V, AD_bits=16, Gain=0.5
            elif ch_info['adapter'] == 'AdapterLoadCell':
                scale = 0.00037217  # PowerSupply=5V, AD_bits=16, Gain=205
            elif ch_info['adapter'] in ['AdapterControl', 'AdapterQuaternions']:
                scale = 1.0  # No scaling for control signals
            elif ch_info['adapter'] == 'Sessantaquattro':
                if ad_bits == 16:
                    scale = 0.00050863  # PowerSupply=5V, AD_bits=16, Gain=150
                else:  # 24-bit
                    scale = 0.00028610  # PowerSupply=4.8V, AD_bits=24, Gain=1
            else:
                scale = 1.0  # Default no scaling
            
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

        try:
            # Extract OTB file
            temp_dir = self._extract_otb(filepath)

            # Find signal file first
            sig_files = [f for f in os.listdir(temp_dir) if f.endswith('.sig')]
            if not sig_files:
                raise ValueError("No signal file found in OTB archive")
            
            # Find corresponding XML file (same name but .xml extension)
            sig_base = os.path.splitext(sig_files[0])[0]
            xml_file = sig_base + '.xml'
            xml_path = os.path.join(temp_dir, xml_file)
            
            if not os.path.exists(xml_path):
                raise ValueError(f"Metadata file not found: {xml_file}")
            
            print(f"Using signal file: {sig_files[0]}")
            print(f"Parsing XML file: {xml_path}")
            metadata = self._parse_xml_metadata(xml_path)
            print("Device metadata:", metadata['device'])
            
            data, sampling_freq = self._read_signal_data(
                os.path.join(temp_dir, sig_files[0]),
                metadata
            )

            # Add channels to EMG object
            for ch_name, ch_info in metadata['channels'].items():
                ch_idx = int(ch_name[2:]) - 1  # Extract channel number from 'CHx'
                emg.add_channel(
                    name=ch_name,
                    data=data[ch_idx],
                    sampling_freq=ch_info['sampling_freq'],
                    unit=ch_info['unit'],
                    ch_type=ch_info['type']
                )

            # Add metadata
            emg.set_metadata('source_file', filepath)
            emg.set_metadata('device', metadata['device']['name'])
            emg.set_metadata('ad_bits', metadata['device']['ad_bits'])

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

        return emg
