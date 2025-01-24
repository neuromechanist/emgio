import os
import warnings
import pandas as pd
import pyedflib
from ..core.emg import EMG


def _truncate_value(value: float, channel_name: str, is_min: bool = True) -> float:
    """
    Truncate a value to fit EDF format (8 chars) and issue warning if precision is lost.

    Args:
        value: The value to truncate
        channel_name: Name of the channel for warning message
        is_min: Boolean indicating if this is a minimum value

    Returns:
        float: Truncated value
    """
    # Convert to mV if value is very small (likely in V)
    if abs(value) < 0.1:  # Threshold for conversion
        value = value * 1000

    # Format with fixed precision to avoid scientific notation
    value_str = f"{value:f}"

    # Handle negative numbers (keep the minus sign)
    is_negative = value < 0
    if is_negative:
        value_str = value_str[1:]  # Remove minus sign temporarily

    # Find decimal point position
    decimal_pos = value_str.find('.')
    if decimal_pos == -1:
        decimal_pos = len(value_str)

    # Determine how many digits we can keep
    max_digits = 7 if is_negative else 8  # Leave room for minus sign

    if len(value_str) > max_digits:
        # Keep as many significant digits as possible
        if decimal_pos > max_digits:
            # Large number: truncate after decimal
            truncated_str = value_str[:max_digits]
        else:
            # Small number: keep decimal point and some decimal places
            truncated_str = value_str[:decimal_pos + (max_digits - decimal_pos)]

        # Add back minus sign if needed
        if is_negative:
            truncated_str = '-' + truncated_str

        truncated = float(truncated_str)
        return truncated

    # Add back minus sign if needed
    if is_negative:
        value_str = '-' + value_str

    # Issue warning if value was truncated with exact amount of lost precision after converting to float
    if value_str != str(value):
        loss = abs((float(value_str) - value) / value) * 100
        loss_str = f"{loss:.4f}%"
        # issue warning if the two numbers are not equal beyond epsilon
        if not abs(float(value_str) - value) < 1e-9:
            warnings.warn(f"The {channel_name} value was truncated from {value} to {value_str} (loss: {loss_str})")

    return float(value_str)


class EDFExporter:
    """Exporter for EDF format with channels.tsv generation."""

    @staticmethod
    def export(emg: EMG, filepath: str) -> None:
        """
        Export EMG data to EDF format with corresponding channels.tsv file.

        Args:
            emg: EMG object containing the data
            filepath: Path to save the EDF file
        """
        if emg.signals is None:
            raise ValueError("No signals to export")

        # Convert all signals to mV if needed and check if BDF format is needed
        signals_list = []
        converted_signals = {}  # Store converted signals for later use
        for ch_name in emg.channels:
            signal = emg.signals[ch_name].copy()
            ch_info = emg.channels[ch_name]

            # Make a copy of the signal
            signal = signal.copy()

            # Convert V to mV for EMG signals
            if ch_info['unit'].lower() == 'v':
                signal = signal * 1000  # Convert to mV
                # Also convert the original signal values for min/max calculation
                signal_values = signal.values * 1000
            else:
                signal_values = signal.values

            signals_list.append(signal)
            converted_signals[ch_name] = signal_values

        # Perform precision analysis and determine format
        print("\nPrecision Analysis:")
        print("------------------")
        signal_info = []
        max_precision_loss = 0.0
        use_bdf = False
        bdf_reason = ""

        for signal, channel_name in zip(signals_list, list(emg.channels.keys())):
            min_val = float(signal.min())
            max_val = float(signal.max())

            # Check if values exceed EDF range
            if max_val > 32767 or min_val < -32768:
                use_bdf = True
                bdf_reason = f"Values for {channel_name} exceed EDF range (-32768 to 32767)"
                break

            if abs(max_val) < 0.1 and abs(min_val) < 0.1:
                min_val *= 1000
                max_val *= 1000
                print(f"\n  {channel_name}: Values converted from V to mV for better precision")

            truncated_min = _truncate_value(min_val, channel_name, True)
            truncated_max = _truncate_value(max_val, channel_name, False)

            # Calculate precision loss relative to original values
            min_loss = abs((min_val - truncated_min) / min_val) * 100 if min_val != 0 else 0
            max_loss = abs((max_val - truncated_max) / max_val) * 100 if max_val != 0 else 0
            current_loss = max(min_loss, max_loss)
            max_precision_loss = max(max_precision_loss, current_loss)

            if current_loss > 1.0 and not use_bdf:  # 1% threshold
                use_bdf = True
                bdf_reason = f"Precision loss for {channel_name} exceeds 1% threshold"

            signal_info.append(
                f"\n  {channel_name}:"
                f"\n    Min: {min_val:.8f} → {truncated_min:.8f} (loss: {min_loss:.4f}%)"
                f"\n    Max: {max_val:.8f} → {truncated_max:.8f} (loss: {max_loss:.4f}%)"
            )

        print("".join(signal_info))
        print(f"\nMaximum precision loss in EDF format would be {max_precision_loss:.4f}%")

        # Set file format based on analysis
        if use_bdf:
            filepath = os.path.splitext(filepath)[0] + '.bdf'
            print("\nUsing BDF format (24-bit) to preserve precision.")
            print(f"Reason: {bdf_reason}")
            print("Note: Use EDF format if you're okay with the truncated values shown above.")
            warnings.warn(f"Using BDF format to preserve precision. Reason: {bdf_reason}")
        else:
            filepath = os.path.splitext(filepath)[0] + '.edf'
            print("\nUsing EDF format (16-bit) as precision loss is within acceptable range.")

        # Create EDF/BDF file
        n_channels = len(emg.channels)
        if use_bdf:
            f = pyedflib.EdfWriter(filepath, n_channels=n_channels, file_type=pyedflib.FILETYPE_BDFPLUS)
        else:
            f = pyedflib.EdfWriter(filepath, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)

        # Prepare channel information
        channel_info_list = []
        channels_tsv_data = {
            'name': [],
            'type': [],
            'units': [],
            'sampling_frequency': [],
            'reference': [],
            'status': []
        }

        # Prepare all channel information
        for channel_name in emg.channels:
            ch_info = emg.channels[channel_name]
            signal = emg.signals[channel_name].values

            # Use converted signal values (already in mV if needed)
            signal_values = converted_signals[channel_name]
            unit = 'mV' if ch_info['unit'].lower() == 'v' else ch_info['unit']

            # Get min/max values from the converted signal
            signal_max = float(signal_values.max())
            signal_min = float(signal_values.min())

            # Always show truncation analysis for EDF format
            # Format values to check length without scientific notation
            value_max_str = f"{signal_max:f}"
            value_min_str = f"{signal_min:f}"

            # Check if truncation would be needed
            if len(value_max_str.replace('-', '')) > 8:
                truncated_max = _truncate_value(signal_max, channel_name, False)
                if not use_bdf:
                    signal_max = truncated_max
            if len(value_min_str.replace('-', '')) > 8:
                truncated_min = _truncate_value(signal_min, channel_name, True)
                if not use_bdf:
                    signal_min = truncated_min

            ch_dict = {
                'label': channel_name[:16],  # EDF+ limits label to 16 chars
                'dimension': unit,
                'sample_rate': int(ch_info['sampling_freq']),
                'physical_max': signal_max,
                'physical_min': signal_min,
                'digital_max': 8388607 if use_bdf else 32767,  # 24-bit vs 16-bit
                'digital_min': -8388608 if use_bdf else -32768,
                'prefilter': 'None',
                'transducer': f"{ch_info['type']} sensor"
            }
            channel_info_list.append(ch_dict)

            # Add to channels.tsv data
            channels_tsv_data['name'].append(channel_name)
            channels_tsv_data['type'].append(ch_info['type'])
            channels_tsv_data['units'].append(unit)
            channels_tsv_data['sampling_frequency'].append(ch_info['sampling_freq'])
            channels_tsv_data['reference'].append('n/a')
            channels_tsv_data['status'].append('good')

        # Set headers
        f.setSignalHeaders(channel_info_list)

        # Write all data at once using converted signals
        all_signals = [converted_signals[ch_name] for ch_name in emg.channels]

        f.writeSamples(all_signals)
        f.close()

        # Create channels.tsv file
        channels_tsv_path = os.path.splitext(filepath)[0] + '_channels.tsv'
        channels_df = pd.DataFrame(channels_tsv_data)
        channels_df.to_csv(channels_tsv_path, sep='\t', index=False)
