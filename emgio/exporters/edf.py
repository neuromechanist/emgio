import os
import warnings
import numpy as np
import pandas as pd
import pyedflib
from ..core.emg import EMG


def _determine_scaling_factors(signal_min: float, signal_max: float, use_bdf: bool = False) -> tuple:
    """
    Calculate optimal scaling factors for EDF/BDF signal conversion.

    Args:
        signal_min: Minimum value of the signal
        signal_max: Maximum value of the signal
        use_bdf: Whether to use BDF (24-bit) format

    Returns:
        tuple: (physical_min, physical_max, digital_min, digital_max, scaling_factor)
    """
    if signal_min > signal_max:
        signal_min, signal_max = signal_max, signal_min

    # Handle special cases
    if np.isclose(signal_min, signal_max):
        if np.isclose(signal_min, 0):
            signal_min, signal_max = -1, 1  # Default range for constant zero signal
        else:
            # For constant non-zero signal, create small range around it
            margin = abs(signal_min) * 0.001
            signal_min -= margin
            signal_max += margin

    # Set digital range based on format
    if use_bdf:
        digital_min, digital_max = -8388608, 8388607  # 24-bit
    else:
        digital_min, digital_max = -32768, 32767  # 16-bit

    digital_range = digital_max - digital_min
    physical_range = signal_max - signal_min

    # Calculate scaling factor to map physical range to digital range
    # We use slightly less than the full range to prevent overflow at boundaries
    # and ensure proper rounding behavior
    scaling_factor = (digital_range - 1) / physical_range

    return signal_min, signal_max, digital_min, digital_max, scaling_factor


def _calculate_precision_loss(signal: np.ndarray, scaling_factor: float, digital_min: int, digital_max: int) -> float:
    """
    Calculate precision loss when scaling signal to digital values.

    Args:
        signal: Original signal values
        scaling_factor: Scaling factor to convert to digital values
        digital_min: Minimum digital value
        digital_max: Maximum digital value

    Returns:
        float: Maximum relative precision loss as percentage
    """
    # Convert to integers (simulating digitization)
    scaled = np.round(signal * scaling_factor)
    digital_values = np.clip(scaled, digital_min, digital_max)
    reconstructed = digital_values / scaling_factor

    # Calculate relative error
    abs_diff = np.abs(signal - reconstructed)
    abs_signal = np.abs(signal)

    # Avoid division by zero and very small values
    eps = np.finfo(float).eps
    nonzero_mask = abs_signal > eps * 1e3
    if not np.any(nonzero_mask):
        return 0.0

    relative_errors = np.zeros_like(signal)
    relative_errors[nonzero_mask] = (
        abs_diff[nonzero_mask] / abs_signal[nonzero_mask]
    )

    # Convert to percentage and ensure we detect small losses
    max_loss = float(np.max(relative_errors) * 100)
    if max_loss < np.finfo(float).eps and np.any(abs_diff > 0):
        # If we have any difference but relative error is too small to measure,
        # return a small but non-zero value
        return 1e-6
    return max_loss


class EDFExporter:
    """Exporter for EDF format with channels.tsv generation."""

    @staticmethod
    def export(emg: EMG, filepath: str, precision_threshold: float = 0.01) -> None:
        """
        Export EMG data to EDF format with corresponding channels.tsv file.

        Args:
            emg: EMG object containing the data
            filepath: Path to save the EDF file
            precision_threshold: Maximum acceptable precision loss percentage (default: 0.01%)
        """
        if emg.signals is None:
            raise ValueError("No signals to export")

        # Analyze signals and determine format
        print("\nPrecision Analysis:")
        print("------------------")

        use_bdf = False
        bdf_reason = ""
        max_precision_loss = 0.0
        signal_info = []
        channel_info_list = []
        channels_tsv_data = {
            'name': [], 'type': [], 'units': [],
            'sampling_frequency': [], 'reference': [], 'status': []
        }

        # First pass: determine if BDF is needed
        for ch_name in emg.channels:
            signal = emg.signals[ch_name].values
            ch_info = emg.channels[ch_name]

            # Get signal range
            signal_min = float(np.min(signal))
            signal_max = float(np.max(signal))

            # Test scaling with EDF format
            phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(
                signal_min, signal_max, use_bdf=False
            )

            # Check if BDF is needed based on signal range first
            if abs(signal_max) > 32767 or abs(signal_min) < -32768:
                use_bdf = True
                bdf_reason = (f"Signal range for {ch_name} "
                            f"({signal_min:.2f} to {signal_max:.2f}) exceeds EDF limits")
                # Recalculate with BDF format
                phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(
                    signal_min, signal_max, use_bdf=True
                )

            # Calculate precision loss
            loss = _calculate_precision_loss(signal, scaling, dig_min, dig_max)
            max_precision_loss = max(max_precision_loss, loss)

            # Check if BDF is needed based on precision
            if not use_bdf and loss > precision_threshold:
                use_bdf = True
                bdf_reason = (f"Precision loss for {ch_name} ({loss:.4f}%) "
                            f"exceeds threshold ({precision_threshold}%)")
                # Recalculate with BDF format
                phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(
                    signal_min, signal_max, use_bdf=True
                )
                # Update loss calculation for info display
                loss = _calculate_precision_loss(signal, scaling, dig_min, dig_max)

            signal_info.append(
                f"\n  {ch_name}:"
                f"\n    Range: {signal_min:.8g} to {signal_max:.8g} {ch_info['unit']}"
                f"\n    Precision loss with {'BDF' if use_bdf else 'EDF'}: {loss:.4f}%"
            )

        # Set file format and create writer
        if use_bdf:
            filepath = os.path.splitext(filepath)[0] + '.bdf'
            print("\nUsing BDF format (24-bit) to preserve precision.")
            print(f"Reason: {bdf_reason}")
            warnings.warn(f"Using BDF format to preserve precision. Reason: {bdf_reason}")
            writer = pyedflib.EdfWriter(filepath, len(emg.channels), file_type=pyedflib.FILETYPE_BDFPLUS)
        else:
            filepath = os.path.splitext(filepath)[0] + '.edf'
            print("\nUsing EDF format (16-bit) as precision loss is within acceptable range.")
            writer = pyedflib.EdfWriter(filepath, len(emg.channels), file_type=pyedflib.FILETYPE_EDFPLUS)

        try:
            # Second pass: prepare channel information and scale signals
            scaled_signals = []
            for ch_name in emg.channels:
                signal = emg.signals[ch_name].values
                ch_info = emg.channels[ch_name]

                # Calculate scaling factors
                phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(
                    float(np.min(signal)), float(np.max(signal)), use_bdf
                )

                # Scale signal (simulate digitization)
                scaled = np.round(signal * scaling)
                scaled_signal = np.clip(scaled, dig_min, dig_max)
                scaled_signals.append(scaled_signal)

                # Prepare channel info
                ch_dict = {
                    'label': ch_name[:16],  # EDF+ limits label to 16 chars
                    'dimension': ch_info['unit'],
                    'sample_frequency': int(ch_info['sampling_freq']),
                    'physical_max': phys_max,
                    'physical_min': phys_min,
                    'digital_max': dig_max,
                    'digital_min': dig_min,
                    'prefilter': 'None',
                    'transducer': f"{ch_info['type']} sensor"
                }
                channel_info_list.append(ch_dict)

                # Add to channels.tsv data
                channels_tsv_data['name'].append(ch_name)
                channels_tsv_data['type'].append(ch_info['type'])
                channels_tsv_data['units'].append(ch_info['unit'])
                channels_tsv_data['sampling_frequency'].append(ch_info['sampling_freq'])
                channels_tsv_data['reference'].append('n/a')
                channels_tsv_data['status'].append('good')

            # Set headers and write data
            writer.setSignalHeaders(channel_info_list)
            writer.writeSamples(scaled_signals)

            print("".join(signal_info))
            print(f"\nMaximum precision loss: {max_precision_loss:.4f}%")

        finally:
            writer.close()

        # Create channels.tsv file
        channels_tsv_path = os.path.splitext(filepath)[0] + '_channels.tsv'
        channels_df = pd.DataFrame(channels_tsv_data)
        channels_df.to_csv(channels_tsv_path, sep='\t', index=False)
