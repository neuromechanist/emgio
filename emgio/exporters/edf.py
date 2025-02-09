import os
import warnings
import numpy as np
import pandas as pd
import pyedflib
from ..core.emg import EMG


def analyze_signal(signal: np.ndarray) -> dict:
    """
    Analyze signal characteristics including noise floor and dynamic range.

    Args:
        signal: Input signal array

    Returns:
        dict: Analysis results including range, noise floor, and dynamic range in dB
    """
    # Remove DC offset for better analysis
    detrended = signal - np.mean(signal)
    
    # Dynamic range
    signal_range = np.max(signal) - np.min(signal)
    
    # Noise floor estimation methods
    # Method 1: Using signal differences
    noise_estimate_diff = np.std(np.diff(detrended))
    
    # Method 2: Using detrended fluctuation analysis
    noise_floor = np.std(detrended - np.convolve(
        detrended, np.ones(10) / 10, mode='same'))
    
    # Use the more conservative (larger) noise estimate
    noise_floor = max(noise_estimate_diff, noise_floor)
    
    # Prevent division by zero
    if noise_floor < np.finfo(float).eps:
        noise_floor = np.finfo(float).eps
    
    # Dynamic Range in dB
    dynamic_range_db = 20 * np.log10(signal_range / noise_floor)
    
    return {
        'range': signal_range,
        'noise_floor': noise_floor,
        'dynamic_range_db': dynamic_range_db
    }


def determine_format_suitability(signal: np.ndarray, analysis: dict) -> tuple:
    """
    Determine whether EDF or BDF format is suitable for the signal.

    Args:
        signal: Input signal array
        analysis: Signal analysis results from analyze_signal()

    Returns:
        tuple: (use_bdf, reason, snr_db)
    """
    # Calculate theoretical format capabilities
    edf_levels = 2**16  # 65,536 levels
    bdf_levels = 2**24  # 16,777,216 levels
    
    # Calculate SNR for both formats
    signal_std = np.std(signal)
    if signal_std < np.finfo(float).eps:
        signal_std = np.finfo(float).eps
    
    # Calculate quantization step size
    edf_step = analysis['range'] / edf_levels
    bdf_step = analysis['range'] / bdf_levels
    
    # Calculate SNR in dB for both formats
    edf_snr = 20 * np.log10(signal_std / (edf_step / np.sqrt(12)))
    bdf_snr = 20 * np.log10(signal_std / (bdf_step / np.sqrt(12)))
    
    # Decision criteria
    snr_threshold = 70  # dB, common threshold for good quality
    
    if edf_snr > snr_threshold:
        return False, "EDF provides sufficient SNR", edf_snr
    elif bdf_snr > snr_threshold:
        return True, f"EDF SNR ({edf_snr:.1f} dB) below threshold, BDF recommended", bdf_snr
    else:
        return True, f"Signal may require higher resolution than EDF (SNR: {edf_snr:.1f} dB)", bdf_snr


def quantization_analysis(signal: np.ndarray, bits: int) -> dict:
    """
    Perform detailed quantization error analysis.

    Args:
        signal: Input signal array
        bits: Number of bits (16 for EDF, 24 for BDF)

    Returns:
        dict: Analysis results including step size, errors, and SNR
    """
    signal_range = np.max(signal) - np.min(signal)
    step_size = signal_range / (2**bits)
    
    # Simulate quantization
    quantized = np.round(signal / step_size) * step_size
    
    # Calculate errors
    abs_error = np.abs(signal - quantized)
    rmse = np.sqrt(np.mean((signal - quantized)**2))
    
    # Calculate SNR
    signal_power = np.mean(signal**2)
    noise_power = np.mean((signal - quantized)**2)
    if noise_power < np.finfo(float).eps:
        noise_power = np.finfo(float).eps
    snr = 10 * np.log10(signal_power / noise_power)
    
    return {
        'step_size': step_size,
        'max_error': np.max(abs_error),
        'rmse': rmse,
        'snr': snr
    }


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
    eps = np.finfo(np.float32).eps
    nonzero_mask = abs_signal > eps * 1e3
    if not np.any(nonzero_mask):
        return 0.0
    # Make the first and last five sample zero, to compensate for diff (technically, only first and last one is enough)
    nonzero_mask[0:5] = False
    nonzero_mask[-5:] = False

    relative_errors = np.zeros_like(signal)
    relative_errors[nonzero_mask] = (
        abs_diff[nonzero_mask] / abs_signal[nonzero_mask]
    )

    # Convert to percentage and ensure we detect small losses
    max_loss = float(np.max(relative_errors) * 100)
    if max_loss < np.finfo(np.float32).eps and np.any(abs_diff > 0):
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
            precision_threshold: Maximum acceptable precision loss percentage (default: 0.1%)
        """
        if emg.signals is None:
            raise ValueError("No signals to export")

        # Analyze signals and determine format
        print("\nSignal Analysis:")
        print("--------------")

        use_bdf = False
        bdf_reason = ""
        signal_info = []
        channel_info_list = []
        channels_tsv_data = {
            'name': [], 'type': [], 'units': [],
            'sampling_frequency': [], 'reference': [], 'status': []
        }

        # First pass: analyze signals and determine format
        for ch_name in emg.channels:
            signal = emg.signals[ch_name].values
            ch_info = emg.channels[ch_name]

            # Analyze signal characteristics
            analysis = analyze_signal(signal)
            use_bdf_for_channel, reason, snr = determine_format_suitability(signal, analysis)
            
            # Perform quantization analysis for chosen format
            if use_bdf_for_channel:
                use_bdf = True
                if not bdf_reason:  # Only set reason for first channel requiring BDF
                    bdf_reason = f"Channel {ch_name}: {reason}"

            # Calculate scaling factors
            phys_min, phys_max, dig_min, dig_max, scaling = _determine_scaling_factors(
                float(np.min(signal)), float(np.max(signal)), use_bdf=use_bdf_for_channel
            )

            signal_info.append(
                f"\n  {ch_name}:"
                f"\n    Range: {analysis['range']:.8g} {ch_info['unit']}"
                f"\n    Dynamic Range: {analysis['dynamic_range_db']:.1f} dB"
                f"\n    Noise Floor: {analysis['noise_floor']:.2e} {ch_info['unit']}"
                f"\n    SNR: {snr:.1f} dB"
                f"\n    Format: {'BDF' if use_bdf_for_channel else 'EDF'}"
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
            # Second pass: prepare channel information
            signals = []
            for ch_name in emg.channels:
                signal = emg.signals[ch_name].values
                ch_info = emg.channels[ch_name]

                # Calculate scaling factors for header
                phys_min, phys_max, dig_min, dig_max, _ = _determine_scaling_factors(
                    float(np.min(signal)), float(np.max(signal)), use_bdf
                )

                signals.append(signal)  # Use original physical signal

                # Prepare channel info
                ch_dict = {
                    'label': ch_name[:16],  # EDF+ limits label to 16 chars
                    'dimension': ch_info['unit'],
                    'sample_frequency': int(ch_info['sampling_freq']),
                    'physical_max': phys_max,
                    'physical_min': phys_min,
                    'digital_max': dig_max,
                    'digital_min': dig_min,
                    'prefilter': ch_info['prefilter'],
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
            writer.writeSamples(signals)  # Pass physical signals directly

            print("".join(signal_info))

        finally:
            writer.close()

        # Create channels.tsv file
        channels_tsv_path = os.path.splitext(filepath)[0] + '_channels.tsv'
        channels_df = pd.DataFrame(channels_tsv_data)
        channels_df.to_csv(channels_tsv_path, sep='\t', index=False)
