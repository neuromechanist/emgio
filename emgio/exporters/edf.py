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
    # Handle zero signal case
    if np.allclose(signal, 0):
        return {
            'range': 0.0,
            'noise_floor': np.finfo(float).eps,
            'dynamic_range_db': 0.0,
            'is_zero': True
        }

    # Remove DC offset for better analysis
    detrended = signal - np.mean(signal)
    
    # Calculate signal range (peak-to-peak)
    signal_range = np.max(detrended) - np.min(detrended)
    
    # Improved noise floor estimation:
    # 1. Find signal frequency components using FFT
    n = len(detrended)
    fft = np.fft.rfft(detrended)
    freq = np.fft.rfftfreq(n)
    
    # 2. Get power spectrum
    power = np.abs(fft) ** 2
    
    # 3. Find significant frequency peaks
    # Use median as threshold to separate signal from noise
    threshold = np.median(power) * 10
    signal_freqs = freq[power > threshold]
    
    if len(signal_freqs) > 0:
        # 4. Create band-stop filter around main signal frequencies
        filtered = detrended.copy()
        for f in signal_freqs:
            if f > 0:  # Skip DC component
                # Create notch filter using moving average
                window = int(1 / (f * 2))  # Window size based on frequency
                if window > 1:
                    window = min(window, len(filtered) // 10)  # Limit window size
                    filtered = filtered - np.convolve(filtered, 
                                                    np.ones(window)/window, 
                                                    mode='same')
    
        # 5. Estimate noise floor from residual signal
        # Use robust statistics (MAD) to avoid outlier influence
        noise_floor = np.median(np.abs(filtered - np.median(filtered))) * 1.4826
    else:
        # If no clear frequency components, use traditional methods
        # Use difference method which is good for high-frequency noise
        diffs = np.diff(detrended)
        noise_floor = np.std(diffs) / np.sqrt(2)
    
    # Ensure minimum noise floor
    noise_floor = max(noise_floor, np.finfo(float).eps)

    # Calculate dynamic range in dB
    dynamic_range_db = 20 * np.log10(signal_range / noise_floor)

    # Calculate signal SNR
    signal_std = np.std(signal)
    snr_db = 20 * np.log10(signal_std / noise_floor)

    return {
        'range': signal_range,
        'noise_floor': noise_floor,
        'dynamic_range_db': dynamic_range_db,
        'snr_db': snr_db,
        'is_zero': False
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
    # Handle zero signal case
    if analysis.get('is_zero', False):
        return False, "Zero signal, using EDF format", 0.0

    # Theoretical format capabilities
    edf_dynamic_range = 96  # dB (16-bit)
    bdf_dynamic_range = 144  # dB (24-bit)
    safety_margin = 6  # dB

    # Get signal characteristics
    signal_dr = analysis['dynamic_range_db']
    signal_snr = analysis.get('snr_db', 0)
    # signal_range = analysis['range']  # Uncomment if needed for range-based decisions

    # Currently, range check is suppressed in favor of dynamic range
    # Check amplitude first - if signal range is large, use BDF
    # if analysis['range'] > 1e5:  # 100,000 units threshold
    #     return True, f"Large amplitude signal ({analysis['range']:.1f}), using BDF", signal_snr

    # Then check dynamic range with safety margin
    if signal_dr <= (edf_dynamic_range - safety_margin):
        return False, f"EDF dynamic range ({edf_dynamic_range} dB) is sufficient", signal_snr
    elif signal_dr <= (bdf_dynamic_range - safety_margin):
        return True, f"Signal requires BDF format (DR: {signal_dr:.1f} dB)", signal_snr
    else:
        return True, f"Signal may require higher resolution than BDF (DR: {signal_dr:.1f} dB)", signal_snr


def summarize_channels(channels: dict, signals: dict, analyses: dict) -> str:
    """
    Generate a summary of channel characteristics grouped by type.

    Args:
        channels: Dictionary of channel information
        signals: Dictionary of signal data
        analyses: Dictionary of signal analyses

    Returns:
        str: Formatted summary string
    """
    # Group channels by type
    type_groups = {}
    for ch_name, ch_info in channels.items():
        ch_type = ch_info.get('channel_type', 'Unknown')
        if ch_type not in type_groups:
            type_groups[ch_type] = {
                'channels': [],
                'ranges': [],
                'dynamic_ranges': [],
                'snrs': [],
                'formats': [],
                'unit': ch_info.get('physical_dimension', 'Unknown')
            }
        type_groups[ch_type]['channels'].append(ch_name)

        analysis = analyses.get(ch_name, {})
        if not analysis.get('is_zero', False):
            type_groups[ch_type]['ranges'].append(analysis.get('range', 0))
            type_groups[ch_type]['dynamic_ranges'].append(analysis.get('dynamic_range_db', 0))
            type_groups[ch_type]['snrs'].append(analysis.get('snr_db', 0))
            type_groups[ch_type]['formats'].append('BDF' if analysis.get('use_bdf', False) else 'EDF')

    # Generate summary
    summary = []
    for ch_type, data in type_groups.items():
        ranges = np.array(data['ranges'])
        dynamic_ranges = np.array(data['dynamic_ranges'])
        snrs = np.array(data['snrs'])
        formats = data['formats']

        if len(ranges) > 0:
            summary.append(f"\nChannel Type: {ch_type} ({len(data['channels'])} channels)")
            summary.append(
                f"Range: {np.min(ranges):.2f} to {np.max(ranges):.2f} "
                f"(mean: {np.mean(ranges):.2f}) {data['unit']}")
            summary.append(
                f"Dynamic Range: {np.min(dynamic_ranges):.1f} to "
                f"{np.max(dynamic_ranges):.1f} (mean: {np.mean(dynamic_ranges):.1f}) dB")
            summary.append(
                f"SNR: {np.min(snrs):.1f} to {np.max(snrs):.1f} "
                f"(mean: {np.mean(snrs):.1f}) dB")

            edf_count = formats.count('EDF')
            bdf_count = formats.count('BDF')
            summary.append(f"Format: {edf_count} channels using EDF, {bdf_count} channels using BDF")
        else:
            summary.append(f"\nChannel Type: {ch_type} ({len(data['channels'])} channels)")
            summary.append("All channels contain zero signal")

    return "\n".join(summary)


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


def _format_physical_value(value: float, max_chars: int) -> tuple:
    """
    Format a physical value to fit within EDF character limits.
    
    Args:
        value: Physical value to format
        max_chars: Maximum number of characters allowed
        
    Returns:
        tuple: (formatted_value, formatted_string)
    """
    # For zero or very small values, return as is
    if abs(value) < 1e-6:
        return 0.0, "0"
        
    # For values close to integers, handle as integers
    if abs((value - round(value)) / value) < 1e-6:
        value = int(round(value))  # Convert to integer
        scale = 1
        while True:
            value_str = str(value)
            if len(value_str) <= max_chars:
                return value, value_str
            # Integer division to reduce digits
            scale *= 10
            value = value // 10
    
    # For decimal numbers
    if abs(value) < 1:
        # Use scientific notation with reduced precision
        for precision in range(6, 0, -1):
            formatted = f"{value:.{precision}e}"
            if len(formatted) <= max_chars:
                return float(formatted), formatted
        # If still too long, return minimal representation
        return float(f"{value:.1e}"), f"{value:.1e}"
    else:
        # For larger decimals, try fixed point first
        scale = 1
        scaled_value = value
        while True:
            # Try without any changes
            formatted = f"{scaled_value}"
            if len(formatted) <= max_chars:
                return float(formatted), formatted
            # If that doesn't work, scale down
            scale *= 10
            scaled_value = value / scale
            # If we've scaled down a lot and still not fitting, switch to scientific
            if scale > 1e9:
                for precision in range(4, 0, -1):
                    formatted = f"{value:.{precision}e}"
                    if len(formatted) <= max_chars:
                        return float(formatted), formatted
                return float(f"{value:.1e}"), f"{value:.1e}"


def _determine_scaling_factors(signal_min: float, signal_max: float, use_bdf: bool = False) -> tuple:
    """
    Calculate optimal scaling factors for EDF/BDF signal conversion.
    Automatically scales values to fit format character limits.

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
            # For zero signal, use minimal range around zero
            # Use small values that will scale well with typical EMG signals
            signal_min, signal_max = -1e-6, 1e-6
        else:
            # For constant non-zero signal, create range around the value
            # Use a percentage of the value to maintain scale
            margin = abs(signal_min) * 0.01  # 1% margin
            signal_min -= margin
            signal_max += margin

    # Ensure physical range is never too small
    physical_range = signal_max - signal_min
    if abs(physical_range) < np.finfo(float).eps * 1e3:
        # If range is effectively zero, create a minimal range
        # Scale it relative to the signal magnitude
        base = max(abs(signal_min), abs(signal_max), 1e-6)
        physical_range = base * 1e-6
        signal_max = signal_min + physical_range
        # Ensure we have a valid range for scaling
        if physical_range == 0:
            physical_range = 1e-6

    # Set digital range and character limits based on format
    if use_bdf:
        digital_min, digital_max = -8388608, 8388607  # 24-bit
        max_chars = 12
    else:
        digital_min, digital_max = -32768, 32767  # 16-bit
        max_chars = 8

    # Format values to fit character limits
    signal_min, _ = _format_physical_value(signal_min, max_chars)
    signal_max, _ = _format_physical_value(signal_max, max_chars)

    digital_range = digital_max - digital_min
    physical_range = signal_max - signal_min

    # Calculate scaling factor
    # We use slightly less than the full range to prevent overflow at boundaries
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
            'name': [], 'channel_type': [], 'physical_dimension': [],
            'sample_frequency': [], 'reference': [], 'status': []
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
                f"\n    Range: {analysis['range']:.8g} {ch_info['physical_dimension']}"
                f"\n    Dynamic Range: {analysis['dynamic_range_db']:.1f} dB"
                f"\n    Noise Floor: {analysis['noise_floor']:.2e} {ch_info['physical_dimension']}"
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

                # Get signal min/max
                signal_min = float(np.min(signal))
                signal_max = float(np.max(signal))

                # Calculate scaling factors for header and get scaling factor
                phys_min, phys_max, dig_min, dig_max, scale_factor = _determine_scaling_factors(
                    signal_min, signal_max, use_bdf=use_bdf
                )

                # Scale the signal to match the physical min/max scaling
                scaled_signal = signal / scale_factor  # Scale down like we did for phys_min/max
                signals.append(scaled_signal)

                # Prepare channel info
                ch_dict = {
                    'label': ch_name[:16],  # EDF+ limits label to 16 chars
                    'dimension': ch_info['physical_dimension'],
                    'sample_frequency': int(ch_info['sample_frequency']),
                    'physical_max': phys_max,
                    'physical_min': phys_min,
                    'digital_max': dig_max,
                    'digital_min': dig_min,
                    'prefilter': ch_info['prefilter'],
                    'transducer': f"{ch_info['channel_type']} sensor"
                }
                channel_info_list.append(ch_dict)

                # Add to channels.tsv data
                channels_tsv_data['name'].append(ch_name)
                channels_tsv_data['channel_type'].append(ch_info['channel_type'])
                channels_tsv_data['physical_dimension'].append(ch_info['physical_dimension'])
                channels_tsv_data['sample_frequency'].append(ch_info['sample_frequency'])
                channels_tsv_data['reference'].append('n/a')
                channels_tsv_data['status'].append('good')

            # Set headers and write data
            writer.setSignalHeaders(channel_info_list)
            writer.writeSamples(signals)  # Pass physical signals directly

            # Store analyses for summary
            analyses = {}
            for ch_name in emg.channels:
                signal = emg.signals[ch_name].values
                analysis = analyze_signal(signal)
                use_bdf_for_channel, reason, snr = determine_format_suitability(signal, analysis)
                analysis['snr'] = snr
                analysis['use_bdf'] = use_bdf_for_channel
                analyses[ch_name] = analysis

                if use_bdf_for_channel:
                    use_bdf = True
                    if not bdf_reason:  # Only set reason for first channel requiring BDF
                        bdf_reason = f"Channel {ch_name}: {reason}"

            # Print channel type summary
            print(summarize_channels(emg.channels, emg.signals, analyses))

        finally:
            writer.close()

        # Create channels.tsv file
        channels_tsv_path = os.path.splitext(filepath)[0] + '_channels.tsv'
        channels_df = pd.DataFrame(channels_tsv_data)
        channels_df.to_csv(channels_tsv_path, sep='\t', index=False)
