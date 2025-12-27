import os
import warnings
from typing import Literal, Optional

# import logging  # Add logging import - Removed as unused in this file
import numpy as np
import pandas as pd
import pyedflib

from ..analysis.signal import analyze_signal, determine_format_suitability
from ..core.emg import EMG


def _format_physical_value(value: float, max_chars: int) -> tuple:
    """
    Format a physical value to fit within EDF character limits.

    Args:
        value: Physical value to format
        max_chars: Maximum number of characters allowed

    Returns:
        tuple: (formatted_value, formatted_string)
    """
    # Handle NaN values
    if np.isnan(value):
        return 0.0, "0"

    # For zero or very small values, return as is
    if abs(value) < 1e-6:
        return 0.0, "0"

    # For values close to integers, handle as integers
    try:
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
    except (ValueError, ZeroDivisionError):
        # Handle any other numerical issues
        return 0.0, "0"

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


def _determine_scaling_factors(
    signal_min: float, signal_max: float, use_bdf: bool = False
) -> tuple:
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
    # Handle NaN values
    if np.isnan(signal_min) or np.isnan(signal_max):
        signal_min = -1e-6 if np.isnan(signal_min) else signal_min
        signal_max = 1e-6 if np.isnan(signal_max) else signal_max

    if signal_min > signal_max:
        signal_min, signal_max = signal_max, signal_min

    # Set digital range based on format
    if use_bdf:
        digital_min, digital_max = -8388608, 8388607  # 24-bit
        max_chars = 12
    else:
        digital_min, digital_max = -32768, 32767  # 16-bit
        max_chars = 8

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
            # Don't normalize constant signals - this preserves test behavior
            return signal_min, signal_max, digital_min, digital_max, digital_max * 1.0

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

    # For high dynamic range signals, preserve the original range
    # This is critical for maintaining the dynamic range in the exported file
    if (signal_max - signal_min) > 1e5 or (signal_max / max(abs(signal_min), 1e-10)) > 1e5:
        # High dynamic range detected - preserve it for BDF format
        if use_bdf:
            # For BDF, we can handle the full range directly
            # Just ensure the values fit within character limits
            signal_min, _ = _format_physical_value(signal_min, max_chars)
            signal_max, _ = _format_physical_value(signal_max, max_chars)

            digital_range = digital_max - digital_min
            physical_range = signal_max - signal_min

            # Calculate scaling factor to use full digital range
            scaling_factor = digital_range / physical_range

            return signal_min, signal_max, digital_min, digital_max, scaling_factor

    # Only normalize extreme values that would cause problems with EDF/BDF format
    # This preserves the original scaling for most signals while handling extreme cases
    if (
        abs(signal_min) > 1e6
        or abs(signal_max) > 1e6
        or abs(signal_min) < 1e-6
        or abs(signal_max) < 1e-6
    ):
        # For extreme values, normalize to a reasonable range
        # But preserve the original ratio between min and max
        ratio = abs(signal_max / signal_min) if signal_min != 0 else 1.0

        if ratio > 1e6 and not use_bdf:  # Very large ratio, use a more balanced range for EDF only
            signal_min = -1.0
            signal_max = 1.0
        else:  # Preserve ratio but scale to reasonable values
            if abs(signal_min) > 1e6 or abs(signal_max) > 1e6:  # Too large
                scale_factor = max(abs(signal_min), abs(signal_max)) / 1000.0
                signal_min /= scale_factor
                signal_max /= scale_factor
            elif abs(signal_min) < 1e-6 or abs(signal_max) < 1e-6:  # Too small
                scale_factor = 1e-3 / max(abs(signal_min), abs(signal_max))
                signal_min *= scale_factor
                signal_max *= scale_factor

    # Format values to fit character limits
    signal_min, _ = _format_physical_value(signal_min, max_chars)
    signal_max, _ = _format_physical_value(signal_max, max_chars)

    digital_range = digital_max - digital_min
    physical_range = signal_max - signal_min

    # Calculate scaling factor
    # We use slightly less than the full range to prevent overflow at boundaries
    scaling_factor = (digital_range - 1) / physical_range

    return signal_min, signal_max, digital_min, digital_max, scaling_factor


def _calculate_precision_loss(
    signal: np.ndarray, scaling_factor: float, digital_min: int, digital_max: int
) -> float:
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
    relative_errors[nonzero_mask] = abs_diff[nonzero_mask] / abs_signal[nonzero_mask]

    # Convert to percentage and ensure we detect small losses
    max_loss = float(np.max(relative_errors) * 100)
    if max_loss < np.finfo(np.float32).eps and np.any(abs_diff > 0):
        # If we have any difference but relative error is too small to measure,
        # return a small but non-zero value
        return 1e-6
    return max_loss


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
        ch_type = ch_info.get("channel_type", "Unknown")
        if ch_type not in type_groups:
            type_groups[ch_type] = {
                "channels": [],
                "ranges": [],
                "dynamic_ranges": [],
                "snrs": [],
                "formats": [],
                "unit": ch_info.get("physical_dimension", "Unknown"),
            }
        type_groups[ch_type]["channels"].append(ch_name)

        analysis = analyses.get(ch_name, {})
        if not analysis.get("is_zero", False):
            type_groups[ch_type]["ranges"].append(analysis.get("range", 0))
            type_groups[ch_type]["dynamic_ranges"].append(analysis.get("dynamic_range_db", 0))
            type_groups[ch_type]["snrs"].append(analysis.get("snr_db", 0))
            type_groups[ch_type]["formats"].append(
                "BDF" if analysis.get("use_bdf", False) else "EDF"
            )

    # Generate summary
    summary = []
    for ch_type, data in type_groups.items():
        ranges = np.array(data["ranges"])
        dynamic_ranges = np.array(data["dynamic_ranges"])
        snrs = np.array(data["snrs"])
        formats = data["formats"]

        if len(ranges) > 0:
            summary.append(f"\nChannel Type: {ch_type} ({len(data['channels'])} channels)")
            summary.append(
                f"Range: {np.min(ranges):.2f} to {np.max(ranges):.2f} "
                f"(mean: {np.mean(ranges):.2f}) {data['unit']}"
            )
            summary.append(
                f"Dynamic Range: {np.min(dynamic_ranges):.1f} to "
                f"{np.max(dynamic_ranges):.1f} (mean: {np.mean(dynamic_ranges):.1f}) dB"
            )
            summary.append(
                f"SNR: {np.min(snrs):.1f} to {np.max(snrs):.1f} (mean: {np.mean(snrs):.1f}) dB"
            )

            edf_count = formats.count("EDF")
            bdf_count = formats.count("BDF")
            summary.append(
                f"Format: {edf_count} channels using EDF, {bdf_count} channels using BDF"
            )
        else:
            summary.append(f"\nChannel Type: {ch_type} ({len(data['channels'])} channels)")
            summary.append("All channels contain zero signal")

    return "\n".join(summary)


class EDFExporter:
    """Exporter for EDF format with channels.tsv generation."""

    @staticmethod
    def export(
        emg: EMG,
        filepath: str,
        precision_threshold: float = 0.01,
        method: str = "both",
        fft_noise_range: tuple = None,
        svd_rank: int = None,
        format: Literal["auto", "edf", "bdf"] = "auto",
        bypass_analysis: bool = False,
        events_df: Optional[pd.DataFrame] = None,
        create_channels_tsv: bool = True,
        **kwargs,
    ) -> None:
        """
        Export EMG data to EDF/BDF format with optional BIDS-compliant channels.tsv file.

        Args:
            emg: EMG object containing the data
            filepath: Path to save the EDF/BDF file
            precision_threshold: Maximum acceptable precision loss percentage (default: 0.01%)
            method: Method for signal analysis ('svd', 'fft', or 'both')
                'svd': Uses Singular Value Decomposition for noise floor estimation
                'fft': Uses Fast Fourier Transform for noise floor estimation
                'both': Uses both methods and takes the minimum noise floor (default)
            fft_noise_range: Optional tuple (min_freq, max_freq) specifying frequency range for noise in FFT method
            svd_rank: Optional manual rank cutoff for signal/noise separation in SVD method
            format: Format to use ('auto', 'edf', or 'bdf'). Default is 'auto'.
                    If 'edf' or 'bdf' is specified, that format will be used directly.
                    If 'auto', the format (EDF/16-bit or BDF/24-bit) is chosen based
                    on signal analysis to minimize precision loss while preferring EDF
                    if sufficient.
            bypass_analysis: If True, skip the signal analysis step. Requires format
                             to be explicitly set to 'edf' or 'bdf'. (default: False)
            events_df: Optional DataFrame containing events/annotations to write.
                     Columns should include 'onset', 'duration', 'description'.
                     If None or empty, no annotations are written.
            create_channels_tsv: If True, create a BIDS-compliant channels.tsv file (default: True)
            **kwargs: Additional arguments for the exporter
        """
        if emg.signals is None:
            raise ValueError("No signals to export")

        print("\nSignal Analysis:")
        print("--------------")

        # Initialize format decision variables
        use_bdf = False
        bdf_reason = ""
        format_decision_made = False

        # --- Format Decision and Bypass Check ---
        if bypass_analysis and format.lower() == "auto":
            raise ValueError("Cannot bypass analysis when format is set to 'auto'.")

        if format.lower() == "bdf":
            use_bdf = True
            format_decision_made = True
            if not bypass_analysis:
                print("\nUser specified BDF format (24-bit).")
            else:
                # Log critical only if bypassing, already logged in EMG.to_edf
                pass  # logging.log(logging.CRITICAL, "Skipping analysis, using specified BDF format.")
        elif format.lower() == "edf":
            use_bdf = False
            format_decision_made = True
            if not bypass_analysis:
                print("\nUser specified EDF format (16-bit).")
            else:
                # Log critical only if bypassing, already logged in EMG.to_edf
                pass  # logging.log(logging.CRITICAL, "Skipping analysis, using specified EDF format.")
        elif format.lower() != "auto":
            warnings.warn(
                f"Unknown format: {format}. Valid options are 'auto', 'edf', or 'bdf'. Using 'auto'.",
                stacklevel=2,
            )
            format = "auto"  # Default to auto if invalid format given
            bypass_analysis = False  # Cannot bypass if format is auto

        signal_analyses = {}
        signal_info_strings = []

        # --- Conditional Signal Analysis ---
        if not bypass_analysis:
            # Analyze signals (needed for summary and potentially for 'auto' format decision)
            for ch_name in emg.channels:
                signal = emg.signals[ch_name].values
                ch_info = emg.channels[ch_name]

                # Analyze signal characteristics
                analysis = analyze_signal(
                    signal, method=method, fft_noise_range=fft_noise_range, svd_rank=svd_rank
                )
                recommend_bdf, reason, snr = determine_format_suitability(signal, analysis)
                analysis["snr"] = snr
                analysis["recommend_bdf"] = recommend_bdf
                analysis["reason"] = reason
                signal_analyses[ch_name] = analysis  # Store analysis for later summary

                # If format is 'auto', check if any channel recommends BDF
                if format == "auto" and recommend_bdf:
                    use_bdf = True  # Switch to BDF if any channel needs it
                    if not bdf_reason:  # Capture the first reason
                        bdf_reason = f"Channel '{ch_name}': {reason}"

                # Prepare info string for printing later
                signal_info_strings.append(
                    f"\n  {ch_name}:"
                    f"\n    Range: {analysis['range']:.8g} {ch_info['physical_dimension']}"
                    f"\n    Dynamic Range: {analysis['dynamic_range_db']:.1f} dB"
                    f"\n    Noise Floor: {analysis['noise_floor']:.2e} {ch_info['physical_dimension']}"
                    f"\n    SNR: {snr:.1f} dB"
                    f"\n    Method: {analysis.get('method', 'svd')}"
                    f"\n    Recommended Format: {'BDF' if recommend_bdf else 'EDF'} ({reason})"
                )

            # Print analysis details after deciding the format
            for info_str in signal_info_strings:
                print(info_str)

            # Final format decision message for 'auto' mode
            if format == "auto":
                if use_bdf:
                    print(
                        "\nUsing BDF format (24-bit) based on signal analysis to preserve precision."
                    )
                    print(f"Reason: {bdf_reason}")
                    warnings.warn(
                        f"Using BDF format based on signal analysis. Reason: {bdf_reason}",
                        stacklevel=2,
                    )
                else:
                    print(
                        "\nUsing EDF format (16-bit) based on signal analysis (precision within acceptable range)."
                    )
        # else: # bypass_analysis is True - logging handled in EMG.to_edf
        #     pass # logging.log(logging.CRITICAL, "Signal analysis bypassed.")

        # Set file format and create writer
        # Initialize BIDS-compliant channels.tsv data structure
        # Required columns in BIDS order: name, type, units
        channels_tsv_data = {
            "name": [],
            "type": [],
            "units": [],
            "sampling_frequency": [],
            "reference": [],
            "status": [],
        }
        channel_info_list = []

        if use_bdf:
            filepath = os.path.splitext(filepath)[0] + ".bdf"
            filetype = pyedflib.FILETYPE_BDFPLUS
        else:
            filepath = os.path.splitext(filepath)[0] + ".edf"
            filetype = pyedflib.FILETYPE_EDFPLUS

        writer = pyedflib.EdfWriter(filepath, len(emg.channels), file_type=filetype)

        try:
            # Prepare channel information and signals for writing
            signals_to_write = []
            for _i, ch_name in enumerate(emg.channels):
                signal = emg.signals[ch_name].values
                ch_info = emg.channels[ch_name]
                # No need for full analysis result for scaling factors anymore

                # Get signal min/max for scaling factor calculation
                signal_min = float(np.min(signal))
                signal_max = float(np.max(signal))

                # Calculate scaling factors for header based on the chosen format (use_bdf)
                phys_min, phys_max, dig_min, dig_max, scale_factor = _determine_scaling_factors(
                    signal_min, signal_max, use_bdf=use_bdf
                )

                # Prepare the signal data (handle NaNs, but DO NOT pre-scale)
                physical_signal = signal.copy()
                physical_signal = np.nan_to_num(physical_signal, nan=0.0)
                signals_to_write.append(physical_signal)

                # Prepare channel header dictionary
                ch_dict = {
                    "label": ch_name[:16],  # EDF+ limits label to 16 chars
                    "dimension": ch_info["physical_dimension"],
                    "sample_frequency": int(ch_info["sample_frequency"]),
                    "physical_max": phys_max,
                    "physical_min": phys_min,
                    "digital_max": dig_max,
                    "digital_min": dig_min,
                    "prefilter": ch_info["prefilter"],
                    "transducer": f"{ch_info.get('channel_type', 'Unknown')} sensor",  # Use get for safety
                }
                channel_info_list.append(ch_dict)

                # Add to BIDS-compliant channels.tsv data
                channels_tsv_data["name"].append(ch_name)

                # Map channel type to BIDS-compliant uppercase values
                ch_type = ch_info.get("channel_type", "Unknown").upper()
                # Map common channel types to BIDS standard values
                if ch_type in [
                    "EMG",
                    "EEG",
                    "MEG",
                    "ECG",
                    "EOG",
                    "VEOG",
                    "HEOG",
                    "REF",
                    "TRIG",
                    "MISC",
                ]:
                    bids_type = ch_type
                elif "EMG" in ch_type:
                    bids_type = "EMG"
                elif "ACC" in ch_type or "ACCEL" in ch_type:
                    bids_type = "MISC"
                elif "GYRO" in ch_type:
                    bids_type = "MISC"
                else:
                    bids_type = "MISC"

                channels_tsv_data["type"].append(bids_type)
                channels_tsv_data["units"].append(ch_info["physical_dimension"])
                channels_tsv_data["sampling_frequency"].append(ch_info["sample_frequency"])
                channels_tsv_data["reference"].append(
                    "n/a"
                )  # Can be updated if reference info is available
                channels_tsv_data["status"].append(
                    "good"
                )  # Default to good, can be updated based on signal quality

            # Set headers and write data (pass physical signals)
            writer.setSignalHeaders(channel_info_list)
            writer.writeSamples(np.array(signals_to_write))

            # Write annotations if provided
            if events_df is not None and not events_df.empty:
                for _index, row in events_df.iterrows():
                    try:
                        # pyedflib uses onset, duration, description
                        onset = float(row["onset"])
                        duration = float(row["duration"])
                        description = str(row["description"])
                        # Write annotation for all channels (-1)
                        writer.writeAnnotation(onset, duration, description)
                    except KeyError as e:
                        warnings.warn(
                            f"Skipping event due to missing column: {e}. Event data: {row}",
                            stacklevel=2,
                        )
                    except (TypeError, ValueError) as e:
                        warnings.warn(
                            f"Skipping event due to invalid data type: {e}. Event data: {row}",
                            stacklevel=2,
                        )

            # Explicitly flush and close the writer to ensure all data is written
            writer.close()

            # Wait a moment to ensure file system operations are complete
            import time

            time.sleep(0.1)

            # Verify the file exists and has the correct size
            if not os.path.exists(filepath):
                raise OSError(f"File {filepath} was not created")

            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise OSError(f"File {filepath} was created but is empty")

            # Generate BIDS-compliant channels.tsv file if requested
            if create_channels_tsv:
                channels_tsv_path = os.path.splitext(filepath)[0] + "_channels.tsv"
                # Create DataFrame with columns in BIDS-specified order
                # Required columns first: name, type, units
                # Then optional columns in the order they appear in data
                ordered_columns = ["name", "type", "units"]
                optional_columns = [
                    col for col in channels_tsv_data.keys() if col not in ordered_columns
                ]
                column_order = ordered_columns + optional_columns

                channels_df = pd.DataFrame(channels_tsv_data)
                channels_df = channels_df[column_order]
                channels_df.to_csv(channels_tsv_path, sep="\t", index=False, na_rep="n/a")
                print(f"\nBIDS-compliant channels metadata saved to: {channels_tsv_path}")

            # Print summary using stored analyses, only if analysis was performed
            if not bypass_analysis:
                # We need to adapt summarize_channels call slightly or assume it uses the analyses dict
                # Let's refine the analyses dict passed to summarize_channels
                summary_analyses = {}
                for ch_name, analysis in signal_analyses.items():
                    summary_analyses[ch_name] = {
                        "range": analysis["range"],
                        "dynamic_range_db": analysis["dynamic_range_db"],
                        "snr_db": analysis["snr"],
                        "use_bdf": use_bdf,  # Use the final decision for the whole file
                    }

                summary = summarize_channels(emg.channels, emg.signals, summary_analyses)
                print("\nSummary:")
                print(summary)
            else:
                print("\nSummary skipped as signal analysis was bypassed.")

            print(f"\nEMG data exported to: {filepath}")
            return filepath
        except Exception as e:
            # Clean up if there was an error
            if "writer" in locals() and hasattr(writer, "close") and callable(writer.close):
                try:
                    # Check if file is open before closing
                    if not writer.header["file_handle"].closed:
                        writer.close()
                except Exception:
                    pass  # Ignore errors during cleanup

            # Wait a moment before trying to delete the file
            import time

            time.sleep(0.1)

            if "filepath" in locals() and os.path.exists(filepath):
                try:
                    os.unlink(filepath)
                    print(f"Cleaned up partially written file: {filepath}")
                except Exception as unlink_e:
                    print(f"Error during cleanup of {filepath}: {unlink_e}")

            raise e
        finally:
            if writer is not None:
                writer.close()  # Ensure writer is closed
