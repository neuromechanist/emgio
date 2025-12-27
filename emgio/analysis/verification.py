"""
Functions for verifying signal integrity after operations like export/import.
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from ..core.emg import EMG


def compare_signals(
    emg_original: "EMG",
    emg_reloaded: "EMG",
    tolerance: float = 0.01,  # Default tolerance 1% for NRMSE and Max Norm Abs Diff
    channel_map: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Compare signals between two EMG objects using normalized metrics.
    Returns a dictionary with comparison results per channel and a summary.
    Does NOT perform logging/printing.

    Args:
        emg_original: The original EMG object before export.
        emg_reloaded: The EMG object reloaded from the exported file.
        tolerance: Relative tolerance for comparisons (default: 0.001 or 0.1%).
                   Used for NRMSE, Max Norm Abs Diff, and identity check.
        channel_map: Optional dictionary mapping original channel names (keys)
                    to reloaded channel names (values). If None, tries exact name
                    match first, then falls back to order-based matching.

    Returns:
        dict: A dictionary containing normalized comparison metrics for each common channel.
              Metrics include 'nrmse' (Normalized RMSE), 'max_norm_abs_diff'.
              Also includes 'channel_summary' with comparison mode and unmatched channels.
    """
    # Removed local import: from emgio.core.emg import EMG

    results = {}
    original_channels = set(emg_original.signals.columns)
    reloaded_channels = set(emg_reloaded.signals.columns)

    # Initialize channel summary
    channel_summary = {
        "comparison_mode": "unknown",
        "unmatched_original": [],
        "unmatched_reloaded": [],
    }

    # Handle channel mapping
    if channel_map is not None:
        # Use provided channel map
        channel_summary["comparison_mode"] = "mapped"
        # Validate all original channels in map exist
        missing_original = [ch for ch in channel_map.keys() if ch not in original_channels]
        if missing_original:
            raise ValueError(
                f"Channel map contains original channels not found in data: {missing_original}"
            )

        # Get mapped channels that exist in reloaded data
        valid_mappings = {
            orig: mapped for orig, mapped in channel_map.items() if mapped in reloaded_channels
        }

        # Track unmatched channels
        channel_summary["unmatched_original"] = [
            ch for ch in original_channels if ch not in channel_map
        ]
        channel_summary["unmatched_reloaded"] = [
            ch for ch in reloaded_channels if ch not in channel_map.values()
        ]

        # Use only valid mappings for comparison
        channel_pairs = list(valid_mappings.items())
    else:
        # Try exact name matching first
        common_channels = list(original_channels.intersection(reloaded_channels))
        if common_channels:
            channel_summary["comparison_mode"] = "exact_name"
            channel_pairs = [(ch, ch) for ch in common_channels]
            channel_summary["unmatched_original"] = list(original_channels - reloaded_channels)
            channel_summary["unmatched_reloaded"] = list(reloaded_channels - original_channels)
        else:
            # Fall back to order-based matching
            channel_summary["comparison_mode"] = "order_based"
            min_len = min(len(original_channels), len(reloaded_channels))
            original_list = sorted(original_channels)
            reloaded_list = sorted(reloaded_channels)
            channel_pairs = list(zip(original_list[:min_len], reloaded_list[:min_len]))
            channel_summary["unmatched_original"] = original_list[min_len:]
            channel_summary["unmatched_reloaded"] = reloaded_list[min_len:]

    results["channel_summary"] = channel_summary

    if not channel_pairs:
        return results

    # Compare each channel pair
    for orig_channel, reloaded_channel in channel_pairs:
        sig_orig = emg_original.signals[orig_channel].values
        sig_reloaded = emg_reloaded.signals[reloaded_channel].values

        # Basic check for length mismatch
        if len(sig_orig) != len(sig_reloaded):
            min_len = min(len(sig_orig), len(sig_reloaded))
            sig_orig = sig_orig[:min_len]
            sig_reloaded = sig_reloaded[:min_len]

        # Calculate normalization factor (peak-to-peak range of original signal)
        sig_orig_range = np.ptp(sig_orig)
        # Use a small epsilon to avoid division by zero for constant signals
        norm_factor = sig_orig_range if sig_orig_range > np.finfo(float).eps else 1.0

        # Calculate metrics
        diff = sig_orig - sig_reloaded
        rmse = np.sqrt(np.mean(diff**2))
        max_abs_diff = np.max(np.abs(diff))

        # Normalize metrics
        # Add epsilon to norm_factor in denominator to prevent division by zero
        nrmse = rmse / (norm_factor + np.finfo(float).eps)
        max_norm_abs_diff = max_abs_diff / (norm_factor + np.finfo(float).eps)

        # Check if nrmse or max_norm_abs_diff are below tolerance
        is_identical = nrmse < tolerance and max_norm_abs_diff < tolerance

        results[orig_channel] = {
            "reloaded_channel": reloaded_channel,
            "original_range": sig_orig_range,  # Store original range for context
            "nrmse": nrmse,
            "max_norm_abs_diff": max_norm_abs_diff,
            "is_identical": is_identical,
        }

    return results


def report_verification_results(verification_results: dict, verify_tolerance: float) -> bool:
    """
    Logs a detailed report based on the results from compare_signals.

    Args:
        verification_results: The dictionary output from compare_signals.
        verify_tolerance: The tolerance used during comparison (for reporting).

    Returns:
        bool: True if all compared channels were identical within tolerance, False otherwise.
    """
    summary = verification_results.get("channel_summary", {})
    logging.info("--- Verification Report ---")
    logging.info(f"Comparison mode: {summary.get('comparison_mode', 'unknown')}")

    if summary.get("unmatched_original"):
        logging.warning(f"Unmatched original channels: {summary['unmatched_original']}")
    if summary.get("unmatched_reloaded"):
        logging.warning(f"Unmatched reloaded channels: {summary['unmatched_reloaded']}")

    all_identical = True
    compared_count = 0
    for orig_channel, metrics in verification_results.items():
        if orig_channel == "channel_summary":
            continue
        compared_count += 1
        reloaded_channel = metrics["reloaded_channel"]
        channel_label = (
            f"'{orig_channel}' -> '{reloaded_channel}'"
            if orig_channel != reloaded_channel
            else f"'{orig_channel}'"
        )

        if not metrics["is_identical"]:
            all_identical = False
            log_msg = (
                f"Channel {channel_label}: Signals differ "
                f"(nRMSE: {metrics['nrmse']:.2e}, "
                f"MaxNormDiff: {metrics['max_norm_abs_diff']:.2e})"
            )
            logging.critical(log_msg)
        else:
            logging.info(
                f"Channel {channel_label}: Signals are identical (within tolerance {verify_tolerance:.1e})."
            )

    if compared_count == 0:
        logging.critical("No channels were actually compared.")
        all_identical = False  # Mark as not successful if nothing compared

    if all_identical:
        log_msg = (
            f"Verification successful: All {compared_count} compared "
            f"channel pairs are identical within tolerance."
        )
        logging.critical(log_msg)
    elif summary.get("comparison_mode") != "failed":
        log_msg = f"Verification finished: Differences found in {compared_count} compared pairs."
        logging.critical(log_msg)
    else:  # Comparison mode failed (e.g., no pairs found)
        logging.error("Verification failed: Could not compare channels.")

    logging.info("---------------------------")
    return all_identical
