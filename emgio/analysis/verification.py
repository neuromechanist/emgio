"""
Functions for verifying signal integrity after operations like export/import.
"""

import numpy as np
from typing import Optional, Dict, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from ..core.emg import EMG


def compare_signals(emg_original: 'EMG', emg_reloaded: 'EMG',
                    tolerance: float = 0.001,  # Default tolerance 0.1% for NRMSE and Max Norm Abs Diff
                    channel_map: Optional[Dict[str, str]] = None) -> dict:
    """
    Compare signals between two EMG objects using normalized metrics.

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
              Metrics include 'nrmse' (Normalized RMSE), 'max_norm_abs_diff',
              'snr_diff_db'.
              Also includes 'channel_summary' with comparison mode and unmatched channels.
    """
    # Removed local import: from emgio.core.emg import EMG

    results = {}
    original_channels = set(emg_original.signals.columns)
    reloaded_channels = set(emg_reloaded.signals.columns)

    # Initialize channel summary
    channel_summary = {
        'comparison_mode': 'unknown',
        'unmatched_original': [],
        'unmatched_reloaded': []
    }

    # Handle channel mapping
    if channel_map is not None:
        # Use provided channel map
        channel_summary['comparison_mode'] = 'mapped'
        # Validate all original channels in map exist
        missing_original = [ch for ch in channel_map.keys() if ch not in original_channels]
        if missing_original:
            raise ValueError(f"Channel map contains original channels not found in data: {missing_original}")

        # Get mapped channels that exist in reloaded data
        valid_mappings = {orig: mapped for orig, mapped in channel_map.items()
                          if mapped in reloaded_channels}

        # Track unmatched channels
        channel_summary['unmatched_original'] = [ch for ch in original_channels
                                                 if ch not in channel_map]
        channel_summary['unmatched_reloaded'] = [ch for ch in reloaded_channels
                                                 if ch not in channel_map.values()]

        # Use only valid mappings for comparison
        channel_pairs = [(orig, mapped) for orig, mapped in valid_mappings.items()]
    else:
        # Try exact name matching first
        common_channels = list(original_channels.intersection(reloaded_channels))
        if common_channels:
            channel_summary['comparison_mode'] = 'exact_name'
            channel_pairs = [(ch, ch) for ch in common_channels]
            channel_summary['unmatched_original'] = list(original_channels - reloaded_channels)
            channel_summary['unmatched_reloaded'] = list(reloaded_channels - original_channels)
        else:
            # Fall back to order-based matching
            channel_summary['comparison_mode'] = 'order_based'
            min_len = min(len(original_channels), len(reloaded_channels))
            original_list = sorted(list(original_channels))
            reloaded_list = sorted(list(reloaded_channels))
            channel_pairs = list(zip(original_list[:min_len], reloaded_list[:min_len]))
            channel_summary['unmatched_original'] = original_list[min_len:]
            channel_summary['unmatched_reloaded'] = reloaded_list[min_len:]

    results['channel_summary'] = channel_summary

    if not channel_pairs:
        print("Warning: No channel pairs found for comparison.")
        return results

    # Compare each channel pair
    for orig_channel, reloaded_channel in channel_pairs:
        sig_orig = emg_original.signals[orig_channel].values
        sig_reloaded = emg_reloaded.signals[reloaded_channel].values

        # Basic check for length mismatch
        if len(sig_orig) != len(sig_reloaded):
            min_len = min(len(sig_orig), len(sig_reloaded))
            print(f"Warning: Signal lengths differ for {orig_channel} -> {reloaded_channel} "
                  f"({len(sig_orig)} vs {len(sig_reloaded)}). Comparing first {min_len} samples.")
            sig_orig = sig_orig[:min_len]
            sig_reloaded = sig_reloaded[:min_len]

        # Calculate normalization factor (peak-to-peak range of original signal)
        sig_orig_range = np.ptp(sig_orig)
        # print(f"Original signal range: {sig_orig_range}") # Optional: uncomment for debugging
        # sig_reloaded_range = np.ptp(sig_reloaded)
        # print(f"Reloaded signal range: {sig_reloaded_range}") # Optional: uncomment for debugging
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

        # SNR of the difference (remains absolute measure in dB)
        signal_power = np.mean(sig_orig**2)
        noise_power = np.mean(diff**2)
        # Avoid division by zero or log(0)
        if signal_power < np.finfo(float).eps or noise_power < np.finfo(float).eps:
            snr_diff_db = np.inf if signal_power > np.finfo(float).eps else -np.inf
        else:
            # Add epsilon
            snr_diff_db = 10 * np.log10(signal_power / (noise_power + np.finfo(float).eps))

        # Check if nrmse or max_norm_abs_diff are below tolerance
        is_identical = nrmse < tolerance and max_norm_abs_diff < tolerance

        results[orig_channel] = {
            'reloaded_channel': reloaded_channel,
            'original_range': sig_orig_range,  # Store original range for context
            # 'reloaded_range': sig_reloaded_range, # Reloaded range not strictly needed
            'nrmse': nrmse,
            'max_norm_abs_diff': max_norm_abs_diff,
            'snr_diff_db': snr_diff_db,
            'is_identical': is_identical
        }

    return results 