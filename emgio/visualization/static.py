"""
Static plotting functions for EMG data.
"""

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from ..core.emg import EMG  # Assuming EMG class is in core.emg


def plot_signals(
    emg_object: "EMG",  # Changed first arg to accept EMG object
    channels: list[str] | None = None,
    time_range: tuple[float, float] | None = None,
    offset_scale: float = 0.8,
    uniform_scale: bool = True,
    detrend: bool = False,
    grid: bool = True,
    title: str | None = None,
    show: bool = True,
    plt_module: Any = plt,
) -> None:
    """
    Plot EMG signals in a single plot with vertical offsets.

    Args:
        emg_object: The EMG object containing the signals and metadata.
        channels: List of channels to plot. If None, plot all channels.
        time_range: Tuple of (start_time, end_time) to plot. If None, plot all data.
        offset_scale: Portion of allocated space each signal can use (0.0 to 1.0).
        uniform_scale: Whether to use the same scale for all signals.
        detrend: Whether to remove mean from signals before plotting.
        grid: Whether to show grid lines.
        title: Optional title for the figure.
        show: Whether to display the plot.
        plt_module: Matplotlib pyplot module to use.
    """
    if emg_object.signals is None:
        raise ValueError("EMG object has no signals loaded.")

    signals_df = emg_object.signals

    if channels is None:
        channels = list(signals_df.columns)
    elif not all(ch in signals_df.columns for ch in channels):
        missing = [ch for ch in channels if ch not in signals_df.columns]
        raise ValueError(f"Channels not found: {missing}")

    # Create figure
    fig, ax = plt_module.subplots(figsize=(12, 8))

    # Set figure title if provided
    if title:
        ax.set_title(title, fontsize=14, pad=20)

    # Process signals
    processed_data = {}
    max_range = 0
    min_val = np.inf
    max_val = -np.inf

    for _i, channel in enumerate(channels):
        data = signals_df[channel]
        if time_range:
            start, end = time_range
            # Use searchsorted for robustness if index isn't perfectly aligned
            start_idx = data.index.searchsorted(start)
            end_idx = data.index.searchsorted(end, side="right")
            data = data.iloc[start_idx:end_idx]

        # Detrend if requested
        if detrend:
            data = data - data.mean()

        processed_data[channel] = data
        channel_range = data.max() - data.min()
        if channel_range > 0:
            max_range = max(max_range, channel_range)
        min_val = min(min_val, data.min())
        max_val = max(max_val, data.max())

    # Handle case where all signals are flat
    if max_range == 0:
        max_range = 1.0  # Avoid division by zero

    # Plot each signal with offset
    n_channels = len(channels)
    yticks = []
    yticklabels = []

    for i, channel in enumerate(channels):
        data = processed_data[channel]
        channel_range = data.max() - data.min()

        # Calculate offset and scaling
        offset = (n_channels - 1 - i) * max_range  # Offset based on max_range

        # Scale only if uniform_scale is False and channel_range > 0
        if not uniform_scale and channel_range > 0:
            scaled_data = (data - data.min()) / channel_range * (max_range * offset_scale) + offset
        else:
            # If uniform scale, just apply offset (no scaling needed as offset is based on max_range)
            # Or if channel_range is 0 (flat line)
            scaled_data = data + offset

        # Plot the signal
        ax.plot(data.index, scaled_data, linewidth=1, label=channel)

        # Store tick position and label (use mean value + offset for tick position)
        yticks.append(data.mean() + offset)
        yticklabels.append(f"{channel}")

    # Set axis labels and ticks
    ax.set_xlabel("Time (s)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Set y-axis limits with some padding based on the overall scaled range
    # Calculate overall min/max of scaled data for limits
    all_scaled_min = np.inf
    all_scaled_max = -np.inf
    for i, channel in enumerate(channels):
        data = processed_data[channel]
        offset = (n_channels - 1 - i) * max_range  # Offset based on max_range
        channel_range = data.max() - data.min()
        if not uniform_scale and channel_range > 0:
            scaled_data = (data - data.min()) / channel_range * (max_range * offset_scale) + offset
        else:
            scaled_data = data + offset
        all_scaled_min = min(all_scaled_min, scaled_data.min())
        all_scaled_max = max(all_scaled_max, scaled_data.max())

    padding = max_range * 0.1  # 10% padding based on max range
    ax.set_ylim(all_scaled_min - padding, all_scaled_max + padding)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt_module.tight_layout()
    if show:
        plt_module.show()


def plot_comparison(
    emg_original: "EMG",
    emg_reloaded: "EMG",
    channels: list[str] | None = None,
    time_range: tuple[float, float] | None = None,
    detrend: bool = False,
    grid: bool = True,
    suptitle: str | None = "Signal Comparison",
    show: bool = True,
    channel_map: dict[str, str] | None = None,
    plt_module: Any = plt,
) -> None:
    """
    Plot original and reloaded signals overlaid for visual comparison.

    Creates subplots for each channel pair.

    Args:
        emg_original: The original EMG object.
        emg_reloaded: The reloaded EMG object.
        channels: List of original channels to plot. If None, plot common/mapped channels.
        time_range: Tuple of (start_time, end_time) to plot. If None, plot all data.
        detrend: Whether to remove mean from signals before plotting.
        grid: Whether to show grid lines on subplots.
        suptitle: Optional main title for the figure.
        show: Whether to display the plot.
        channel_map: Optional dictionary mapping original channel names (keys)
                    to reloaded channel names (values). If None, tries exact name
                    match first, then falls back to order-based matching.
        plt_module: Matplotlib pyplot module to use.
    """
    # Removed local import: from emgio.core.emg import EMG

    original_channel_names = set(emg_original.signals.columns)
    reloaded_channel_names = set(emg_reloaded.signals.columns)

    # --- Channel Matching Logic (adapted from compare_signals) ---
    comparison_mode = "unknown"
    unmatched_original = []
    unmatched_reloaded = []
    channel_pairs = []

    if channel_map is not None:
        comparison_mode = "mapped"
        valid_mappings = {
            orig: mapped
            for orig, mapped in channel_map.items()
            if orig in original_channel_names and mapped in reloaded_channel_names
        }

        # Filter by user-specified channels if provided
        if channels is not None:
            valid_mappings = {
                orig: mapped for orig, mapped in valid_mappings.items() if orig in channels
            }

        channel_pairs = list(valid_mappings.items())

        specified_orig_in_map = set(channel_map.keys())
        specified_reloaded_in_map = set(channel_map.values())

        unmatched_original = list(original_channel_names - specified_orig_in_map)
        unmatched_reloaded = list(reloaded_channel_names - specified_reloaded_in_map)

        if channels is not None:  # Further filter unmatched if specific channels were requested
            unmatched_original = [ch for ch in unmatched_original if ch in channels]

    else:
        # Try exact name matching first
        common_channels = list(original_channel_names.intersection(reloaded_channel_names))
        if channels is not None:  # Filter by user-specified channels
            common_channels = [ch for ch in common_channels if ch in channels]

        if common_channels:
            comparison_mode = "exact_name"
            channel_pairs = [(ch, ch) for ch in common_channels]

            specified_channels_set = set(channels) if channels else original_channel_names
            common_channels_set = set(common_channels)

            unmatched_original = list(specified_channels_set - common_channels_set)
            # Reloaded unmatched are those not in common from the full reloaded set
            unmatched_reloaded = list(reloaded_channel_names - common_channels_set)
        else:
            # Fall back to order-based matching if no exact matches or map provided
            comparison_mode = "order_based"
            original_list = sorted(original_channel_names)
            reloaded_list = sorted(reloaded_channel_names)

            if channels is not None:  # Filter original list by user-specified channels
                original_list = [ch for ch in original_list if ch in channels]

            min_len = min(len(original_list), len(reloaded_list))
            channel_pairs = list(
                zip(original_list[:min_len], reloaded_list[:min_len], strict=False)
            )
            unmatched_original = original_list[min_len:]
            unmatched_reloaded = reloaded_list[min_len:]

    if not channel_pairs:
        print("Warning: No channel pairs found for plotting based on selection criteria.")
        if unmatched_original:
            print(f"  Unmatched original channels: {unmatched_original}")
        if unmatched_reloaded:
            print(f"  Unmatched reloaded channels: {unmatched_reloaded}")
        return

    print(f"Plotting comparison using mode: {comparison_mode}")
    if unmatched_original:
        print(f"  Original channels not plotted: {unmatched_original}")
    if unmatched_reloaded:
        print(f"  Reloaded channels not plotted: {unmatched_reloaded}")

    # --- Plotting ---
    n_pairs = len(channel_pairs)
    fig, axes = plt_module.subplots(
        n_pairs, 1, figsize=(15, 2.5 * n_pairs), sharex=True, squeeze=False
    )
    axes = axes.flatten()  # Ensure axes is always a 1D array

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    for i, (orig_ch, reloaded_ch) in enumerate(channel_pairs):
        ax = axes[i]
        sig_orig = emg_original.signals[orig_ch]
        sig_reloaded = emg_reloaded.signals[reloaded_ch]

        # Apply time range
        if time_range:
            start, end = time_range
            # Be robust to time range slightly outside index
            # Break long line 608
            sig_orig_idx_start = sig_orig.index.searchsorted(start)
            sig_orig_idx_end = sig_orig.index.searchsorted(end, side="right")
            sig_orig = sig_orig.loc[sig_orig_idx_start:sig_orig_idx_end]

            sig_reloaded_idx_start = sig_reloaded.index.searchsorted(start)
            sig_reloaded_idx_end = sig_reloaded.index.searchsorted(end, side="right")
            sig_reloaded = sig_reloaded.loc[sig_reloaded_idx_start:sig_reloaded_idx_end]

        time_vector = sig_orig.index  # Assuming time vectors are aligned

        # Detrend if requested
        if detrend:
            sig_orig = sig_orig - sig_orig.mean()
            sig_reloaded = sig_reloaded - sig_reloaded.mean()

        # Plot signals
        ax.plot(time_vector, sig_orig, label="Original", color="blue", linewidth=1.0)
        ax.plot(
            time_vector, sig_reloaded, label="Reloaded", color="red", linestyle="--", linewidth=1.0
        )

        title_str = f"{orig_ch} -> {reloaded_ch}" if orig_ch != reloaded_ch else orig_ch
        ax.set_ylabel(title_str, rotation=0, labelpad=30, ha="right", va="center")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))  # Use scientific notation

        if grid:
            ax.grid(True, linestyle=":", alpha=0.6)

        # Add legend to the first subplot only for clarity
        if i == 0:
            ax.legend(loc="upper right")

    # Set common x-label
    axes[-1].set_xlabel("Time (s)")

    # Adjust layout
    plt_module.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust rect for suptitle

    if show:
        plt_module.show()
