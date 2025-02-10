import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Any


class EMG:
    """
    Core EMG class for handling EMG data and metadata.

    Attributes:
        signals (pd.DataFrame): Raw signal data with time as index, for now, all channels are stored
        in the same DataFrame, so they require the same sampling frequency
        metadata (dict): Metadata dictionary containing recording information
        channels (dict): Channel information including type, unit, sampling frequency
    """

    def __init__(self):
        """Initialize an empty EMG object."""
        self.signals = None
        self.metadata = {}
        self.channels = {}

    @classmethod
    def from_file(cls, filepath: str, importer: str = 'trigno') -> 'EMG':
        """
        Factory method to create EMG object from file.

        Args:
            filepath: Path to the input file
            importer: Name of the importer to use ('trigno', 'noraxon', 'otb')

        Returns:
            EMG: New EMG object with loaded data
        """
        importers = {
            'trigno': 'TrignoImporter',
            'otb': 'OTBImporter',  # OTB/OTB+ EMG system data
            'edf': 'EDFImporter'  # EDF/EDF+/BDF format
        }

        if importer not in importers:
            raise ValueError(
                f"Unsupported importer: {importer}. "
                f"Available importers: {list(importers.keys())}\n"
                "- trigno: Delsys Trigno EMG system\n"
                "- otb: OTB/OTB+ EMG system\n"
                "- edf: EDF/EDF+/BDF format"
            )

        # Import the appropriate importer class
        importer_module = __import__(
            f'emgio.importers.{importer}',
            globals(),
            locals(),
            [importers[importer]]
        )
        importer_class = getattr(importer_module, importers[importer])

        # Create importer instance and load data
        return importer_class().load(filepath)

    def select_channels(self, channels: Union[str, List[str], None] = None,
                        channel_type: Optional[str] = None) -> 'EMG':
        """
        Select specific channels from the data.

        Args:
            channels: Channel name or list of channel names to select. If None and
                     channel_type is specified, selects all channels of that type.
            channel_type: Type of channels to select ('EMG', 'ACC', 'GYRO', etc.).
                         If specified with channels, filters the selection to only
                         channels of this type.

        Returns:
            EMG: Self for method chaining

        Examples:
            # Select specific channels
            emg.select_channels(['EMG1', 'EMG2'])

            # Select all EMG channels
            emg.select_channels(channel_type='EMG')

            # Select specific EMG channels only
            emg.select_channels(['EMG1', 'ACC1'], channel_type='EMG')
        """
        if self.signals is None:
            raise ValueError("No signals loaded")

        # If channel_type specified but no channels, select all of that type
        if channels is None and channel_type is not None:
            channels = [ch for ch, info in self.channels.items()
                        if info['type'] == channel_type]
            if not channels:
                raise ValueError(f"No channels found of type: {channel_type}")
        elif isinstance(channels, str):
            channels = [channels]

        # Validate channels exist
        if not all(ch in self.signals.columns for ch in channels):
            missing = [ch for ch in channels if ch not in self.signals.columns]
            raise ValueError(f"Channels not found: {missing}")

        # Filter by type if specified
        if channel_type is not None:
            channels = [ch for ch in channels
                        if self.channels[ch]['type'] == channel_type]
            if not channels:
                raise ValueError(
                    f"None of the selected channels are of type: {channel_type}")

        self.signals = self.signals[channels]
        self.channels = {ch: self.channels[ch] for ch in channels}
        return self

    def get_channel_types(self) -> List[str]:
        """
        Get list of unique channel types in the data.

        Returns:
            List of channel types (e.g., ['EMG', 'ACC', 'GYRO'])
        """
        return list(set(info['type'] for info in self.channels.values()))

    def get_channels_by_type(self, channel_type: str) -> List[str]:
        """
        Get list of channels of a specific type.

        Args:
            channel_type: Type of channels to get ('EMG', 'ACC', 'GYRO', etc.)

        Returns:
            List of channel names of the specified type
        """
        return [ch for ch, info in self.channels.items()
                if info['type'] == channel_type]

    def plot_signals(self, channels: Optional[List[str]] = None,
                     time_range: Optional[tuple] = None,
                     style: str = 'line',
                     grid: bool = True,
                     title: Optional[str] = None,
                     show: bool = True,
                     plt_module: Any = plt) -> None:
        """
        Plot EMG signals with enhanced visualization options.

        Args:
            channels: List of channels to plot. If None, plot all channels
            time_range: Tuple of (start_time, end_time) to plot. If None, plot all data
            style: Plot style ('line' or 'dots')
            grid: Whether to show grid lines
            title: Optional title for the entire figure
        """
        if self.signals is None:
            raise ValueError("No signals loaded")

        if channels is None:
            channels = self.signals.columns
        elif not all(ch in self.signals.columns for ch in channels):
            missing = [ch for ch in channels if ch not in self.signals.columns]
            raise ValueError(f"Channels not found: {missing}")

        # Create figure with shared x-axis
        fig, axes = plt_module.subplots(
            len(channels), 1,
            figsize=(12, 3 * len(channels)),
            sharex=True
        )
        if len(channels) == 1:
            axes = [axes]

        # Set figure title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=1.02)

        for ax, channel in zip(axes, channels):
            data = self.signals[channel]
            if time_range:
                start, end = time_range
                data = data.loc[start:end]

            # Plot based on style
            if style == 'dots':
                ax.scatter(data.index, data.values, s=1, alpha=0.6)
            else:  # default to line
                ax.plot(data.index, data.values, linewidth=1)

            # Channel info in title
            ch_info = self.channels[channel]
            ax.set_title(f"{channel} ({ch_info['type']} - {ch_info['sampling_freq']} Hz)")
            ax.set_ylabel(f"{ch_info['unit']}")

            if grid:
                ax.grid(True, linestyle='--', alpha=0.7)

        # Common x-axis label
        axes[-1].set_xlabel("Time (s)")

        # Adjust layout to prevent label overlap
        plt_module.tight_layout()
        if show:
            plt_module.show()

    def to_edf(self, filepath: str, **kwargs) -> None:
        """
        Export data to EDF format with corresponding channels.tsv file.

        Args:
            filepath: Path to save the EDF file
            **kwargs: Additional arguments for the EDF exporter

        Raises:
            ValueError: If no signals are loaded
        """
        if self.signals is None:
            raise ValueError("No signals loaded")
            
        from ..exporters.edf import EDFExporter
        EDFExporter.export(self, filepath, **kwargs)

    def set_metadata(self, key: str, value: any) -> None:
        """
        Set metadata value.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str) -> any:
        """
        Get metadata value.

        Args:
            key: Metadata key

        Returns:
            Value associated with the key
        """
        return self.metadata.get(key)

    def add_channel(self, name: str, data: np.ndarray, sampling_freq: float,
                    unit: str, prefilter: str = 'n/a', ch_type: str = 'EMG') -> None:
        """
        Add a new channel to the EMG data.

        Args:
            name: Channel name
            data: Channel data
            sampling_freq: Sampling frequency in Hz
            unit: Unit of measurement
            prefilter: Pre-filtering applied to the channel
            ch_type: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        if self.signals is None:
            # Create DataFrame with time index
            time = np.arange(len(data)) / sampling_freq
            self.signals = pd.DataFrame(index=time)

        self.signals[name] = data
        self.channels[name] = {
            'sampling_freq': sampling_freq,
            'unit': unit,
            'prefilter': prefilter,
            'type': ch_type
        }
