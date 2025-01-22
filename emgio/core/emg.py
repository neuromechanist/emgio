import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Union


class EMG:
    """
    Core EMG class for handling EMG data and metadata.

    Attributes:
        signals (pd.DataFrame): Raw signal data with time as index
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
            'trigno': 'TrignoImporter'
            # Add more importers here as they are implemented
            # 'noraxon': 'NoraxonImporter',
            # 'otb': 'OTBImporter'
        }

        if importer not in importers:
            raise ValueError(f"Unsupported importer: {importer}. "
                           f"Available importers: {list(importers.keys())}")

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

    def select_channels(self, channels: Union[str, List[str]]) -> 'EMG':
        """
        Select specific channels from the data.

        Args:
            channels: Channel name or list of channel names to select

        Returns:
            EMG: Self for method chaining
        """
        if self.signals is None:
            raise ValueError("No signals loaded")

        if isinstance(channels, str):
            channels = [channels]

        if not all(ch in self.signals.columns for ch in channels):
            missing = [ch for ch in channels if ch not in self.signals.columns]
            raise ValueError(f"Channels not found: {missing}")

        self.signals = self.signals[channels]
        self.channels = {ch: self.channels[ch] for ch in channels}
        return self

    def plot_signals(self, channels: Optional[List[str]] = None,
                    time_range: Optional[tuple] = None) -> None:
        """
        Plot EMG signals.

        Args:
            channels: List of channels to plot. If None, plot all channels
            time_range: Tuple of (start_time, end_time) to plot. If None, plot all data
        """
        if self.signals is None:
            raise ValueError("No signals loaded")

        if channels is None:
            channels = self.signals.columns
        elif not all(ch in self.signals.columns for ch in channels):
            missing = [ch for ch in channels if ch not in self.signals.columns]
            raise ValueError(f"Channels not found: {missing}")

        fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2*len(channels)))
        if len(channels) == 1:
            axes = [axes]

        for ax, channel in zip(axes, channels):
            data = self.signals[channel]
            if time_range:
                start, end = time_range
                data = data.loc[start:end]

            ax.plot(data.index, data.values)
            ax.set_title(channel)
            ax.set_ylabel(f"{self.channels[channel]['unit']}")

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    def to_edf(self, filepath: str) -> None:
        """
        Export data to EDF format with corresponding channels.tsv file.

        Args:
            filepath: Path to save the EDF file
        """
        from ..exporters.edf import EDFExporter
        EDFExporter.export(self, filepath)

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
                   unit: str, ch_type: str = 'EMG') -> None:
        """
        Add a new channel to the EMG data.

        Args:
            name: Channel name
            data: Channel data
            sampling_freq: Sampling frequency in Hz
            unit: Unit of measurement
            ch_type: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        if self.signals is None:
            # Create DataFrame with time index
            time = np.arange(len(data))/sampling_freq
            self.signals = pd.DataFrame(index=time)

        self.signals[name] = data
        self.channels[name] = {
            'sampling_freq': sampling_freq,
            'unit': unit,
            'type': ch_type
        }
