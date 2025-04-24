import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Optional, Union, Any, Literal
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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
    def _infer_importer(cls, filepath: str) -> str:
        """
        Infer the importer to use based on the file extension.
        """
        extension = os.path.splitext(filepath)[1].lower()
        if extension in {'.edf', '.edf+', '.bdf'}:
            return 'edf'
        elif extension in {'.set'}:
            return 'eeglab'
        elif extension in {'.otb', '.otb+'}:
            return 'otb'
        elif extension in {'.csv', '.txt'}:
            return 'csv'
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    @classmethod
    def from_file(
            cls,
            filepath: str,
            importer: Literal['trigno', 'otb', 'eeglab', 'edf', 'csv'] | None = None,
            force_csv: bool = False,
            **kwargs
    ) -> 'EMG':
        """
        The method to create EMG object from file.

        Args:
            filepath: Path to the input file
            importer: Name of the importer to use. Can be one of the following:
                - 'trigno': Delsys Trigno EMG system (CSV)
                - 'otb': OTB/OTB+ EMG system (OTB, OTB+)
                - 'eeglab': EEGLAB .set files (SET)
                - 'edf': EDF/EDF+/BDF format (EDF, EDF+, BDF)
                - 'csv': Generic CSV (or TXT) files with columnar data
                If None, the importer will be inferred from the file extension.
                Automatic import is supported for CSV/TXT files.
            force_csv: If True and importer is 'csv', forces using the generic CSV
                      importer even if the file appears to match a specialized format.
            **kwargs: Additional arguments passed to the importer

        Returns:
            EMG: New EMG object with loaded data
        """
        if importer is None:
            importer = cls._infer_importer(filepath)

        importers = {
            'trigno': 'TrignoImporter',  # CSV with Delsys Trigno Headers
            'otb': 'OTBImporter',  # OTB/OTB+ EMG system data
            'edf': 'EDFImporter',  # EDF/EDF+/BDF format
            'eeglab': 'EEGLABImporter',  # EEGLAB .set files
            'csv': 'CSVImporter'  # Generic CSV/Text files
        }

        if importer not in importers:
            raise ValueError(
                f"Unsupported importer: {importer}. "
                f"Available importers: {list(importers.keys())}\n"
                "- trigno: Delsys Trigno EMG system\n"
                "- otb: OTB/OTB+ EMG system\n"
                "- edf: EDF/EDF+/BDF format\n"
                "- eeglab: EEGLAB .set files\n"
                "- csv: Generic CSV/Text files"
            )

        # If using CSV importer and force_csv is set, pass it as force_generic
        if importer == 'csv':
            kwargs['force_generic'] = force_csv

        # Import the appropriate importer class
        importer_module = __import__(
            f'emgio.importers.{importer}',
            globals(),
            locals(),
            [importers[importer]]
        )
        importer_class = getattr(importer_module, importers[importer])

        # Create importer instance and load data
        return importer_class().load(filepath, **kwargs)

    def select_channels(
            self,
            channels: Union[str, List[str], None] = None,
            channel_type: Optional[str] = None,
            inplace: bool = False) -> 'EMG':
        """
        Select specific channels from the data and return a new EMG object.

        Args:
            channels: Channel name or list of channel names to select. If None and
                    channel_type is specified, selects all channels of that type.
            channel_type: Type of channels to select ('EMG', 'ACC', 'GYRO', etc.).
                        If specified with channels, filters the selection to only
                        channels of this type.

        Returns:
            EMG: A new EMG object containing only the selected channels

        Examples:
            # Select specific channels
            new_emg = emg.select_channels(['EMG1', 'ACC1'])

            # Select all EMG channels
            emg_only = emg.select_channels(channel_type='EMG')

            # Select specific EMG channels only, this example does not select ACC channels
            emg_subset = emg.select_channels(['EMG1', 'ACC1'], channel_type='EMG')
        """
        if self.signals is None:
            raise ValueError("No signals loaded")

        # If channel_type specified but no channels, select all of that type
        if channels is None and channel_type is not None:
            channels = [ch for ch, info in self.channels.items()
                        if info['channel_type'] == channel_type]
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
                        if self.channels[ch]['channel_type'] == channel_type]
            if not channels:
                raise ValueError(
                    f"None of the selected channels are of type: {channel_type}")

        # Create new EMG object
        new_emg = EMG()

        # Copy selected signals and channels
        new_emg.signals = self.signals[channels].copy()
        new_emg.channels = {ch: self.channels[ch].copy() for ch in channels}

        # Copy metadata
        new_emg.metadata = self.metadata.copy()

        if not inplace:
            return new_emg
        else:
            self.signals = new_emg.signals
            self.channels = new_emg.channels
            self.metadata = new_emg.metadata
            return self

    def get_channel_types(self) -> List[str]:
        """
        Get list of unique channel types in the data.

        Returns:
            List of channel types (e.g., ['EMG', 'ACC', 'GYRO'])
        """
        return list(set(info['channel_type'] for info in self.channels.values()))

    def get_channels_by_type(self, channel_type: str) -> List[str]:
        """
        Get list of channels of a specific type.

        Args:
            channel_type: Type of channels to get ('EMG', 'ACC', 'GYRO', etc.)

        Returns:
            List of channel names of the specified type
        """
        return [ch for ch, info in self.channels.items()
                if info['channel_type'] == channel_type]

    def plot_signals(
            self, channels: Optional[List[str]] = None,
            time_range: Optional[tuple] = None,
            offset_scale: float = 0.8,
            uniform_scale: bool = True,
            detrend: bool = False,
            grid: bool = True,
            title: Optional[str] = None,
            show: bool = True,
            plt_module: Any = plt) -> None:
        """
        Plot EMG signals in a single plot with vertical offsets.

        Args:
            channels: List of channels to plot. If None, plot all channels
            time_range: Tuple of (start_time, end_time) to plot. If None, plot all data
            offset_scale: Portion of allocated space each signal can use (0.0 to 1.0)
            uniform_scale: Whether to use the same scale for all signals
            detrend: Whether to remove mean from signals before plotting
            grid: Whether to show grid lines
            title: Optional title for the figure
            show: Whether to display the plot
            plt_module: Matplotlib pyplot module to use
        """
        if self.signals is None:
            raise ValueError("No signals loaded")

        if channels is None:
            channels = self.signals.columns
        elif not all(ch in self.signals.columns for ch in channels):
            missing = [ch for ch in channels if ch not in self.signals.columns]
            raise ValueError(f"Channels not found: {missing}")

        # Create figure
        fig, ax = plt_module.subplots(figsize=(12, 8))

        # Set figure title if provided
        if title:
            ax.set_title(title, fontsize=14, pad=20)

        # Process signals
        processed_data = {}
        max_range = 0

        for i, channel in enumerate(channels):
            data = self.signals[channel]
            if time_range:
                start, end = time_range
                data = data.loc[start:end]

            # Detrend if requested
            if detrend:
                data = data - data.mean()

            processed_data[channel] = data
            max_range = max(max_range, data.max() - data.min())

        # Plot each signal with offset
        n_channels = len(channels)
        yticks = []
        yticklabels = []

        for i, channel in enumerate(channels):
            data = processed_data[channel]

            # Calculate offset and scaling
            offset = n_channels - i - 1  # Reverse order (top to bottom)
            if uniform_scale:
                scale = offset_scale / max_range
            else:
                scale = offset_scale / (data.max() - data.min())

            # Scale and offset the signal
            scaled_data = data * scale + offset

            # Plot the signal
            ax.plot(data.index, scaled_data, linewidth=1, label=channel)

            # Store tick position and label
            yticks.append(offset)
            yticklabels.append(f"{channel}")

        # Set axis labels and ticks
        ax.set_xlabel("Time (s)")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        # Set y-axis limits with some padding
        ax.set_ylim(-0.5, n_channels - 0.5)

        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust layout
        plt_module.tight_layout()
        if show:
            plt_module.show()

    def to_edf(self, filepath: str, method: str = 'both',
               fft_noise_range: tuple = None, svd_rank: int = None,
               precision_threshold: float = 0.01, format: str = 'auto',
               force_format: bool = False, verify: bool = False,
               verify_tolerance: float = 1e-6, **kwargs) -> Optional[dict]:
        """
        Export data to EDF/BDF format with corresponding channels.tsv file.

        Args:
            filepath: Path to save the EDF/BDF file
            method: Method for signal analysis ('svd', 'fft', or 'both')
                'svd': Uses Singular Value Decomposition for noise floor estimation
                'fft': Uses Fast Fourier Transform for noise floor estimation
                'both': Uses both methods and takes the minimum noise floor (default)
            fft_noise_range: Optional tuple (min_freq, max_freq) specifying frequency range for noise in FFT method
            svd_rank: Optional manual rank cutoff for signal/noise separation in SVD method
            precision_threshold: Maximum acceptable precision loss percentage (default: 0.01%)
            format: Format to use ('auto', 'edf', or 'bdf'). Default is 'auto' which selects
                based on signal characteristics. When specified, the system will still warn
                if the chosen format is not optimal.
            force_format: When True, bypasses all format suitability checks and uses the
                specified format without warnings. Default is False.
            verify: If True, reload the exported file and compare signals with the original
                    to check for data integrity loss. Results are printed. (default: False)
            verify_tolerance: Absolute tolerance used when comparing signals during verification. (default: 1e-6)
            **kwargs: Additional arguments for the EDF exporter

        Returns:
            Optional[dict]: If verify is True, returns a dictionary with verification results.
                           Otherwise, returns None.

        Raises:
            ValueError: If no signals are loaded
        """
        if self.signals is None:
            raise ValueError("No signals loaded")

        from ..exporters.edf import EDFExporter
        from ..analysis.signal import compare_signals
        from .emg import EMG

        # Pass analysis parameters to the exporter
        analysis_params = {
            'method': method,
            'fft_noise_range': fft_noise_range,
            'svd_rank': svd_rank,
            'precision_threshold': precision_threshold
        }

        # Add format parameters
        if format != 'auto':
            analysis_params['format'] = format
        if force_format:
            analysis_params['force_format'] = force_format

        # Combine with any other kwargs
        all_params = {**analysis_params, **kwargs}

        # Perform export
        EDFExporter.export(self, filepath, **all_params)

        verification_results = None
        if verify:
            logging.info(f"Verification requested. Reloading exported file: {filepath}")
            try:
                # Reload the exported file
                reloaded_emg = EMG.from_file(filepath, importer='edf')

                logging.info("Comparing original signals with reloaded signals...")
                # Compare signals
                verification_results = compare_signals(self, reloaded_emg, tolerance=verify_tolerance)

                # Report results
                logging.info("--- Verification Report ---")
                if verification_results.get('channel_diffs', {}).get('added'):
                    logging.warning(f"Channels added during export/import: {verification_results['channel_diffs']['added']}")
                if verification_results.get('channel_diffs', {}).get('removed'):
                    logging.warning(f"Channels removed during export/import: {verification_results['channel_diffs']['removed']}")

                all_identical = True
                for channel, metrics in verification_results.items():
                    if channel == 'channel_diffs': continue # Skip metadata key
                    if not metrics['is_identical']:
                        all_identical = False
                        logging.warning(f"Channel '{channel}': Signals differ (RMSE: {metrics['rmse']:.2e}, MaxDiff: {metrics['max_abs_diff']:.2e}, Corr: {metrics['correlation']:.6f}, SNR_Diff: {metrics['snr_diff_db']:.1f} dB)")
                    else:
                         logging.info(f"Channel '{channel}': Signals are identical (within tolerance {verify_tolerance:.1e}).")

                if all_identical:
                    logging.info("Verification successful: All common channel signals are identical within tolerance.")
                else:
                    logging.warning("Verification finished: Some differences found.")
                logging.info("---------------------------")

            except Exception as e:
                logging.error(f"Verification failed during reload or comparison: {e}")
                verification_results = {'error': str(e)}

        return verification_results

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

    def add_channel(
            self, label: str, data: np.ndarray, sample_frequency: float,
            physical_dimension: str, prefilter: str = 'n/a', channel_type: str = 'EMG') -> None:
        """
        Add a new channel to the EMG data.

        Args:
            label: Channel label or name (as per EDF specification)
            data: Channel data
            sample_frequency: Sampling frequency in Hz (as per EDF specification)
            physical_dimension: Physical dimension/unit of measurement (as per EDF specification)
            prefilter: Pre-filtering applied to the channel
            channel_type: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        if self.signals is None:
            # Create DataFrame with time index
            time = np.arange(len(data)) / sample_frequency
            self.signals = pd.DataFrame(index=time)

        self.signals[label] = data
        self.channels[label] = {
            'sample_frequency': sample_frequency,
            'physical_dimension': physical_dimension,
            'prefilter': prefilter,
            'channel_type': channel_type
        }
