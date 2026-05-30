import logging
import os
from typing import Literal

import numpy as np
import pandas as pd

from ..analysis.verification import compare_signals, report_verification_results
from ..visualization.static import plot_comparison
from ..visualization.static import plot_signals as static_plot_signals
from .modality import (
    infer_modality_from_channel_type,
    validate_channel_type,
    validate_modality,
)

# --- Configuration ---
enable_logging = False  # Set to False to disable most logging

# Configure logging
if enable_logging:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
else:
    logging.basicConfig(level=logging.CRITICAL)  # Effectively turns off most logging


class EMG:
    """
    Core EMG class for handling EMG data and metadata.

    Attributes:
        signals (pd.DataFrame): Raw signal data with time as index.
        metadata (dict): Metadata dictionary containing recording information.
        channels (dict): Channel information including type, unit, sampling frequency.
        events (pd.DataFrame): Annotations or events associated with the signals,
                               with columns 'onset', 'duration', 'description'.
    """

    def __init__(self):
        """Initialize an empty EMG object."""
        self.signals = None
        self.metadata = {}
        self.channels = {}
        # Initialize events as an empty DataFrame with specified columns
        self.events = pd.DataFrame(columns=["onset", "duration", "description"])

    def plot_signals(
        self,
        channels=None,
        time_range=None,
        offset_scale=0.8,
        uniform_scale=True,
        detrend=False,
        grid=True,
        title=None,
        show=True,
        plt_module=None,
    ):
        """
        Plot EMG signals in a single plot with vertical offsets.

        Args:
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
        # Delegate to the static plotting function in visualization module
        static_plot_signals(
            emg_object=self,
            channels=channels,
            time_range=time_range,
            offset_scale=offset_scale,
            uniform_scale=uniform_scale,
            detrend=detrend,
            grid=grid,
            title=title,
            show=show,
            plt_module=plt_module,
        )

    @classmethod
    def _infer_importer(cls, filepath: str) -> str:
        """
        Infer the importer to use based on the file extension.
        """
        extension = os.path.splitext(filepath)[1].lower()
        if extension in {".edf", ".bdf"}:
            return "edf"
        elif extension in {".set"}:
            return "eeglab"
        elif extension in {".otb", ".otb+"}:
            return "otb"
        elif extension in {".csv", ".txt"}:
            return "csv"
        elif extension in {".hea", ".dat", ".atr"}:
            return "wfdb"
        elif extension in {".xdf", ".xdfz"}:
            return "xdf"
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    @classmethod
    def from_file(
        cls,
        filepath: str,
        importer: Literal["trigno", "otb", "eeglab", "edf", "csv", "wfdb", "xdf"] | None = None,
        force_csv: bool = False,
        bids_channels: str = "auto",
        **kwargs,
    ) -> "EMG":
        """
        The method to create EMG object from file.

        Args:
            filepath: Path to the input file
            importer: Name of the importer to use. Can be one of the following:
                - 'trigno': Delsys Trigno EMG system (CSV)
                - 'otb': OTB/OTB+ EMG system (OTB, OTB+)
                - 'eeglab': EEGLAB .set files (SET)
                - 'edf': EDF/EDF+/BDF/BDF+ format (EDF, BDF)
                - 'csv': Generic CSV (or TXT) files with columnar data
                - 'wfdb': Waveform Database (WFDB)
                - 'xdf': XDF format (multi-stream Lab Streaming Layer files)
                If None, the importer will be inferred from the file extension.
                Automatic import is supported for CSV/TXT files.
            force_csv: If True and importer is 'csv', forces using the generic CSV
                      importer even if the file appears to match a specialized format.
            bids_channels: When 'auto' (default), look for a sibling BIDS
                      _channels.tsv next to the file and apply its per-channel
                      type/units over the importer's inferred values. Pass 'off'
                      to disable.
            **kwargs: Additional arguments passed to the importer.
                For XDF files, useful kwargs include:
                - stream_names: List of stream names to import
                - stream_types: List of stream types to import (e.g., ["EMG", "EXG"])
                - stream_ids: List of stream IDs to import

        Returns:
            EMG: New EMG object with loaded data
        """
        if importer is None:
            importer = cls._infer_importer(filepath)

        importers = {
            "trigno": "TrignoImporter",  # CSV with Delsys Trigno Headers
            "otb": "OTBImporter",  # OTB/OTB+ EMG system data
            "edf": "EDFImporter",  # EDF/EDF+/BDF format
            "eeglab": "EEGLABImporter",  # EEGLAB .set files
            "csv": "CSVImporter",  # Generic CSV/Text files
            "wfdb": "WFDBImporter",  # Waveform Database format
            "xdf": "XDFImporter",  # XDF multi-stream format
        }

        if importer not in importers:
            raise ValueError(
                f"Unsupported importer: {importer}. "
                f"Available importers: {list(importers.keys())}\n"
                "- trigno: Delsys Trigno EMG system\n"
                "- otb: OTB/OTB+ EMG system\n"
                "- edf: EDF/EDF+/BDF format\n"
                "- eeglab: EEGLAB .set files\n"
                "- csv: Generic CSV/Text files\n"
                "- wfdb: Waveform Database\n"
                "- xdf: XDF multi-stream format"
            )

        # If using CSV importer and force_csv is set, pass it as force_generic
        if importer == "csv":
            kwargs["force_generic"] = force_csv

        # Import the appropriate importer class
        importer_module = __import__(
            f"emgio.importers.{importer}", globals(), locals(), [importers[importer]]
        )
        importer_class = getattr(importer_module, importers[importer])

        # Create importer instance and load data
        emg = importer_class().load(filepath, **kwargs)

        # In a BIDS layout, the sibling _channels.tsv is the authoritative source
        # of per-channel type/units; apply it over the importer's header/label
        # guesses unless explicitly disabled with bids_channels="off".
        if bids_channels != "off":
            from ..bids import apply_channels_tsv, find_channels_tsv

            channels_tsv = find_channels_tsv(filepath)
            if channels_tsv:
                apply_channels_tsv(emg, channels_tsv)

        return emg

    def select_channels(
        self,
        channels: str | list[str] | None = None,
        channel_type: str | None = None,
        inplace: bool = False,
        *,
        modality: str | None = None,
    ) -> "EMG":
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

        if channels is None and channel_type is None and modality is None:
            raise ValueError("Specify at least one of: channels, channel_type, or modality.")

        # If type/modality specified but no channels, select all matching channels
        if channels is None and channel_type is not None:
            channels = self.get_channels_by_type(channel_type)
            if not channels:
                raise ValueError(f"No channels found of type: {channel_type}")
        elif channels is None and modality is not None:
            channels = self.get_channels_by_modality(modality)
            if not channels:
                raise ValueError(f"No channels found of modality: {modality}")
        elif isinstance(channels, str):
            channels = [channels]

        # Validate channels exist
        if not all(ch in self.signals.columns for ch in channels):
            missing = [ch for ch in channels if ch not in self.signals.columns]
            raise ValueError(f"Channels not found: {missing}")

        # Filter by type if specified
        if channel_type is not None:
            channels = [ch for ch in channels if self.channels[ch]["channel_type"] == channel_type]
            if not channels:
                raise ValueError(f"None of the selected channels are of type: {channel_type}")

        # Filter by modality if specified
        if modality is not None:
            canonical_modality = validate_modality(modality)
            channels = [
                ch for ch in channels if self.channels[ch].get("modality") == canonical_modality
            ]
            if not channels:
                raise ValueError(f"None of the selected channels are of modality: {modality}")

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

    def get_channel_types(self) -> list[str]:
        """
        Get list of unique channel types in the data.

        Returns:
            List of channel types (e.g., ['EMG', 'ACC', 'GYRO'])
        """
        return list({info["channel_type"] for info in self.channels.values()})

    def get_channels_by_type(self, channel_type: str) -> list[str]:
        """
        Get list of channels of a specific type.

        Args:
            channel_type: Type of channels to get ('EMG', 'ACC', 'GYRO', etc.)

        Returns:
            List of channel names of the specified type
        """
        return [ch for ch, info in self.channels.items() if info["channel_type"] == channel_type]

    def get_modalities(self) -> list[str]:
        """
        Get the list of unique modalities present in the data.

        Returns:
            List of modalities (e.g., ['EEG', 'EMG', 'MISC']).
        """
        return list(
            {info.get("modality") for info in self.channels.values() if info.get("modality")}
        )

    def get_channels_by_modality(self, modality: str) -> list[str]:
        """
        Get the channels belonging to a given modality.

        Args:
            modality: Modality to filter by ('EEG', 'EMG', 'IEEG', 'MEG', 'BEH', 'MISC').

        Returns:
            List of channel names of the specified modality.
        """
        canonical_modality = validate_modality(modality)
        return [
            ch for ch, info in self.channels.items() if info.get("modality") == canonical_modality
        ]

    def to_edf(
        self,
        filepath: str,
        method: str = "both",
        fft_noise_range: tuple = None,
        svd_rank: int = None,
        precision_threshold: float = 0.01,
        format: Literal["auto", "edf", "bdf"] = "auto",
        bypass_analysis: bool | None = None,
        verify: bool = False,
        verify_tolerance: float = 1e-6,
        verify_channel_map: dict[str, str] | None = None,
        verify_plot: bool = False,
        events_df: pd.DataFrame | None = None,
        create_channels_tsv: bool = True,
        **kwargs,
    ) -> str | None:
        """
        Export EMG data to EDF/BDF format, optionally including events.

        Args:
            filepath: Path to save the EDF/BDF file
            method: Method for signal analysis ('svd', 'fft', or 'both')
                'svd': Uses Singular Value Decomposition for noise floor estimation
                'fft': Uses Fast Fourier Transform for noise floor estimation
                'both': Uses both methods and takes the minimum noise floor (default)
            fft_noise_range: Optional tuple (min_freq, max_freq) specifying frequency range for noise in FFT method
            svd_rank: Optional manual rank cutoff for signal/noise separation in SVD method
            precision_threshold: Maximum acceptable precision loss percentage (default: 0.01%)
            format: Format to use ('auto', 'edf', or 'bdf'). Default is 'auto'.
                    If 'edf' or 'bdf' is specified, that format will be used directly.
                    If 'auto', the format (EDF/16-bit or BDF/24-bit) is chosen based
                    on signal analysis to minimize precision loss while preferring EDF
                    if sufficient.
            bypass_analysis: If True, skip signal analysis step when format is explicitly
                             set to 'edf' or 'bdf'. If None (default), analysis is skipped
                             automatically when format is forced. Set to False to force
                             analysis even with a specified format. Ignored if format='auto'.
            verify: If True, reload the exported file and compare signals with the original
                    to check for data integrity loss. Results are printed. (default: False)
            verify_tolerance: Absolute tolerance used when comparing signals during verification. (default: 1e-6)
            verify_channel_map: Optional dictionary mapping original channel names (keys)
                                to reloaded channel names (values) for verification.
                                Used if `verify` is True and channel names might differ.
            verify_plot: If True and verify is True, plots a comparison of original vs reloaded signals.
            events_df: Optional DataFrame with events ('onset', 'duration', 'description').
                      If None, uses self.events. (This provides flexibility)
            create_channels_tsv: If True, create a BIDS-compliant channels.tsv file (default: True)
            **kwargs: Additional arguments for the EDF exporter

        Returns:
            Union[str, None]: If verify is True, returns a string with verification results.
                             Otherwise, returns None.

        Raises:
            ValueError: If no signals are loaded
        """
        from ..exporters.edf import EDFExporter  # Local import

        if self.signals is None:
            raise ValueError("No signals loaded")

        # --- Determine if analysis should be bypassed ---
        final_bypass_analysis = False
        if format.lower() == "auto":
            if bypass_analysis is True:
                logging.warning(
                    "bypass_analysis=True ignored because format='auto'. Analysis is required."
                )
            # Analysis is always needed for 'auto' format
            final_bypass_analysis = False
        elif format.lower() in ["edf", "bdf"]:
            if bypass_analysis is None:
                # Default behaviour: skip analysis if format is forced
                final_bypass_analysis = True
                msg = (
                    f"Format forced to '{format}'. Skipping signal analysis for faster export. "
                    "Set bypass_analysis=False to force analysis."
                )
                logging.log(logging.CRITICAL, msg)
            elif bypass_analysis is True:
                final_bypass_analysis = True
                logging.log(logging.CRITICAL, "bypass_analysis=True set. Skipping signal analysis.")
            else:  # bypass_analysis is False
                final_bypass_analysis = False
                logging.info(
                    f"Format forced to '{format}' but bypass_analysis=False. Performing signal analysis."
                )
        else:
            # Should not happen if Literal type hint works, but good practice
            logging.warning(
                f"Unknown format '{format}'. Defaulting to 'auto' behavior (analysis enabled)."
            )
            format = "auto"
            final_bypass_analysis = False

        # Determine which events DataFrame to use
        if events_df is None:
            events_to_export = self.events
        else:
            events_to_export = events_df

        # Combine parameters
        all_params = {
            "precision_threshold": precision_threshold,
            "method": method,
            "fft_noise_range": fft_noise_range,
            "svd_rank": svd_rank,
            "format": format,
            "bypass_analysis": final_bypass_analysis,
            "events_df": events_to_export,  # Pass the events dataframe
            "create_channels_tsv": create_channels_tsv,
            **kwargs,
        }

        EDFExporter.export(self, filepath, **all_params)

        verification_report_dict = None
        if verify:
            logging.info(f"Verification requested. Reloading exported file: {filepath}")
            try:
                # Reload the exported file
                reloaded_emg = EMG.from_file(filepath, importer="edf")

                logging.info("Comparing original signals with reloaded signals...")
                # Compare signals using the imported function
                verification_results = compare_signals(
                    self, reloaded_emg, tolerance=verify_tolerance, channel_map=verify_channel_map
                )

                # Generate and log report using the imported function
                report_verification_results(verification_results, verify_tolerance)
                verification_report_dict = verification_results

                # Plot comparison using imported function if requested
                summary = verification_results.get("channel_summary", {})
                comparison_mode = summary.get("comparison_mode", "unknown")
                compared_count = sum(1 for k in verification_results if k != "channel_summary")

                if verify_plot and compared_count > 0 and comparison_mode != "failed":
                    plot_comparison(self, reloaded_emg, channel_map=verify_channel_map)
                elif verify_plot:
                    logging.warning(
                        "Skipping verification plot: No channels were successfully compared."
                    )

            except Exception as e:
                logging.error(f"Verification failed during reload or comparison: {e}")
                verification_report_dict = {
                    "error": str(e),
                    "channel_summary": {"comparison_mode": "failed"},
                }

        return verification_report_dict

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
        self,
        label: str,
        data: np.ndarray,
        sample_frequency: float,
        physical_dimension: str,
        channel_type: str,
        *,
        modality: str | None = None,
        prefilter: str = "n/a",
    ) -> None:
        """
        Add a new channel to the EMG data.

        Args:
            label: Channel label or name (as per EDF specification)
            data: Channel data
            sample_frequency: Sampling frequency in Hz (as per EDF specification)
            physical_dimension: Physical dimension/unit of measurement (as per EDF specification)
            channel_type: BIDS channel type ('EEG', 'EMG', 'ECG', 'ACC', 'SEEG', ...).
                Required; validated against the modality vocabulary. There is no
                default (a missing type must be explicit, e.g. 'OTHER'/'MISC').
            modality: Coarse modality ('EEG', 'EMG', 'IEEG', 'MEG', 'BEH', 'MISC').
                If None, it is inferred from ``channel_type``.
            prefilter: Pre-filtering applied to the channel (keyword-only).
        """
        canonical_type = validate_channel_type(channel_type)
        canonical_modality = (
            infer_modality_from_channel_type(canonical_type)
            if modality is None
            else validate_modality(modality)
        )

        if self.signals is None:
            # Create DataFrame with time index
            time = np.arange(len(data)) / sample_frequency
            self.signals = pd.DataFrame(index=time)

        self.signals[label] = data
        self.channels[label] = {
            "sample_frequency": sample_frequency,
            "physical_dimension": physical_dimension,
            "prefilter": prefilter,
            "channel_type": canonical_type,
            "modality": canonical_modality,
        }

    def set_channel(
        self,
        label: str,
        *,
        channel_type: str | None = None,
        modality: str | None = None,
        physical_dimension: str | None = None,
        prefilter: str | None = None,
    ) -> None:
        """
        Update metadata of an existing channel (the supported relabel path).

        Args:
            label: Existing channel label.
            channel_type: New BIDS channel type (validated). When given without an
                explicit ``modality``, the modality is re-derived from it.
            modality: New coarse modality (validated).
            physical_dimension: New physical unit.
            prefilter: New prefilter string.

        Raises:
            KeyError: If ``label`` is not an existing channel.
            ValueError: If ``channel_type`` or ``modality`` is not in the
                modality vocabulary.
        """
        if label not in self.channels:
            raise KeyError(f"Channel not found: {label}")
        info = self.channels[label]
        if channel_type is not None:
            info["channel_type"] = validate_channel_type(channel_type)
            if modality is None:
                info["modality"] = infer_modality_from_channel_type(info["channel_type"])
        if modality is not None:
            info["modality"] = validate_modality(modality)
        if physical_dimension is not None:
            info["physical_dimension"] = physical_dimension
        if prefilter is not None:
            info["prefilter"] = prefilter

    def add_event(self, onset: float, duration: float, description: str) -> None:
        """
        Add an event/annotation to the EMG object.

        Args:
            onset: Event onset time in seconds.
            duration: Event duration in seconds.
            description: Event description string.
        """
        new_event = pd.DataFrame(
            [{"onset": float(onset), "duration": float(duration), "description": description}]
        )
        # Avoid concatenating onto the empty, object-dtype events frame, which
        # would coerce the numeric columns to object. Start from the typed
        # new_event when there are no existing events.
        if self.events is None or self.events.empty:
            self.events = new_event
        else:
            self.events = pd.concat([self.events, new_event], ignore_index=True)
        # Sort events by onset time for consistency
        self.events = self.events.sort_values(by="onset").reset_index(drop=True)
