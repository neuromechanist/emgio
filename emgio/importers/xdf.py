"""XDF (Extensible Data Format) importer for EMG data.

XDF files can contain multiple streams (EMG, EEG, markers, etc.). This module
provides tools to explore XDF contents and selectively import specific streams.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.emg import EMG
from .base import BaseImporter


@dataclass
class XDFStreamInfo:
    """Information about a single XDF stream."""

    stream_id: int
    name: str
    stream_type: str
    channel_count: int
    nominal_srate: float
    effective_srate: float | None
    channel_format: str
    source_id: str
    hostname: str
    sample_count: int
    duration_seconds: float
    channel_labels: list[str]
    channel_types: list[str]
    channel_units: list[str]

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Stream {self.stream_id}: {self.name}",
            f"  Type: {self.stream_type}",
            f"  Channels: {self.channel_count}",
            f"  Nominal srate: {self.nominal_srate} Hz",
        ]
        if self.effective_srate:
            lines.append(f"  Effective srate: {self.effective_srate:.2f} Hz")
        lines.extend(
            [
                f"  Samples: {self.sample_count}",
                f"  Duration: {self.duration_seconds:.2f} s",
                f"  Format: {self.channel_format}",
            ]
        )
        if self.channel_labels:
            labels_preview = ", ".join(self.channel_labels[:5])
            if len(self.channel_labels) > 5:
                labels_preview += f", ... (+{len(self.channel_labels) - 5} more)"
            lines.append(f"  Channel labels: {labels_preview}")
        return "\n".join(lines)


@dataclass
class XDFSummary:
    """Summary of an XDF file's contents."""

    filepath: str
    streams: list[XDFStreamInfo]
    header_info: dict[str, Any]

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"XDF File: {self.filepath}",
            f"Number of streams: {len(self.streams)}",
            "",
        ]
        for stream in self.streams:
            lines.append(str(stream))
            lines.append("")
        return "\n".join(lines)

    def get_streams_by_type(self, stream_type: str) -> list[XDFStreamInfo]:
        """Get all streams of a specific type (case-insensitive)."""
        return [s for s in self.streams if s.stream_type.upper() == stream_type.upper()]

    def get_stream_by_name(self, name: str) -> XDFStreamInfo | None:
        """Get a stream by name (case-insensitive)."""
        for stream in self.streams:
            if stream.name.lower() == name.lower():
                return stream
        return None

    def get_stream_by_id(self, stream_id: int) -> XDFStreamInfo | None:
        """Get a stream by its ID."""
        for stream in self.streams:
            if stream.stream_id == stream_id:
                return stream
        return None


def summarize_xdf(filepath: str | Path) -> XDFSummary:
    """
    Summarize the contents of an XDF file without fully loading all data.

    This function loads the XDF file and extracts metadata about all streams,
    including channel names, types, sampling rates, and data shapes. Use this
    to explore an XDF file before deciding which streams to import.

    Args:
        filepath: Path to the XDF file

    Returns:
        XDFSummary: Object containing information about all streams in the file

    Example:
        >>> summary = summarize_xdf("recording.xdf")
        >>> print(summary)
        >>> # Find EMG streams
        >>> emg_streams = summary.get_streams_by_type("EMG")
    """
    try:
        import pyxdf
    except ImportError as e:
        raise ImportError(
            "pyxdf is required for XDF file support. Install it with: pip install pyxdf"
        ) from e

    filepath = str(filepath)
    data, header = pyxdf.load_xdf(filepath)

    streams = []
    for stream in data:
        info = stream["info"]

        # Extract basic info
        stream_id = info.get("stream_id", 0)
        name = info["name"][0] if "name" in info else "Unknown"
        stream_type = info["type"][0] if "type" in info else "Unknown"
        channel_count = int(info["channel_count"][0]) if "channel_count" in info else 0
        nominal_srate = float(info["nominal_srate"][0]) if "nominal_srate" in info else 0.0
        effective_srate = stream.get("effective_srate")
        channel_format = info["channel_format"][0] if "channel_format" in info else "unknown"
        source_id = info["source_id"][0] if "source_id" in info else ""
        hostname = info["hostname"][0] if "hostname" in info else ""

        # Get data shape info - handle both numpy arrays and lists (marker streams)
        time_series = stream["time_series"]
        if isinstance(time_series, np.ndarray):
            sample_count = time_series.shape[0] if time_series.ndim > 0 else 0
        elif isinstance(time_series, list):
            sample_count = len(time_series)
        else:
            sample_count = 0

        # Calculate duration
        timestamps = stream.get("time_stamps", np.array([]))
        if len(timestamps) > 1:
            duration_seconds = timestamps[-1] - timestamps[0]
        elif effective_srate and effective_srate > 0 and sample_count > 0:
            duration_seconds = sample_count / effective_srate
        else:
            duration_seconds = 0.0

        # Extract channel info from desc
        channel_labels = []
        channel_types = []
        channel_units = []

        if "desc" in info and info["desc"] and info["desc"][0]:
            desc = info["desc"][0]
            if isinstance(desc, dict) and "channels" in desc and desc["channels"]:
                channels_info = desc["channels"][0]
                if isinstance(channels_info, dict) and "channel" in channels_info:
                    for ch in channels_info["channel"]:
                        if isinstance(ch, dict):
                            label = ch.get("label", [""])[0] if "label" in ch else ""
                            ch_type = ch.get("type", [""])[0] if "type" in ch else ""
                            unit = ch.get("unit", [""])[0] if "unit" in ch else ""
                            channel_labels.append(label)
                            channel_types.append(ch_type)
                            channel_units.append(unit)

        # If no channel info in desc, create default labels
        if not channel_labels:
            channel_labels = [f"Ch{i + 1}" for i in range(channel_count)]
            channel_types = [""] * channel_count
            channel_units = [""] * channel_count

        stream_info = XDFStreamInfo(
            stream_id=stream_id,
            name=name,
            stream_type=stream_type,
            channel_count=channel_count,
            nominal_srate=nominal_srate,
            effective_srate=effective_srate,
            channel_format=channel_format,
            source_id=source_id,
            hostname=hostname,
            sample_count=sample_count,
            duration_seconds=duration_seconds,
            channel_labels=channel_labels,
            channel_types=channel_types,
            channel_units=channel_units,
        )
        streams.append(stream_info)

    header_info = dict(header.get("info", {})) if header else {}

    return XDFSummary(filepath=filepath, streams=streams, header_info=header_info)


class XDFImporter(BaseImporter):
    """
    Importer for XDF (Extensible Data Format) files.

    XDF files can contain multiple data streams. This importer allows selective
    import of specific streams by name, type, or ID.

    Example:
        >>> # First, explore the file
        >>> from emgio.importers.xdf import summarize_xdf
        >>> summary = summarize_xdf("recording.xdf")
        >>> print(summary)
        >>>
        >>> # Import specific streams
        >>> importer = XDFImporter()
        >>> emg = importer.load("recording.xdf", stream_names=["EMG_stream"])
        >>>
        >>> # Or import by type
        >>> emg = importer.load("recording.xdf", stream_types=["EMG", "EXG"])
    """

    def load(
        self,
        filepath: str,
        stream_names: list[str] | None = None,
        stream_types: list[str] | None = None,
        stream_ids: list[int] | None = None,
        sync_streams: bool = True,
        default_channel_type: str = "EMG",
        include_timestamps: bool = False,
    ) -> EMG:
        """
        Load EMG data from an XDF file.

        Streams can be selected by name, type, or ID. If multiple selection
        criteria are provided, streams matching ANY criterion are included.
        If no selection criteria are provided, all streams with numeric data
        are loaded.

        Args:
            filepath: Path to the XDF file
            stream_names: List of stream names to import (case-insensitive)
            stream_types: List of stream types to import (e.g., ["EMG", "EXG"])
            stream_ids: List of stream IDs to import
            sync_streams: If True, synchronize streams to common timestamps.
                         If False, only the first matching stream is loaded.
            default_channel_type: Default channel type for channels without
                                 explicit type info (default: "EMG")
            include_timestamps: If True, add a timestamp channel for each stream
                               named "{stream_name}_LSL_timestamps" containing
                               the original LSL timestamps. Useful for preserving
                               timing information when exporting to formats like
                               EDF that require regular sampling.

        Returns:
            EMG: EMG object containing the loaded data

        Raises:
            ValueError: If no matching streams found or file cannot be read
            ImportError: If pyxdf is not installed
        """
        try:
            import pyxdf
        except ImportError as e:
            raise ImportError(
                "pyxdf is required for XDF file support. Install it with: pip install pyxdf"
            ) from e

        filepath = str(filepath)
        data, header = pyxdf.load_xdf(filepath)

        if not data:
            raise ValueError(f"No streams found in XDF file: {filepath}")

        # Filter streams based on selection criteria
        selected_streams = self._select_streams(data, stream_names, stream_types, stream_ids)

        if not selected_streams:
            # If no criteria specified, select all streams with numeric data
            if stream_names is None and stream_types is None and stream_ids is None:
                selected_streams = [
                    s
                    for s in data
                    if isinstance(s["time_series"], np.ndarray)
                    and s["time_series"].dtype.kind in "iufc"
                ]
            if not selected_streams:
                raise ValueError(
                    "No matching streams found. Use summarize_xdf() to explore the file."
                )

        # Create EMG object
        emg = EMG()

        # Store metadata
        emg.set_metadata("source_file", filepath)
        emg.set_metadata("device", "XDF")
        emg.set_metadata("stream_count", len(selected_streams))

        if sync_streams and len(selected_streams) > 1:
            self._load_synchronized_streams(
                emg, selected_streams, default_channel_type, include_timestamps
            )
        else:
            # Load streams independently (use first stream for time base)
            self._load_streams(emg, selected_streams, default_channel_type, include_timestamps)

        return emg

    def _select_streams(
        self,
        data: list[dict],
        stream_names: list[str] | None,
        stream_types: list[str] | None,
        stream_ids: list[int] | None,
    ) -> list[dict]:
        """Select streams based on criteria."""
        if stream_names is None and stream_types is None and stream_ids is None:
            return []  # Return empty to trigger "all streams" behavior

        selected = []
        for stream in data:
            info = stream["info"]
            name = info["name"][0] if "name" in info else ""
            stype = info["type"][0] if "type" in info else ""
            sid = info.get("stream_id", 0)

            # Check name match (case-insensitive)
            if stream_names and any(name.lower() == n.lower() for n in stream_names):
                selected.append(stream)
                continue

            # Check type match (case-insensitive)
            if stream_types and any(stype.upper() == t.upper() for t in stream_types):
                selected.append(stream)
                continue

            # Check ID match
            if stream_ids and sid in stream_ids:
                selected.append(stream)
                continue

        return selected

    def _load_streams(
        self,
        emg: EMG,
        streams: list[dict],
        default_channel_type: str,
        include_timestamps: bool = False,
    ) -> None:
        """Load streams without synchronization."""
        all_data = {}
        all_timestamps = None
        base_srate = None
        stream_timestamp_data = {}  # Store timestamp data per stream

        for stream in streams:
            info = stream["info"]
            stream_name = info["name"][0] if "name" in info else "Unknown"
            time_series = stream["time_series"]
            timestamps = stream["time_stamps"]

            # Skip non-numpy arrays (e.g., marker streams are lists) or non-numeric data
            if not isinstance(time_series, np.ndarray):
                continue
            if time_series.dtype.kind not in "iufc" or len(time_series) == 0:
                continue

            # Get sampling rate
            srate = stream.get("effective_srate")
            if not srate:
                srate = float(info["nominal_srate"][0]) if "nominal_srate" in info else 0.0

            # Use first stream's timestamps as base
            if all_timestamps is None:
                all_timestamps = timestamps
                base_srate = srate

            # Store timestamp data for this stream if requested
            if include_timestamps:
                stream_timestamp_data[stream_name] = {
                    "timestamps": timestamps,
                    "srate": srate,
                }

            # Get channel info
            channel_labels, channel_types, channel_units = self._extract_channel_info(
                info, time_series.shape[1] if time_series.ndim > 1 else 1, stream_name
            )

            # Handle 1D data (single channel)
            if time_series.ndim == 1:
                time_series = time_series.reshape(-1, 1)

            # Add channels
            for i, label in enumerate(channel_labels):
                if i < time_series.shape[1]:
                    # Make label unique if needed
                    unique_label = label
                    counter = 1
                    while unique_label in all_data:
                        unique_label = f"{label}_{counter}"
                        counter += 1

                    all_data[unique_label] = {
                        "data": time_series[:, i],
                        "timestamps": timestamps,
                        "srate": srate,
                        "unit": channel_units[i] if i < len(channel_units) else "uV",
                        "type": channel_types[i] if channel_types[i] else default_channel_type,
                    }

        if not all_data:
            raise ValueError("No valid data found in selected streams")

        # Use the first stream's timestamps for the DataFrame index
        # Convert to relative time starting from 0
        if all_timestamps is not None and len(all_timestamps) > 0:
            time_index = all_timestamps - all_timestamps[0]
        else:
            time_index = np.arange(len(next(iter(all_data.values()))["data"])) / base_srate

        # Create DataFrame
        df = pd.DataFrame(index=time_index)

        for label, ch_info in all_data.items():
            # Resample if needed (different stream lengths)
            ch_data = ch_info["data"]
            if len(ch_data) != len(time_index):
                # Interpolate to match base timestamps
                ch_timestamps = ch_info["timestamps"] - ch_info["timestamps"][0]
                ch_data = np.interp(time_index, ch_timestamps, ch_data)

            df[label] = ch_data

            emg.channels[label] = {
                "sample_frequency": ch_info["srate"] if ch_info["srate"] else base_srate,
                "physical_dimension": ch_info["unit"],
                "prefilter": "n/a",
                "channel_type": ch_info["type"],
            }

        # Add timestamp channels if requested
        if include_timestamps and stream_timestamp_data:
            for stream_name, ts_info in stream_timestamp_data.items():
                ts_label = f"{stream_name}_LSL_timestamps"
                original_timestamps = ts_info["timestamps"]

                # Resample timestamps to match the common time index
                if len(original_timestamps) != len(time_index):
                    relative_ts = original_timestamps - original_timestamps[0]
                    resampled_ts = np.interp(time_index, relative_ts, original_timestamps)
                else:
                    resampled_ts = original_timestamps

                df[ts_label] = resampled_ts

                emg.channels[ts_label] = {
                    "sample_frequency": ts_info["srate"] if ts_info["srate"] else base_srate,
                    "physical_dimension": "s",  # seconds
                    "prefilter": "n/a",
                    "channel_type": "MISC",  # Miscellaneous channel type
                }

        emg.signals = df
        emg.set_metadata("srate", base_srate)

    def _load_synchronized_streams(
        self,
        emg: EMG,
        streams: list[dict],
        default_channel_type: str,
        include_timestamps: bool = False,
    ) -> None:
        """Load streams with timestamp synchronization."""
        # For now, use the same approach as _load_streams
        # pyxdf already handles synchronization during load
        self._load_streams(emg, streams, default_channel_type, include_timestamps)

    def _extract_channel_info(
        self,
        info: dict,
        n_channels: int,
        stream_name: str,
    ) -> tuple:
        """Extract channel labels, types, and units from stream info."""
        channel_labels = []
        channel_types = []
        channel_units = []

        if "desc" in info and info["desc"] and info["desc"][0]:
            desc = info["desc"][0]
            if isinstance(desc, dict) and "channels" in desc and desc["channels"]:
                channels_info = desc["channels"][0]
                if isinstance(channels_info, dict) and "channel" in channels_info:
                    for ch in channels_info["channel"]:
                        if isinstance(ch, dict):
                            label = ch.get("label", [""])[0] if "label" in ch else ""
                            ch_type = ch.get("type", [""])[0] if "type" in ch else ""
                            unit = ch.get("unit", [""])[0] if "unit" in ch else ""
                            channel_labels.append(
                                label if label else f"{stream_name}_Ch{len(channel_labels) + 1}"
                            )
                            channel_types.append(ch_type)
                            channel_units.append(unit if unit else "uV")

        # Fill in missing labels
        while len(channel_labels) < n_channels:
            channel_labels.append(f"{stream_name}_Ch{len(channel_labels) + 1}")
            channel_types.append("")
            channel_units.append("uV")

        return channel_labels, channel_types, channel_units


def _determine_channel_type_from_label(label: str) -> str:
    """Determine channel type based on label naming conventions."""
    label_upper = label.upper()

    if "EMG" in label_upper or "MUS" in label_upper:
        return "EMG"
    elif "ACC" in label_upper:
        return "ACC"
    elif "GYRO" in label_upper:
        return "GYRO"
    elif "EEG" in label_upper or label_upper in [
        "FP1",
        "FP2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "FZ",
        "CZ",
        "PZ",
        "OZ",
    ]:
        return "EEG"
    elif "ECG" in label_upper or "EKG" in label_upper:
        return "ECG"
    elif "EOG" in label_upper:
        return "EOG"
    elif "TRIG" in label_upper or "MARKER" in label_upper or "EVENT" in label_upper:
        return "TRIG"

    return ""
