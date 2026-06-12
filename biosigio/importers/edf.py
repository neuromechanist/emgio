from typing import cast

import numpy as np
import pandas as pd
import pyedflib

from ..core.emg import Recording
from .base import BaseImporter

# Accepted `mixed_rate` policies for a recording whose signals carry differing
# per-channel sampling rates (EDF/BDF allow this; e.g. PSG: EEG ~200 Hz + SpO2
# ~12.5 Hz). "error" (default) refuses to load -- biosigIO stores one uniform grid
# and silently fabricating a common grid would be unfaithful. "resample" opts in to
# upsampling the slower channels to the fastest rate so the whole recording lands on
# one grid; this is a lossy DERIVED view (fine for a serving/plot copy, not the
# authoritative source), and the original per-channel rate is preserved on each
# channel as ``original_sample_frequency`` for provenance.
MIXED_RATE_POLICIES = ("error", "resample")


def _resample_to_length(data: np.ndarray, n_out: int) -> np.ndarray:
    """Polyphase-resample a 1-D channel to exactly ``n_out`` samples.

    Used only by the mixed-rate ``resample`` path to lift a slower channel onto the
    fastest channel's grid. The integer ratio ``n_out/len(data)`` drives a
    Kaiser-windowed ``resample_poly`` (the same anti-alias resampler the Zarr
    exporter uses), then the result is trimmed/edge-padded to the exact length so
    every channel stays length-aligned on the shared grid.
    """
    from math import gcd

    from scipy.signal import resample_poly

    x = np.asarray(data, dtype=float)
    n_in = len(x)
    if n_in == 0 or n_in == n_out:
        return x
    g = gcd(n_out, n_in)
    # padtype="line" extends each edge with a ramp matching the signal's end slope,
    # not zeros. Slow physiological channels (SpO2, respiration) sit on a large DC
    # offset; default zero-padding makes a step at the boundary and the polyphase FIR
    # rings (a 95-unit SpO2 baseline overshot to ~109). "line" removes that transient.
    y = resample_poly(x, n_out // g, n_in // g, padtype="line")
    if len(y) > n_out:
        y = y[:n_out]
    elif len(y) < n_out:
        y = np.pad(y, (0, n_out - len(y)), mode="edge")
    return y


class EDFImporter(BaseImporter):
    """Importer for EDF/EDF+/BDF format files."""

    def _determine_channel_type(self, label: str, transducer: str) -> str:
        """
        Determine channel type based on label and transducer info.

        Args:
            label: Channel label from EDF header
            transducer: Transducer type from EDF header

        Returns:
            str: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        label = label.upper()
        transducer = transducer.upper()

        if "EMG" in label or "EMG" in transducer:
            return "EMG"
        elif "EEG" in label or "EEG" in transducer:
            return "EEG"
        elif "ECG" in label or "EKG" in label or "ECG" in transducer:
            return "ECG"
        elif "EOG" in label or "EOG" in transducer:
            return "EOG"
        elif "ACC" in label or "ACCELEROMETER" in transducer:
            return "ACC"
        elif "GYRO" in label or "GYROSCOPE" in transducer:
            return "GYRO"
        elif "TRIG" in label or "TRIGGER" in transducer:
            return "TRIG"
        else:
            return "OTHER"

    def _extract_metadata(self, edf_reader: pyedflib.EdfReader) -> dict:
        """
        Extract metadata from EDF file header.

        Args:
            edf_reader: pyedflib EdfReader instance

        Returns:
            dict: Dictionary containing file metadata
        """
        header = edf_reader.getHeader()
        signals_headers = edf_reader.getSignalHeaders()

        metadata = {
            "recording_info": {
                "startdate": header.get("startdate", None),
                "patientcode": header.get("patientcode", ""),
                "gender": header.get("gender", ""),
                "birthdate": header.get("birthdate", ""),
                "patient_name": header.get("patient_name", ""),
                "patient_additional": header.get("patient_additional", ""),
                "admincode": header.get("admincode", ""),
                "technician": header.get("technician", ""),
                "equipment": header.get("equipment", ""),
                "recording_additional": header.get("recording_additional", ""),
            },
            "file_info": {
                "filetype": header.get("filetype", 0),  # 0: EDF, 1: EDF+, 2: BDF+
                "number_of_signals": len(signals_headers),
                "file_duration": header.get("file_duration", 0),
                "datarecord_duration": header.get("datarecord_duration", 0),
            },
        }

        return metadata

    def _read_signal_data(
        self, edf_reader: pyedflib.EdfReader, signal_idx: int
    ) -> tuple[np.ndarray, dict]:
        """
        Read signal data and header information for a specific channel.

        Args:
            edf_reader: pyedflib EdfReader instance
            signal_idx: Index of the signal to read

        Returns:
            tuple: (signal_data, signal_info)
        """
        # Get signal header
        signal_header = edf_reader.getSignalHeaders()[signal_idx]

        # Read the signal data
        signal_data = edf_reader.readSignal(signal_idx)

        # Extract signal information
        signal_info = {
            "label": cast(str, signal_header["label"]).strip(),
            "transducer": cast(str, signal_header.get("transducer", "")).strip(),
            "physical_dimension": cast(str, signal_header.get("dimension", "")).strip() or "n/a",
            "physical_min": signal_header["physical_min"],
            "physical_max": signal_header["physical_max"],
            "digital_min": signal_header["digital_min"],
            "digital_max": signal_header["digital_max"],
            "prefilter": cast(str, signal_header.get("prefilter", "")).strip() or "n/a",
            "sample_frequency": signal_header["sample_frequency"],
        }

        return signal_data, signal_info

    def _read_annotations(self, edf_reader: pyedflib.EdfReader) -> pd.DataFrame:
        """Read EDF+/BDF+ annotations (events) into an events DataFrame.

        pyedflib parses the dedicated annotations channel and returns parallel
        arrays of onsets (seconds), durations (seconds), and descriptions.
        Plain EDF (no annotations) yields empty arrays. Descriptions may be
        bytes or str depending on the pyedflib version; both are handled, and
        empty placeholder annotations are skipped.

        Returns:
            pd.DataFrame: events with float64 ``onset``/``duration`` and string
            ``description``, sorted by onset (matching :meth:`Recording.add_event`).
        """
        onsets, durations, descriptions = edf_reader.readAnnotations()
        rows = []
        for onset, duration, description in zip(onsets, durations, descriptions, strict=False):
            text = (
                description.decode("utf-8", "replace")
                if isinstance(description, bytes)
                else str(description)
            )
            if text == "":
                continue  # skip empty placeholder annotations
            rows.append((float(onset), float(duration), text))

        events = pd.DataFrame(rows, columns=["onset", "duration", "description"])
        if not events.empty:
            events = events.sort_values(by="onset").reset_index(drop=True)
            events["onset"] = events["onset"].astype("float64")
            events["duration"] = events["duration"].astype("float64")
        return events

    def load(self, filepath: str, *, mixed_rate: str = "error") -> Recording:
        """
        Load EMG data from EDF/EDF+/BDF file.

        Args:
            filepath: Path to the EDF file
            mixed_rate: Policy when the file's signals carry differing per-channel
                sampling rates (EDF/BDF allow it, e.g. polysomnography). ``"error"``
                (default) raises, since biosigIO stores one uniform grid and a
                fabricated common grid would be unfaithful; ``"resample"`` upsamples
                the slower channels to the fastest rate (a lossy derived view -- the
                original rate is kept on each channel as ``original_sample_frequency``).

        Returns:
            Recording: Recording object containing the loaded data
        """
        if mixed_rate not in MIXED_RATE_POLICIES:
            raise ValueError(f"mixed_rate must be one of {MIXED_RATE_POLICIES}, got {mixed_rate!r}")
        try:
            edf_reader = pyedflib.EdfReader(filepath)

            # Create Recording object
            rec = Recording()

            # Extract and store metadata
            metadata = self._extract_metadata(edf_reader)
            for key, value in metadata["recording_info"].items():
                if value:  # Only store non-empty values
                    rec.set_metadata(key, value)
            for key, value in metadata["file_info"].items():
                rec.set_metadata(key, value)

            # Store source file information
            rec.set_metadata("source_file", filepath)

            # Read every signal up front so a mixed per-channel rate is detected
            # before any channel is added: channels of differing native length
            # cannot share the Recording's single time grid, so add_channel would
            # raise mid-load. Each entry: (signal_info, signal_data, channel_type).
            collected: list[tuple[dict, np.ndarray, str]] = []
            for i in range(edf_reader.signals_in_file):
                signal_data, signal_info = self._read_signal_data(edf_reader, i)
                channel_type = self._determine_channel_type(
                    signal_info["label"], signal_info["transducer"]
                )
                collected.append((signal_info, signal_data, channel_type))

            rates = {info["sample_frequency"] for info, _, _ in collected}
            target_rate: float | None = None
            n_out = 0
            if len(rates) > 1:
                if mixed_rate == "error":
                    raise ValueError(
                        "EDF/BDF recording has mixed per-channel sampling rates "
                        f"({sorted(rates)} Hz); biosigIO stores one uniform grid. This "
                        "is a real BIDS montage (e.g. polysomnography: EEG ~200 Hz + "
                        'SpO2/respiration ~12.5 Hz). Pass mixed_rate="resample" to '
                        "upsample the slower channels to the fastest rate for a single-"
                        "grid serving copy (lossy: a derived view, not the faithful "
                        'source), or read channels[ch]["sample_frequency"] per channel '
                        "to handle the rates yourself."
                    )
                # resample: lift every slower channel onto the fastest channel's grid.
                target_rate = max(rates)
                n_out = max(len(data) for _, data, _ in collected)

            # Add channels (resampling the slow ones first when mixed_rate="resample").
            for signal_info, signal_data, channel_type in collected:
                native = float(signal_info["sample_frequency"])
                freq = native
                resampled = target_rate is not None and native != target_rate
                if resampled:
                    signal_data = _resample_to_length(signal_data, n_out)
                    freq = float(target_rate)  # type: ignore[arg-type]  # non-None when resampled

                rec.add_channel(
                    label=signal_info["label"],
                    data=signal_data,
                    sample_frequency=freq,
                    physical_dimension=signal_info["physical_dimension"],
                    prefilter=signal_info["prefilter"],
                    channel_type=channel_type,
                )

                # Store additional channel-specific metadata
                channel_metadata = {
                    "physical_min": signal_info["physical_min"],
                    "physical_max": signal_info["physical_max"],
                    "digital_min": signal_info["digital_min"],
                    "digital_max": signal_info["digital_max"],
                    "transducer": signal_info["transducer"],
                }
                if resampled:
                    # Preserve the true acquisition rate (metadata loss is data loss).
                    channel_metadata["original_sample_frequency"] = native
                rec.channels[signal_info["label"]].update(channel_metadata)

            if target_rate is not None:
                # Flag the recording as a resampled derived view so downstream
                # consumers (and humans) know the grid is fabricated, not native.
                rec.set_metadata("mixed_rate_resampled", True)
                rec.set_metadata("mixed_rate_target_hz", float(target_rate))

            # Read EDF+/BDF+ annotations into events so they survive the
            # import->export->import round-trip (issue #47). Assigned directly
            # (not via add_event) for a single sort; _read_annotations already
            # returns the same schema add_event produces (float64 onset/duration,
            # sorted by onset). Left as the empty __init__ frame when none exist.
            events = self._read_annotations(edf_reader)
            if not events.empty:
                rec.events = events

            return rec

        except Exception as e:
            raise ValueError(f"Error reading EDF file: {str(e)}") from e

        finally:
            if "edf_reader" in locals():
                edf_reader.close()
