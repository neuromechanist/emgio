import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..core.emg import Recording
from ..core.modality import infer_modality_from_channel_type
from ..exceptions import classify_read_error
from .base import BaseImporter


class EEGLABImporter(BaseImporter):
    """Importer for EEGLAB .set files containing EMG data."""

    def _extract_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract metadata from EEGLAB .set file.

        Args:
            data: Dictionary containing EEGLAB .set file data

        Returns:
            dict: Dictionary containing extracted metadata
        """
        metadata = {}

        # Extract basic recording information
        if "setname" in data:
            metadata["setname"] = str(data["setname"][0]) if data["setname"].size > 0 else ""

        if "filename" in data:
            metadata["filename"] = str(data["filename"][0]) if data["filename"].size > 0 else ""

        if "filepath" in data:
            metadata["filepath"] = str(data["filepath"][0]) if data["filepath"].size > 0 else ""

        # Extract subject information
        if "subject" in data:
            metadata["subject"] = str(data["subject"][0]) if data["subject"].size > 0 else ""

        if "group" in data:
            metadata["group"] = str(data["group"][0]) if data["group"].size > 0 else ""

        if "condition" in data:
            metadata["condition"] = str(data["condition"][0]) if data["condition"].size > 0 else ""

        if "session" in data:
            metadata["session"] = str(data["session"][0]) if data["session"].size > 0 else ""

        if "comments" in data:
            metadata["comments"] = str(data["comments"][0]) if data["comments"].size > 0 else ""

        # Extract recording parameters
        if "srate" in data:
            metadata["srate"] = float(data["srate"][0][0]) if data["srate"].size > 0 else 0

        if "nbchan" in data:
            metadata["nbchan"] = int(data["nbchan"][0][0]) if data["nbchan"].size > 0 else 0

        if "trials" in data:
            metadata["trials"] = int(data["trials"][0][0]) if data["trials"].size > 0 else 0

        if "pnts" in data:
            metadata["pnts"] = int(data["pnts"][0][0]) if data["pnts"].size > 0 else 0

        if "xmin" in data and data["xmin"].size > 0:
            metadata["xmin"] = float(data["xmin"][0][0])

        if "xmax" in data and data["xmax"].size > 0:
            metadata["xmax"] = float(data["xmax"][0][0])

        # Add device information
        metadata["device"] = "EEGLAB"

        return metadata

    def _determine_channel_type(self, channel_info: dict[str, Any]) -> str:
        """
        Determine channel type based on channel information.

        Args:
            channel_info: Dictionary containing channel information

        Returns:
            str: Channel type ('EMG', 'ACC', 'GYRO', etc.)
        """
        # Check if type is explicitly specified. After _process_channel_info,
        # 'type' and 'label' are plain strings (scalar-flattened), so consume them
        # as strings rather than indexing/.size like the old array form.
        ch_type = channel_info.get("type")
        if ch_type:
            ch_type_upper = ch_type.upper()
            # Neural/physiological BIDS types pass straight through.
            if ch_type_upper in ("EEG", "EMG", "ECG", "EKG", "EOG", "SEEG", "ECOG"):
                return ch_type_upper
            elif ch_type_upper in ["ACC", "ACCELEROMETER"]:
                return "ACC"
            elif ch_type_upper in ["GYRO", "GYROSCOPE"]:
                return "GYRO"
            elif ch_type_upper in ["TRIG", "TRIGGER"]:
                return "TRIG"

        # If type is not specified or not recognized, try to determine from label.
        label = channel_info.get("label", "")
        if label:
            label_upper = label.upper()
            if "EMG" in label_upper:
                return "EMG"
            elif "ECG" in label_upper or "EKG" in label_upper:
                return "ECG"
            elif "EOG" in label_upper:
                return "EOG"
            elif "ACC" in label_upper:
                return "ACC"
            elif "GYRO" in label_upper:
                return "GYRO"
            elif "TRIG" in label_upper:
                return "TRIG"

        # No reliable type info: do NOT assume EMG (avoids modality creep onto
        # EEG/other data). Caller-supplied BIDS channels.tsv types can refine this.
        return "OTHER"

    def _process_channel_info(self, chanlocs: np.ndarray) -> list[dict[str, Any]]:
        """
        Process channel location information.

        Args:
            chanlocs: Array containing channel location information

        Returns:
            list: List of dictionaries containing channel information
        """
        channel_info_list = []

        field_names = chanlocs.dtype.names
        if field_names is None:
            return channel_info_list

        # EEGLAB saves chanlocs as a struct array whose MATLAB shape survives
        # loadmat: (1, N) row form OR (N, 1) column form (real exports use both).
        # Indexing chanlocs[0] assumed the row form and saw exactly ONE channel
        # on column-form files, silently truncating every later channel; ravel
        # makes the walk shape-agnostic.
        for element in np.asarray(chanlocs).ravel():
            channel_info = {}

            # Extract channel fields
            for field in field_names:
                # Get the field value for this channel
                field_value = np.asarray(element[field])
                if field_value.size == 0:
                    continue
                # scipy.io.loadmat returns scalar struct fields as nested
                # (1, 1)-shaped arrays, so field_value[0] is still 1-D and
                # float()/str() on it fails or mis-formats. Flatten to a scalar.
                scalar = field_value.ravel()[0]

                # Process based on field name
                if field == "labels":
                    channel_info["label"] = str(scalar)
                elif field == "type":
                    channel_info["type"] = str(scalar)
                elif field == "X":
                    channel_info["X"] = float(scalar)
                elif field == "Y":
                    channel_info["Y"] = float(scalar)
                elif field == "Z":
                    channel_info["Z"] = float(scalar)

            # Determine channel type
            channel_info["channel_type"] = self._determine_channel_type(channel_info)

            # Add to list
            channel_info_list.append(channel_info)

        return channel_info_list

    def _process_events(self, events: np.ndarray) -> list[dict[str, Any]]:
        """
        Process event information.

        Args:
            events: Array containing event information

        Returns:
            list: List of dictionaries containing event information
        """
        event_list = []

        # Check if events exist
        if events.size == 0:
            return event_list

        field_names = events.dtype.names
        if field_names is None:
            return event_list

        # Same row-vs-column shape hazard as chanlocs: walk the raveled struct
        # array and flatten each field to a scalar instead of assuming (1, 1)
        # nesting.
        for element in np.asarray(events).ravel():
            event_info = {}

            # Extract event fields
            for field in field_names:
                # Get the field value for this event
                field_value = np.asarray(element[field])
                if field_value.size == 0:
                    continue
                scalar = field_value.ravel()[0]

                # Process based on field name
                if field == "latency":
                    event_info["latency"] = float(scalar)
                elif field == "type":
                    event_info["type"] = str(scalar)
                elif field == "duration":
                    event_info["duration"] = float(scalar)

            # Add to list if it has required fields
            if "latency" in event_info and "type" in event_info:
                event_list.append(event_info)

        return event_list

    def load(self, filepath: str) -> Recording:
        """
        Load EMG data from EEGLAB .set file.

        Args:
            filepath: Path to the EEGLAB .set file

        Returns:
            Recording: Recording object containing the loaded data
        """
        return self._load(filepath)

    @staticmethod
    def _normalize_eeglab_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Flatten the two on-disk EEGLAB ``.set`` save forms to one flat dict.

        A real EEGLAB export saves the whole dataset as a single MATLAB struct
        variable named ``EEG``, so ``scipy.io.loadmat`` returns
        ``{'EEG': <(1, 1) struct>}`` and every field (``nbchan``, ``data``,
        ``chanlocs``, ``event``, ...) is nested one level under ``data['EEG']``.
        Some files (and the synthetic test fixtures) instead store those fields at
        the top level of the ``.mat`` dict. ``_extract_metadata`` and the
        signal/channel/event reads in :meth:`_load` expect top-level fields, so
        this normalizes the nested form to the flat form before either runs and
        passes the flat form through unchanged.

        For a struct *array* (multiple datasets, shape ``(1, N)``) only the first
        dataset is used; the rest is currently unsupported. Any non-``EEG``
        top-level keys (e.g. loadmat's ``__header__``) are preserved so nothing is
        lost.

        Args:
            data: The raw mapping returned by ``scipy.io.loadmat``.

        Returns:
            A flat dict whose EEGLAB fields are at the top level.
        """
        eeg = data.get("EEG")
        if eeg is None:
            return data
        eeg = np.asarray(eeg)
        # Only unwrap a MATLAB struct (it exposes named fields via dtype.names);
        # a flat-form dict has no "EEG" key, or an "EEG" that is not a struct.
        names = eeg.dtype.names
        if names is None or eeg.size == 0:
            return data

        # A struct array (shape (1, N)) holds several datasets; take the first.
        record = eeg.ravel()[0]

        flat: dict[str, Any] = {name: record[name] for name in names}
        # Keep any non-EEG top-level keys (loadmat's __header__/__version__, or a
        # second variable saved alongside) without letting them clobber EEG fields.
        for key, value in data.items():
            if key != "EEG" and key not in flat:
                flat[key] = value
        return flat

    @staticmethod
    def _read_fdt(
        set_filepath: str, data_field: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """Load the signal matrix from a sibling EEGLAB ``.fdt`` float32 file.

        EEGLAB writes the ``[nbchan x pnts x trials]`` matrix to a separate
        ``.fdt`` in MATLAB column-major order when the data is large; ``EEG.data``
        then stores only the ``.fdt``'s original (pre-BIDS-rename) filename. The
        sibling ``.fdt`` next to the ``.set`` is resolved first because BIDS
        renames the file on disk but not the embedded reference, with the
        embedded name as a fallback.

        Args:
            set_filepath: Path to the ``.set`` file being loaded.
            data_field: The ``EEG.data`` char array holding the ``.fdt`` name.
            metadata: Extracted header metadata (needs ``nbchan``/``pnts``).

        Returns:
            The signal matrix as a ``(nbchan, pnts * trials)`` float32 array.
        """
        nbchan = int(metadata.get("nbchan", 0))
        pnts = int(metadata.get("pnts", 0))
        trials = int(metadata.get("trials", 1)) or 1
        if nbchan <= 0 or pnts <= 0:
            raise ValueError(
                "EEGLAB data is in a separate .fdt file but the .set header is "
                "missing nbchan/pnts needed to reshape it"
            )
        directory = os.path.dirname(set_filepath)
        sibling = os.path.splitext(set_filepath)[0] + ".fdt"
        embedded = "".join(np.atleast_1d(data_field).ravel().astype(str)).strip()
        candidates = [sibling]
        if embedded:
            candidates.append(os.path.join(directory, os.path.basename(embedded)))
        fdt_path = next((p for p in candidates if os.path.isfile(p)), None)
        if fdt_path is None:
            raise FileNotFoundError(
                f"EEGLAB data is in a separate .fdt file but none was found "
                f"(tried sibling {sibling!r} and embedded name {embedded!r})"
            )
        raw = np.fromfile(fdt_path, dtype="<f4")
        expected = nbchan * pnts * trials
        if raw.size != expected:
            raise ValueError(
                f"{fdt_path} holds {raw.size} float32 samples but the .set header "
                f"implies {expected} (nbchan {nbchan} x pnts {pnts} x trials {trials})"
            )
        # MATLAB stores column-major, so the matrix is (nbchan, samples).
        return raw.reshape((nbchan, pnts * trials), order="F")

    def _load(self, filepath: str) -> Recording:
        """Internal loader (see :meth:`load`)."""
        try:
            # Load the .set file. A real EEGLAB export wraps every field in a
            # single ``EEG`` struct, so unwrap it to the flat top-level form that
            # the metadata/signal/channel/event reads below expect (a no-op for
            # files already saved flat).
            data = self._normalize_eeglab_dict(loadmat(filepath))

            # Create Recording object
            rec = Recording()

            # Extract and store metadata
            metadata = self._extract_metadata(data)
            for key, value in metadata.items():
                rec.set_metadata(key, value)

            # Store source file information
            rec.set_metadata("source_file", filepath)

            # Sampling rate (used for the event onset/duration conversion below and
            # for the signal time index).
            # Fall back to 1000 Hz when srate is absent OR present-but-zero (an
            # empty srate field), so the event/time-index divisions never hit 0.
            srate = float(metadata.get("srate", 1000)) or 1000.0

            # Process channel information
            if "chanlocs" in data and data["chanlocs"].size > 0:
                channel_info_list = self._process_channel_info(data["chanlocs"])
            else:
                # If no channel locations, create default channel info
                channel_info_list = []
                for i in range(metadata.get("nbchan", 0)):
                    channel_info_list.append({"label": f"Channel{i + 1}", "channel_type": "OTHER"})

            # Process event information into the standard events table. EEGLAB
            # stores event latency/duration in samples (latency is 1-based), so
            # convert to seconds; the event ``type`` becomes the description.
            if "event" in data and data["event"].size > 0:
                for ev in self._process_events(data["event"]):
                    rec.add_event(
                        onset=(ev["latency"] - 1) / srate,
                        duration=ev.get("duration", 0.0) / srate,
                        description=ev["type"],
                    )

            # Extract signal data
            if "data" in data and data["data"].size > 0:
                # Get data array. EEGLAB stores large recordings with the signal
                # matrix in a separate float32 ``.fdt`` file; in that case
                # ``EEG.data`` holds the ``.fdt`` filename (a char array) rather
                # than the numeric matrix, so load the sibling ``.fdt`` instead.
                signal_data = data["data"]
                if signal_data.dtype.kind in ("U", "S"):
                    signal_data = self._read_fdt(filepath, signal_data, metadata)

                # Derive the time index in seconds from the sample count. EEGLAB's
                # `times` field is in milliseconds, so dividing it by srate (the old
                # behavior) mis-scaled the index; sample_index / srate is correct and
                # always matches the data length.
                time_index = np.arange(signal_data.shape[1]) / srate

                # The data matrix is authoritative for the channel count: every
                # row ships to the caller. chanlocs only NAMES the rows, so a
                # short/missing chanlocs (or one misread from an odd save form)
                # must never truncate the signal -- pad with default labels
                # instead. A longer-than-data chanlocs is trimmed (labels
                # without signal rows describe nothing).
                n_data_ch = signal_data.shape[0]
                if len(channel_info_list) != n_data_ch:
                    warnings.warn(
                        f"EEGLAB chanlocs describes {len(channel_info_list)} "
                        f"channel(s) but the data matrix has {n_data_ch} row(s); "
                        f"keeping all data rows",
                        stacklevel=3,
                    )
                    channel_info_list = channel_info_list[:n_data_ch]
                    for i in range(len(channel_info_list), n_data_ch):
                        channel_info_list.append(
                            {"label": f"Channel{i + 1}", "channel_type": "OTHER"}
                        )
                header_nbchan = int(metadata.get("nbchan", 0))
                if header_nbchan and header_nbchan != n_data_ch:
                    warnings.warn(
                        f"EEGLAB header nbchan={header_nbchan} disagrees with the "
                        f"data matrix ({n_data_ch} rows); using the data matrix",
                        stacklevel=3,
                    )

                # Labels must be unique: rec.channels is keyed by label and the
                # exporters slice rec.signals by column name, so a duplicate
                # (a real channel literally named "Channel4" colliding with a
                # padded default, or duplicated labels in chanlocs itself)
                # silently clobbers channel metadata and breaks per-channel
                # export slicing. Disambiguate with a numeric suffix and warn.
                used_labels: set[str] = set()
                for i, channel_info in enumerate(channel_info_list):
                    label = channel_info.get("label", f"Channel{i + 1}")
                    if label in used_labels:
                        k = 2
                        while f"{label}_{k}" in used_labels:
                            k += 1
                        warnings.warn(
                            f"duplicate EEGLAB channel label {label!r}; renaming to {label}_{k}",
                            stacklevel=3,
                        )
                        label = f"{label}_{k}"
                    used_labels.add(label)
                    channel_info["label"] = label

                # Build the frame in a single allocation (samples x channels)
                # rather than assigning one column at a time. The per-column path
                # reallocates the block manager O(n_channels) times and roughly
                # doubles peak RAM, which OOMs large / high-channel-count .fdt
                # recordings (#66, #95); the .fdt's native float32 is preserved so
                # a multi-GB recording is not silently upcast to float64.
                labels = [info["label"] for info in channel_info_list]
                rec.signals = pd.DataFrame(
                    signal_data.T, columns=labels, index=time_index, copy=False
                )
                del signal_data

                # Add channel information (the pad/trim above guarantees
                # channel_info_list has exactly n_data_ch entries with unique labels)
                for channel_info in channel_info_list:
                    channel_label = channel_info["label"]

                    # Add channel info (no silent EMG default; carry modality)
                    ch_type = channel_info.get("channel_type", "OTHER")
                    rec.channels[channel_label] = {
                        "sample_frequency": srate,
                        "physical_dimension": "uV",  # Default unit for EEG/EMG
                        "prefilter": "n/a",
                        "channel_type": ch_type,
                        "modality": infer_modality_from_channel_type(ch_type),
                    }

                    # Add additional channel metadata
                    if "X" in channel_info:
                        rec.channels[channel_label]["X"] = channel_info["X"]
                    if "Y" in channel_info:
                        rec.channels[channel_label]["Y"] = channel_info["Y"]
                    if "Z" in channel_info:
                        rec.channels[channel_label]["Z"] = channel_info["Z"]

            return rec

        except Exception as e:
            raise classify_read_error(e, filepath) from e
