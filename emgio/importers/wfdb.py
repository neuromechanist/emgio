import wfdb
import os
# import numpy as np # Keep commented out until needed
# from typing import List, Dict # Keep commented out until needed
from .base import BaseImporter
from ..core.emg import EMG


class WFDBImporter(BaseImporter):
    """Importer for WFDB format files."""

    def load(self, filepath: str) -> EMG:
        """
        Load EMG data and annotations from WFDB files.

        Assumes annotation files (.atr) share the same base name as the record file
        and are located in the same directory.

        Args:
            filepath: Path to the WFDB header file (.hea) or just the record name.

        Returns:
            EMG: EMG object containing the loaded data and annotations.
        """
        record_name = os.path.splitext(filepath)[0]
        # directory = os.path.dirname(filepath) # Removed as unused for now

        try:
            # Read record data and header
            # physical=True ensures data is in physical units
            record = wfdb.rdrecord(record_name=record_name, sampfrom=0, sampto=None, physical=True)

            # Create EMG object
            emg = EMG()

            # Store metadata from header
            emg.set_metadata('source_file', filepath)
            emg.set_metadata('record_name', record.record_name)
            emg.set_metadata('sampling_frequency', record.fs)  # Store overall sampling frequency
            if record.base_date and record.base_time:
                emg.set_metadata('startdate', record.base_date)
                emg.set_metadata('starttime', record.base_time)
            if record.comments:
                emg.set_metadata('comments', '\n'.join(record.comments))

            # Add channels
            for i, sig_name in enumerate(record.sig_name):
                emg.add_channel(
                    label=sig_name,
                    data=record.p_signal[:, i],
                    sample_frequency=record.fs,  # Use record's fs, assuming uniform sampling
                    physical_dimension=record.units[i] if record.units and i < len(record.units) else 'n/a',
                    # WFDB doesn't store prefilter info directly in standard header fields
                    prefilter='n/a',
                    # Determine channel type based on label/units if possible, default to EMG
                    channel_type='EMG' if 'EMG' in sig_name.upper() else 'OTHER'
                )
                # Store additional WFDB-specific metadata if needed
                channel_metadata = {
                    'baseline': record.baseline[i] if record.baseline and i < len(record.baseline) else None,
                    'adc_gain': record.adc_gain[i] if record.adc_gain and i < len(record.adc_gain) else None,
                    'adc_zero': record.adc_zero[i] if record.adc_zero and i < len(record.adc_zero) else None,
                    'init_value': record.init_value[i] if record.init_value and i < len(record.init_value) else None,
                    'checksum': record.checksum[i] if record.checksum and i < len(record.checksum) else None,
                }
                # Filter out None values before updating
                filtered_metadata = {k: v for k, v in channel_metadata.items() if v is not None}
                if filtered_metadata:  # Only update if there is metadata to add
                    emg.channels[sig_name].update(filtered_metadata)

            # Read annotations if available (look for .atr file by default)
            # Note: wfdb-python automatically searches common annotation extensions
            try:
                annotation = wfdb.rdann(record_name=record_name, extension='atr', sampfrom=0, sampto=None)

                # Add annotations to the EMG object
                # Assuming emg object has an `add_event` or `add_annotation` method
                # The structure (onset, duration, description) is common
                if hasattr(emg, 'add_event') and callable(emg.add_event):
                    # Map WFDB annotations to events
                    # Use annotation symbols as descriptions, sample indices as onsets
                    for i, symbol in enumerate(annotation.symbol):
                        onset_sec = annotation.sample[i] / record.fs
                        description = f"WFDB Annotation: {symbol}"
                        # WFDB annotations are typically point events (zero duration)
                        duration_sec = 0
                        emg.add_event(onset=onset_sec, duration=duration_sec, description=description)

                else:
                    # Fallback: Store raw annotation data in metadata if no specific method exists
                    emg.set_metadata('wfdb_annotations', {
                        'sample': annotation.sample.tolist(),
                        'symbol': annotation.symbol,
                        'subtype': annotation.subtype.tolist(),
                        'chan': annotation.chan.tolist(),
                        'num': annotation.num.tolist(),
                        'aux_note': annotation.aux_note,
                        'fs': annotation.fs
                    })

            except FileNotFoundError:
                # Annotation file not found, proceed without annotations
                emg.set_metadata('annotation_status', 'Annotation file (.atr) not found.')
            except Exception as e:
                # Handle other potential errors during annotation reading
                emg.set_metadata('annotation_error', f"Error reading annotations: {str(e)}")

            return emg

        except FileNotFoundError:
            raise FileNotFoundError(f"WFDB record not found: {record_name}. Ensure .hea and data files exist.")
        except Exception as e:
            raise ValueError(f"Error reading WFDB file '{filepath}': {str(e)}")
