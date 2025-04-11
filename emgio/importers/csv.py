import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base import BaseImporter
from ..core.emg import EMG


class CSVImporter(BaseImporter):
    """
    General purpose CSV importer for EMG data.

    This importer can handle various CSV formats with columnar data, auto-detect
    headers, time columns, and allow for specific column selection.
    """

    def load(self, filepath: str, **kwargs) -> EMG:
        """
        Load EMG data from a CSV file.

        Args:
            filepath: Path to the CSV file
            **kwargs: Additional options including:
                - columns: List of column names or indices to include
                - time_column: Name or index of column to use as time index (default: auto-detect)
                - has_header: Whether file has a header row (default: auto-detect)
                - skiprows: Number of rows to skip at the beginning (default: auto-detect)
                - delimiter: Column delimiter (default: auto-detect)
                - sample_frequency: Sampling frequency in Hz (required if no time column)
                - channel_types: Dict mapping column names to channel types ('EMG', 'ACC', etc.)
                - physical_dimensions: Dict mapping column names to physical dimensions
                - metadata: Dict of additional metadata to include

        Returns:
            EMG: EMG object containing the loaded data
        """
        # Extract kwargs with defaults
        columns = kwargs.get('columns', None)
        time_column = kwargs.get('time_column', None)
        has_header = kwargs.get('has_header', None)
        skiprows = kwargs.get('skiprows', None)
        delimiter = kwargs.get('delimiter', None)
        sample_frequency = kwargs.get('sample_frequency', None)
        channel_types = kwargs.get('channel_types', {})
        physical_dimensions = kwargs.get('physical_dimensions', {})
        metadata = kwargs.get('metadata', {})

        # Analyze file structure if parameters not explicitly provided
        if any(param is None for param in [has_header, skiprows, delimiter]):
            analyzed_params = self._analyze_csv_structure(filepath)

            # Use analyzed parameters for any not explicitly provided
            has_header = has_header if has_header is not None else analyzed_params['has_header']
            skiprows = skiprows if skiprows is not None else analyzed_params['skiprows']
            delimiter = delimiter if delimiter is not None else analyzed_params['delimiter']

        # Read the CSV file
        try:
            df = pd.read_csv(
                filepath,
                header=0 if has_header else None,
                skiprows=skiprows,
                delimiter=delimiter,
                index_col=None
            )
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {str(e)}")

        # If no header, generate column names
        if not has_header:
            df.columns = [f"Channel_{i}" for i in range(len(df.columns))]

        # Filter columns if specified
        if columns is not None:
            if all(isinstance(col, int) for col in columns):
                # Convert numerical indices to column names
                col_names = [df.columns[i] for i in columns]
                df = df[col_names]
            else:
                # Filter by column names
                df = df[columns]

        # Handle time column
        if time_column is not None:
            # If time_column is an index, convert to column name
            if isinstance(time_column, int):
                time_column = df.columns[time_column]

            # Set time column as index
            if time_column in df.columns:
                df.set_index(time_column, inplace=True)
            else:
                raise ValueError(f"Time column '{time_column}' not found in data")
        else:
            # Try to auto-detect time column
            time_col = self._detect_time_column(df)

            if time_col:
                df.set_index(time_col, inplace=True)
            elif sample_frequency:
                # Create time index based on provided sampling frequency
                time_index = np.arange(len(df)) / sample_frequency
                df.index = time_index
            else:
                # No time column and no sample frequency provided
                raise ValueError(
                    "No time column detected and no sample_frequency provided. "
                    "Please specify either time_column or sample_frequency."
                )

        # Create EMG object
        emg = EMG()

        # Add metadata
        emg.set_metadata('source_file', filepath)
        emg.set_metadata('file_format', 'CSV')

        # Add any user-provided metadata
        for key, value in metadata.items():
            emg.set_metadata(key, value)

        # Default sampling frequency if not specified
        default_sample_frequency = 1000.0  # 1 kHz is a common default for EMG
        if hasattr(df.index, 'to_series'):
            # Calculate sampling frequency from time index if possible
            try:
                time_diffs = df.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    avg_diff = time_diffs.mean()
                    if avg_diff > 0:
                        calculated_freq = 1.0 / avg_diff
                        default_sample_frequency = calculated_freq
            except Exception:
                # If calculation fails, keep default
                pass

        # Add each column as a channel
        for column in df.columns:
            # Determine channel type
            if column in channel_types:
                ch_type = channel_types[column]
            else:
                # Try to infer channel type from name
                ch_type = self._infer_channel_type(column)

            # Determine physical dimension
            if column in physical_dimensions:
                phys_dim = physical_dimensions[column]
            else:
                # Default based on channel type
                phys_dim = self._default_physical_dimension(ch_type)

            # Add the channel to the EMG object
            emg.add_channel(
                label=column,
                data=df[column].values,
                sample_frequency=sample_frequency or default_sample_frequency,
                physical_dimension=phys_dim,
                channel_type=ch_type
            )

        # Encourage user to add metadata if missing essential information
        self._print_metadata_reminder(emg)

        return emg

    def _analyze_csv_structure(self, filepath: str) -> Dict:
        """
        Analyze the CSV file structure to detect delimiter, headers, and rows to skip.

        Args:
            filepath: Path to the CSV file

        Returns:
            Dict with detected parameters:
                - delimiter: Detected delimiter character
                - has_header: Whether the file has a header row
                - skiprows: Number of rows to skip
        """
        # Default results
        results = {
            'delimiter': ',',
            'has_header': True,
            'skiprows': 0
        }

        # Read first few lines to analyze
        try:
            with open(filepath, 'r') as f:
                lines = [f.readline().strip() for _ in range(20)]
                lines = [line for line in lines if line]  # Remove empty lines
        except Exception:
            return results

        if not lines:
            return results

        # Detect delimiter
        delimiters = [',', '\t', ';', '|', ' ']
        delimiter_counts = {}

        for delim in delimiters:
            counts = [line.count(delim) for line in lines]
            if all(count > 0 for count in counts[:5]):  # Check first 5 non-empty lines
                avg_count = sum(counts) / len(counts)
                delimiter_counts[delim] = avg_count

        if delimiter_counts:
            results['delimiter'] = max(delimiter_counts, key=delimiter_counts.get)

        # Detect header and rows to skip
        potential_header_line = None
        skiprows = 0

        for i, line in enumerate(lines):
            parts = line.split(results['delimiter'])

            # Check if line might be a header (contains text fields)
            contains_text = any(
                part.strip() and not self._is_numeric(part.strip())
                for part in parts
            )

            if contains_text:
                # Header might be mixing text/numbers, but predominantly text
                text_ratio = sum(1 for p in parts if p.strip() and not self._is_numeric(p.strip())) / len(parts)
                if text_ratio > 0.5:  # More than half are text fields
                    potential_header_line = i
                    break

            skiprows += 1

        # If we found a potential header line
        if potential_header_line is not None:
            results['has_header'] = True
            results['skiprows'] = potential_header_line
        else:
            results['has_header'] = False
            results['skiprows'] = 0

        return results

    def _is_numeric(self, value: str) -> bool:
        """Check if a string value is numeric."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _detect_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Try to detect which column represents time.

        Args:
            df: DataFrame with loaded data

        Returns:
            Name of detected time column or None if not found
        """
        time_keywords = ['time', 'second', 'seconds', 's']

        # Check column names for time keywords
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in time_keywords):
                return col

        # Check if first column is monotonically increasing (typical for time)
        first_col = df.columns[0]
        if len(df) > 1 and pd.Series(df[first_col]).is_monotonic_increasing:
            # Check if the values are plausible time values (e.g., not all integers if diff is small)
            if df[first_col].dtype in [np.float64, np.float32]:
                return first_col
            elif df[first_col].diff().dropna().mean() > 1e-9:  # Avoid treating integer indices as time
                return first_col

        return None

    def _infer_channel_type(self, column_name: str) -> str:
        """
        Infer channel type from column name.

        Args:
            column_name: Name of the column

        Returns:
            Inferred channel type
        """
        name_lower = column_name.lower()

        if any(keyword in name_lower for keyword in ['emg', 'muscle']):
            return 'EMG'
        elif any(keyword in name_lower for keyword in ['acc', 'accel']):
            return 'ACC'
        elif any(keyword in name_lower for keyword in ['gyro']):
            return 'GYRO'
        elif any(keyword in name_lower for keyword in ['time', 'second']):
            return 'TIME'  # Might be redundant if used as index, but useful for metadata
        else:
            return 'OTHER'

    def _default_physical_dimension(self, channel_type: str) -> str:
        """
        Return default physical dimension for a channel type.

        Args:
            channel_type: Type of channel

        Returns:
            Default physical dimension
        """
        dimensions = {
            'EMG': 'µV',
            'ACC': 'g',
            'GYRO': 'deg/s',
            'TIME': 's',
            'OTHER': 'a.u.'
        }
        return dimensions.get(channel_type, 'a.u.')

    def _print_metadata_reminder(self, emg: EMG) -> None:
        """
        Print a reminder to add metadata if essential information is missing.

        Args:
            emg: EMG object to check
        """
        essential_metadata = ['subject', 'device', 'recording_date']
        missing = [meta for meta in essential_metadata if meta not in emg.metadata]

        if missing:
            print("[INFO] Reminder: Consider adding essential metadata for better context:")
            for meta in missing:
                print(f"  emg.set_metadata('{meta}', '<Your {meta.replace('_', ' ').title()}>')")
            print("Example: emg.set_metadata('subject', 'S001')")
