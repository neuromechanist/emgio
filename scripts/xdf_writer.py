"""
Simple XDF file writer for creating test files.

Based on the XDF specification: https://github.com/sccn/xdf/wiki/Specifications
"""

import struct
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np


class XDFWriter:
    """Write XDF files following the XDF 1.0 specification."""

    # Chunk tags
    TAG_FILE_HEADER = 1
    TAG_STREAM_HEADER = 2
    TAG_SAMPLES = 3
    TAG_CLOCK_OFFSET = 4
    TAG_BOUNDARY = 5
    TAG_STREAM_FOOTER = 6

    # Format codes
    FORMAT_MAP = {
        "float32": ("f", 4),
        "double64": ("d", 8),
        "int8": ("b", 1),
        "int16": ("h", 2),
        "int32": ("i", 4),
        "int64": ("q", 8),
        "string": ("s", 0),  # Variable length
    }

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.file = None
        self.stream_formats = {}  # stream_id -> (format_code, bytes_per_value)

    def __enter__(self):
        self.file = open(self.filepath, "wb")
        self._write_magic()
        self._write_file_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def _write_magic(self):
        """Write the XDF magic code."""
        self.file.write(b"XDF:")

    def _write_varlen_int(self, value: int):
        """Write a variable-length integer."""
        if value < 256:
            self.file.write(struct.pack("<B", 1))  # 1 byte for length
            self.file.write(struct.pack("<B", value))
        elif value < 65536:
            self.file.write(struct.pack("<B", 4))  # 4 bytes for length
            self.file.write(struct.pack("<I", value))
        else:
            self.file.write(struct.pack("<B", 8))  # 8 bytes for length
            self.file.write(struct.pack("<Q", value))

    def _write_chunk(self, tag: int, content: bytes, stream_id: int | None = None):
        """Write a chunk with optional stream ID."""
        # Calculate total length: tag (2 bytes) + optional stream_id (4 bytes) + content
        length = 2 + len(content)
        if stream_id is not None:
            length += 4

        # Write length
        self._write_varlen_int(length)

        # Write tag
        self.file.write(struct.pack("<H", tag))

        # Write stream ID if present
        if stream_id is not None:
            self.file.write(struct.pack("<I", stream_id))

        # Write content
        self.file.write(content)

    def _write_file_header(self):
        """Write the file header chunk."""
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
        content = f'<?xml version="1.0"?>\n<info>\n  <version>1.0</version>\n  <datetime>{now}</datetime>\n</info>'
        self._write_chunk(self.TAG_FILE_HEADER, content.encode("utf-8"))

    def write_stream_header(
        self,
        stream_id: int,
        name: str,
        stream_type: str,
        channel_count: int,
        nominal_srate: float,
        channel_format: str,
        source_id: str = "",
        channel_labels: list[str] | None = None,
    ):
        """Write a stream header chunk."""
        # Store format info for later sample writing
        if channel_format in self.FORMAT_MAP:
            self.stream_formats[stream_id] = self.FORMAT_MAP[channel_format]
        else:
            self.stream_formats[stream_id] = self.FORMAT_MAP["float32"]

        # Build XML header
        xml_parts = [
            '<?xml version="1.0"?>',
            "<info>",
            f"  <name>{name}</name>",
            f"  <type>{stream_type}</type>",
            f"  <channel_count>{channel_count}</channel_count>",
            f"  <nominal_srate>{nominal_srate}</nominal_srate>",
            f"  <channel_format>{channel_format}</channel_format>",
            f"  <source_id>{source_id}</source_id>",
            "  <version>1.0</version>",
        ]

        if channel_labels:
            xml_parts.append("  <desc>")
            xml_parts.append("    <channels>")
            for label in channel_labels:
                xml_parts.append(f"      <channel><label>{label}</label></channel>")
            xml_parts.append("    </channels>")
            xml_parts.append("  </desc>")

        xml_parts.append("</info>")
        content = "\n".join(xml_parts)

        self._write_chunk(self.TAG_STREAM_HEADER, content.encode("utf-8"), stream_id)

    def _write_varlen_int_to_buf(self, buf: BytesIO, value: int):
        """Write a variable-length integer to a buffer."""
        if value < 256:
            buf.write(struct.pack("<B", 1))
            buf.write(struct.pack("<B", value))
        elif value < 65536:
            buf.write(struct.pack("<B", 4))
            buf.write(struct.pack("<I", value))
        else:
            buf.write(struct.pack("<B", 8))
            buf.write(struct.pack("<Q", value))

    def write_samples(
        self,
        stream_id: int,
        timestamps: np.ndarray,
        data: np.ndarray,
    ):
        """Write sample data for a stream."""
        format_code, bytes_per_value = self.stream_formats.get(stream_id, ("f", 4))

        n_samples = len(timestamps)
        if data.ndim == 1:
            n_channels = 1
            data = data.reshape(-1, 1)
        else:
            n_channels = data.shape[1]

        # Build the samples chunk content
        buf = BytesIO()

        # Write NumSamples as variable-length integer (required by XDF spec)
        self._write_varlen_int_to_buf(buf, n_samples)

        for i in range(n_samples):
            # Write timestamp (8 bytes if present, 0 if not)
            ts = timestamps[i]
            if ts != 0:
                buf.write(struct.pack("<B", 8))  # 8 bytes for timestamp
                buf.write(struct.pack("<d", ts))  # Timestamp as double
            else:
                buf.write(struct.pack("<B", 0))  # No timestamp

            # Write sample values
            for j in range(n_channels):
                buf.write(struct.pack(f"<{format_code}", data[i, j]))

        self._write_chunk(self.TAG_SAMPLES, buf.getvalue(), stream_id)

    def write_string_samples(
        self,
        stream_id: int,
        timestamps: np.ndarray,
        data: list[list[str]],
    ):
        """Write string (marker) samples for a stream."""
        n_samples = len(timestamps)
        buf = BytesIO()

        # Write NumSamples as variable-length integer
        self._write_varlen_int_to_buf(buf, n_samples)

        for i, ts in enumerate(timestamps):
            # Write timestamp
            if ts != 0:
                buf.write(struct.pack("<B", 8))
                buf.write(struct.pack("<d", ts))
            else:
                buf.write(struct.pack("<B", 0))

            # Write string values (length-prefixed)
            marker = data[i][0] if isinstance(data[i], list) else data[i]
            marker_bytes = marker.encode("utf-8")
            # Variable length integer for string length
            self._write_varlen_int_to_buf(buf, len(marker_bytes))
            buf.write(marker_bytes)

        self._write_chunk(self.TAG_SAMPLES, buf.getvalue(), stream_id)

    def write_stream_footer(
        self, stream_id: int, first_timestamp: float, last_timestamp: float, sample_count: int
    ):
        """Write a stream footer chunk."""
        content = f"""<?xml version="1.0"?>
<info>
  <first_timestamp>{first_timestamp}</first_timestamp>
  <last_timestamp>{last_timestamp}</last_timestamp>
  <sample_count>{sample_count}</sample_count>
</info>"""
        self._write_chunk(self.TAG_STREAM_FOOTER, content.encode("utf-8"), stream_id)


def create_multistream_test_xdf(output_path: str, duration: float = 5.0):
    """Create a multi-stream XDF test file with synthetic data."""

    output_path = Path(output_path)

    # Generate synthetic data
    np.random.seed(42)

    # Stream 1: EEG-like data (8 channels, 256 Hz)
    eeg_srate = 256
    eeg_n_samples = int(duration * eeg_srate)
    eeg_timestamps = np.arange(eeg_n_samples) / eeg_srate
    eeg_data = np.random.randn(eeg_n_samples, 8).astype(np.float32) * 50  # microvolts
    eeg_labels = [f"EEG{i + 1}" for i in range(8)]

    # Stream 2: EMG-like data (2 channels, 2048 Hz)
    emg_srate = 2048
    emg_n_samples = int(duration * emg_srate)
    emg_timestamps = np.arange(emg_n_samples) / emg_srate
    emg_data = np.random.randn(emg_n_samples, 2).astype(np.float32) * 100  # microvolts
    emg_labels = ["EMG_L", "EMG_R"]

    # Stream 3: Mocap-like data (6 channels - 2 markers x 3D, 120 Hz)
    mocap_srate = 120
    mocap_n_samples = int(duration * mocap_srate)
    mocap_timestamps = np.arange(mocap_n_samples) / mocap_srate
    mocap_data = np.random.randn(mocap_n_samples, 6).astype(np.float32)
    mocap_labels = ["Marker1_X", "Marker1_Y", "Marker1_Z", "Marker2_X", "Marker2_Y", "Marker2_Z"]

    # Stream 4: Markers (irregular)
    marker_times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    marker_times = marker_times[marker_times < duration]
    marker_data = [[f"Event_{i + 1}"] for i in range(len(marker_times))]

    with XDFWriter(output_path) as writer:
        # Write stream headers
        writer.write_stream_header(
            stream_id=1,
            name="TestEEG",
            stream_type="EEG",
            channel_count=8,
            nominal_srate=eeg_srate,
            channel_format="float32",
            source_id="emgio_test_eeg",
            channel_labels=eeg_labels,
        )

        writer.write_stream_header(
            stream_id=2,
            name="TestEMG",
            stream_type="EMG",
            channel_count=2,
            nominal_srate=emg_srate,
            channel_format="float32",
            source_id="emgio_test_emg",
            channel_labels=emg_labels,
        )

        writer.write_stream_header(
            stream_id=3,
            name="TestMocap",
            stream_type="Mocap",
            channel_count=6,
            nominal_srate=mocap_srate,
            channel_format="float32",
            source_id="emgio_test_mocap",
            channel_labels=mocap_labels,
        )

        writer.write_stream_header(
            stream_id=4,
            name="TestMarkers",
            stream_type="Markers",
            channel_count=1,
            nominal_srate=0,  # Irregular
            channel_format="string",
            source_id="emgio_test_markers",
        )

        # Write sample data
        writer.write_samples(1, eeg_timestamps, eeg_data)
        writer.write_samples(2, emg_timestamps, emg_data)
        writer.write_samples(3, mocap_timestamps, mocap_data)
        writer.write_string_samples(4, marker_times, marker_data)

        # Write stream footers
        writer.write_stream_footer(1, eeg_timestamps[0], eeg_timestamps[-1], len(eeg_timestamps))
        writer.write_stream_footer(2, emg_timestamps[0], emg_timestamps[-1], len(emg_timestamps))
        writer.write_stream_footer(
            3, mocap_timestamps[0], mocap_timestamps[-1], len(mocap_timestamps)
        )
        writer.write_stream_footer(4, marker_times[0], marker_times[-1], len(marker_times))

    print(f"Created multi-stream XDF: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


if __name__ == "__main__":
    # Create the test file in examples directory (relative to script location)
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "examples" / "multi_stream_test.xdf"
    create_multistream_test_xdf(output_path, duration=5.0)

    # Verify with pyxdf
    print("\nVerifying with pyxdf...")
    import pyxdf

    data, header = pyxdf.load_xdf(str(output_path))
    print(f"Loaded {len(data)} streams")
    for stream in data:
        info = stream["info"]
        name = info["name"][0]
        stype = info["type"][0]
        print(
            f"  {name} ({stype}): {stream['time_series'].shape if hasattr(stream['time_series'], 'shape') else len(stream['time_series'])} samples"
        )
