#!/usr/bin/env python3
"""
Create a small multi-stream XDF test file by replaying data from a larger XDF file.

This script:
1. Loads a large multi-stream XDF file
2. Extracts a short time window (5 seconds)
3. Creates LSL streams and pushes the data
4. Uses LabRecorderCLI to record to a new XDF file

Requirements:
- pylsl
- pyxdf
- LabRecorderCLI (built from App-LabRecorder)
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths - configured via environment variables or defaults
# SOURCE_XDF: Path to a large multi-stream XDF file to extract from
SOURCE_XDF = Path(os.environ.get("EMGIO_SOURCE_XDF", ""))
# OUTPUT_XDF: Where to save the test file (default: examples/multi_stream_test.xdf)
OUTPUT_XDF = Path(
    os.environ.get("EMGIO_OUTPUT_XDF", str(PROJECT_ROOT / "examples" / "multi_stream_test.xdf"))
)
# LABRECORDER_CLI: Path to LabRecorderCLI executable
LABRECORDER_CLI = Path(os.environ.get("EMGIO_LABRECORDER_CLI", ""))

# Duration to extract (seconds)
EXTRACT_DURATION = 5.0


def _validate_paths():
    """Validate required paths are configured and exist."""
    errors = []

    if not SOURCE_XDF or not SOURCE_XDF.exists():
        errors.append(
            "SOURCE_XDF not found. Set EMGIO_SOURCE_XDF environment variable to a valid XDF file path."
        )

    if not LABRECORDER_CLI or not LABRECORDER_CLI.exists():
        errors.append(
            "LABRECORDER_CLI not found. Set EMGIO_LABRECORDER_CLI environment variable to LabRecorderCLI path."
        )

    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        print("\nExample:")
        print("  export EMGIO_SOURCE_XDF=/path/to/source.xdf")
        print("  export EMGIO_LABRECORDER_CLI=/path/to/LabRecorderCLI")
        return False

    return True


def main():
    # Validate paths before proceeding
    if not _validate_paths():
        return

    import pyxdf
    from pylsl import StreamInfo, StreamOutlet, local_clock

    print(f"Loading source XDF: {SOURCE_XDF}")
    data, header = pyxdf.load_xdf(str(SOURCE_XDF))
    print(f"Loaded {len(data)} streams")

    # Find when all streams are active
    stream_starts = {}
    for stream in data:
        name = stream["info"]["name"][0]
        ts = stream["time_stamps"]
        if len(ts) > 0:
            stream_starts[name] = ts[0]
            print(f"  {name}: starts at {ts[0]:.2f}")

    # Start from when all streams are active
    start_time = max(stream_starts.values())
    end_time = start_time + EXTRACT_DURATION
    print(f"\nExtracting window: {start_time:.2f} to {end_time:.2f} ({EXTRACT_DURATION}s)")

    # Prepare stream data
    streams_to_replay = []

    for stream in data:
        info = stream["info"]
        name = info["name"][0]
        stype = info["type"][0]
        channel_count = int(info["channel_count"][0])
        nominal_srate = float(info["nominal_srate"][0])
        channel_format = info["channel_format"][0]

        ts = stream["time_stamps"]
        time_series = stream["time_series"]

        # Get data in time window
        if isinstance(ts, np.ndarray) and len(ts) > 0:
            mask = (ts >= start_time) & (ts <= end_time)
            indices = np.where(mask)[0]

            if len(indices) > 0:
                if isinstance(time_series, np.ndarray):
                    extracted_data = time_series[indices]
                    extracted_ts = ts[indices] - start_time  # Relative timestamps

                    # Map XDF format to LSL format
                    lsl_format_map = {
                        "float32": "float32",
                        "double64": "double64",
                        "int32": "int32",
                        "int16": "int16",
                        "int8": "int8",
                        "string": "string",
                    }
                    lsl_format = lsl_format_map.get(channel_format, "float32")

                    streams_to_replay.append(
                        {
                            "name": name,
                            "type": stype,
                            "channel_count": channel_count,
                            "nominal_srate": nominal_srate,
                            "format": lsl_format,
                            "data": extracted_data,
                            "timestamps": extracted_ts,
                        }
                    )
                    print(f"  {name}: {len(indices)} samples extracted")
                elif isinstance(time_series, list):
                    # Marker stream
                    extracted_markers = [time_series[i] for i in indices]
                    extracted_ts = ts[indices] - start_time

                    streams_to_replay.append(
                        {
                            "name": name,
                            "type": stype,
                            "channel_count": 1,
                            "nominal_srate": 0,  # Irregular rate for markers
                            "format": "string",
                            "data": extracted_markers,
                            "timestamps": extracted_ts,
                            "is_marker": True,
                        }
                    )
                    print(f"  {name}: {len(indices)} markers extracted")

    if not streams_to_replay:
        print("No data to replay!")
        return

    # Create LSL outlets
    print("\nCreating LSL outlets...")
    outlets = []

    for stream_info in streams_to_replay:
        # Map format string to pylsl constant
        from pylsl import cf_double64, cf_float32, cf_int8, cf_int16, cf_int32, cf_string

        format_map = {
            "float32": cf_float32,
            "double64": cf_double64,
            "int32": cf_int32,
            "int16": cf_int16,
            "int8": cf_int8,
            "string": cf_string,
        }

        lsl_info = StreamInfo(
            name=stream_info["name"],
            type=stream_info["type"],
            channel_count=stream_info["channel_count"],
            nominal_srate=stream_info["nominal_srate"],
            channel_format=format_map[stream_info["format"]],
            source_id=f"emgio_test_{stream_info['name']}",
        )

        outlet = StreamOutlet(lsl_info)
        outlets.append((outlet, stream_info))
        print(f"  Created outlet: {stream_info['name']}")

    # Give outlets time to be discovered
    print("\nWaiting for outlets to be discoverable...")
    time.sleep(2)

    # Start LabRecorder
    print(f"\nStarting LabRecorderCLI to record to: {OUTPUT_XDF}")
    recorder_proc = subprocess.Popen(
        [str(LABRECORDER_CLI), str(OUTPUT_XDF)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for recorder to start
    time.sleep(2)

    # Push data
    print("\nPushing data to LSL streams...")
    local_clock()

    # Find the maximum duration
    max_duration = max(s["timestamps"][-1] for s in streams_to_replay if len(s["timestamps"]) > 0)

    # Push samples in real-time (or faster)
    current_time = 0
    time_step = 0.001  # 1ms resolution

    while current_time <= max_duration:
        for outlet, stream_info in outlets:
            ts = stream_info["timestamps"]
            data = stream_info["data"]

            # Find samples at this time
            mask = (ts >= current_time) & (ts < current_time + time_step)
            indices = np.where(mask)[0]

            for idx in indices:
                if stream_info.get("is_marker"):
                    # Push marker
                    outlet.push_sample(data[idx])
                else:
                    # Push numeric sample
                    sample = data[idx].tolist() if hasattr(data[idx], "tolist") else data[idx]
                    outlet.push_sample(sample)

        current_time += time_step
        # Small sleep to not overwhelm
        if current_time % 0.1 < time_step:
            time.sleep(0.01)

    print("\nFinishing recording...")
    time.sleep(1)

    # Stop recorder
    recorder_proc.terminate()
    recorder_proc.wait(timeout=5)

    print(f"\nDone! Output file: {OUTPUT_XDF}")

    # Verify the output
    if OUTPUT_XDF.exists():
        print(f"File size: {OUTPUT_XDF.stat().st_size / 1024:.1f} KB")

        # Load and verify
        from emgio.importers.xdf import summarize_xdf

        summary = summarize_xdf(str(OUTPUT_XDF))
        print(f"\nVerification:\n{summary}")


if __name__ == "__main__":
    main()
