"""Memory regression test for the streaming Zarr exporter (issue #95).

The exporter used to hold, per (modality, rate) group, a float64 list of every
channel plus a float64 ``vstack`` copy plus the int16 copy at once (~4.5x the
int16 output); a 5 GB recording peaked at ~26 GB RSS and OOM'd every free CI
runner. The fix streams one channel at a time into a preallocated output-dtype
array. This test runs a real conversion in a subprocess and asserts the export's
marginal peak RSS (``ru_maxrss`` delta across ``to_zarr``) is below 1x the
float64 group size -- the old path needed ~2.25x and would fail. NO MOCKS: real
synthetic signals, real Zarr write. Skips when the ``zarr`` extra is absent.
"""

import subprocess
import sys

import pytest

pytest.importorskip("zarr", reason="Zarr serving format requires the optional 'zarr' extra")

# A single MISC group (no modality rate cap -> no resample, n_time == n_in) so
# the float64 group size is exactly n_ch * n_in * 8. The signals frame is built
# in one shot so the build's high-water mark does not pollute the export delta.
_MEASURE = """
import gc, os, resource, sys, tempfile
import numpy as np
import pandas as pd
from biosigio import Recording

# Large enough that the old float64 double-buffer (~2.25x the group) dwarfs
# Zarr/Blosc's fixed write buffers, so the delta cleanly reflects the fix.
N_CH, N_IN = 96, 1_000_000
rng = np.random.default_rng(0)
arr = (rng.standard_normal((N_IN, N_CH)) * 100.0).astype(np.float64)
rec = Recording()
rec.signals = pd.DataFrame(arr, columns=[f"M{i}" for i in range(N_CH)])
for i in range(N_CH):
    rec.channels[f"M{i}"] = {
        "sample_frequency": 500,
        "physical_dimension": "uV",
        "channel_type": "MISC",
        "modality": "MISC",
    }

unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
gc.collect()
baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
rec.to_zarr(os.path.join(tempfile.mkdtemp(), "mem"))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit

group_bytes = N_CH * N_IN * 8
print(f"{peak - baseline} {group_bytes}")
"""


def test_zarr_export_peak_memory_bounded():
    proc = subprocess.run(
        [sys.executable, "-c", _MEASURE],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    delta_bytes, group_bytes = (int(x) for x in proc.stdout.split())
    # The streaming export's marginal peak is the int16 output (~0.25x the
    # float64 group) + the min/max pyramid + Blosc write buffers, measured ~1.2x
    # the group. The old path held a float64 channel list AND a float64 vstack
    # copy at once (>=2x the group on top of that). A 1.75x ceiling passes
    # streaming with headroom and fails if any full float64 copy of the group is
    # reintroduced (which would push it past ~2.2x). The byte ratio is
    # hardware-independent.
    ceiling = int(1.75 * group_bytes)
    assert delta_bytes < ceiling, (
        f"export added {delta_bytes / 1e6:.0f} MB (>{ceiling / 1e6:.0f} MB = 1.75x the "
        f"{group_bytes / 1e6:.0f} MB float64 group); a full-group float64 copy was likely reintroduced"
    )
