"""Regression test: EDF/BDF export must preserve the full recording length.

Guards against the exporter bug where ``writePhysicalSamples`` was called once
per channel with the full array. pyedflib writes exactly one data record per
such call, so every export was silently truncated to a single record (one
second). Uses a real, multi-second OTB fixture (NO MOCKS).
"""

import os

import pytest

from emgio import EMG

EXAMPLE_DIR = "examples"
OTB_FIXTURE = os.path.join(EXAMPLE_DIR, "one_sessantaquattro_truncated.otb+")


@pytest.mark.skipif(not os.path.exists(OTB_FIXTURE), reason="OTB example fixture missing")
@pytest.mark.parametrize("fmt", ["edf", "bdf"])
def test_export_preserves_full_recording_length(tmp_path, fmt):
    """A multi-second recording must round-trip without losing samples."""
    emg = EMG.from_file(OTB_FIXTURE, importer="otb")
    first = emg.signals.columns[0]
    n_in = len(emg.signals[first])
    samples_per_record = emg.channels[first]["sample_frequency"]
    # The fixture must be longer than one data record, or it cannot expose the bug.
    assert n_in > samples_per_record

    out = tmp_path / f"roundtrip.{fmt}"
    emg.to_edf(str(out), format=fmt, bypass_analysis=True)

    reloaded = EMG.from_file(str(out))
    n_out = len(reloaded.signals[reloaded.signals.columns[0]])
    assert n_out == n_in, f"{fmt} export truncated {n_in} samples to {n_out}"
