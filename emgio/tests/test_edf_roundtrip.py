"""Regression test: EDF/BDF export must preserve the full recording length.

Guards against the exporter bug where ``writePhysicalSamples`` was called once
per channel with the full array. pyedflib writes exactly one data record per
such call, so every export was silently truncated to a single record (one
second). Uses a real, multi-second OTB fixture (NO MOCKS).
"""

import os

import pytest

from emgio import Recording

EXAMPLE_DIR = "examples"
OTB_FIXTURE = os.path.join(EXAMPLE_DIR, "one_sessantaquattro_truncated.otb+")


@pytest.mark.skipif(not os.path.exists(OTB_FIXTURE), reason="OTB example fixture missing")
@pytest.mark.parametrize("fmt", ["edf", "bdf"])
def test_export_preserves_full_recording_length(tmp_path, fmt):
    """A multi-second recording must round-trip without losing samples."""
    emg = Recording.from_file(OTB_FIXTURE, importer="otb")
    first = emg.signals.columns[0]
    n_in = len(emg.signals[first])
    samples_per_record = emg.channels[first]["sample_frequency"]
    # The fixture must be longer than one data record, or it cannot expose the bug.
    assert n_in > samples_per_record

    out = tmp_path / f"roundtrip.{fmt}"
    emg.to_edf(str(out), format=fmt, bypass_analysis=True)

    reloaded = Recording.from_file(str(out))
    n_out = len(reloaded.signals[reloaded.signals.columns[0]])
    # The true invariant is "no truncation". EDF stores whole data records, so a
    # recording that is not an integer number of seconds is zero-padded by at most
    # one record on the last block; never fewer samples than the input.
    assert n_in <= n_out <= n_in + samples_per_record, (
        f"{fmt} export changed length: {n_in} samples -> {n_out}"
    )


@pytest.mark.skipif(
    not os.path.exists(os.path.join(EXAMPLE_DIR, "truncated_trigno_sample.csv")),
    reason="Trigno example fixture missing",
)
def test_export_rejects_mixed_sample_rates(tmp_path):
    """Mixed per-channel sample rates must fail loudly, not write a corrupt file."""
    emg = Recording.from_file(
        os.path.join(EXAMPLE_DIR, "truncated_trigno_sample.csv"), importer="trigno"
    )
    rates = {int(emg.channels[c]["sample_frequency"]) for c in emg.channels}
    assert len(rates) > 1, "fixture must have mixed rates to exercise the guard"
    with pytest.raises(ValueError, match="single sampling rate"):
        emg.to_edf(str(tmp_path / "mixed.edf"), format="edf", bypass_analysis=True)
