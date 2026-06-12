"""KIT/Yokogawa MEG importer (read_raw_kit) on a real .sqd fixture (no mocks).

Uses a small vendored KIT recording (examples/kit/sub-01_task-test_meg.sqd, ~100 KB,
from MNE's BSD-licensed test suite; see examples/kit/README.md). Validates that
biosigIO reads KIT data through Recording.from_file and that the recording converts
to a Zarr serving store -- the NEMAR use for nm000229's 196 .con files.

Skips cleanly if the optional 'meg' extra (mne) is not installed.
"""

import os
import pathlib
import tempfile

import pytest

from biosigio import Recording

pytest.importorskip("mne", reason="KIT import requires the optional 'meg' extra (mne)")

_REPO = pathlib.Path(__file__).resolve().parents[2]
KIT = _REPO / "examples/kit/sub-01_task-test_meg.sqd"

pytestmark = pytest.mark.skipif(not KIT.exists(), reason="KIT fixture missing")


@pytest.fixture(scope="module")
def kit_rec():
    return Recording.from_file(str(KIT))


def test_kit_reads_channels_and_rate(kit_rec):
    """A KIT .sqd opens with channels and a single sampling rate."""
    assert kit_rec.get_n_channels() == 193
    assert kit_rec.get_sampling_frequency() == 1000.0
    assert kit_rec.get_n_samples() > 0


def test_kit_has_meg_channels(kit_rec):
    """The MEG sensors are present and typed as MEG (not collapsed/fabricated)."""
    modalities = {i.get("modality") for i in kit_rec.channels.values()}
    assert "MEG" in modalities


def test_kit_source_format_recorded(kit_rec):
    """Provenance: the recording knows it came through the meg importer."""
    assert kit_rec.metadata.get("source_format") == "meg"


def test_kit_converts_to_zarr():
    """The NEMAR path: a KIT recording exports to a valid Zarr store."""
    pytest.importorskip("zarr", reason="Zarr export requires the 'zarr' extra")
    rec = Recording.from_file(str(KIT))
    # Force MEG modality on all channels (mirrors the NEMAR suffix-driven grouping).
    for label in rec.channels:
        rec.channels[label]["modality"] = "MEG"
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "rec.zarr")
        rec.to_zarr(store, dtype="int16")
        assert os.path.exists(os.path.join(store, "zarr.json"))
        import zarr

        root = zarr.open_group(store, mode="r")
        groups = dict(root.attrs).get("channel_groups", [])
        assert any(g.startswith("meg_") for g in groups)
