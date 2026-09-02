"""Real-recording regression for BIDS unit adoption (issue #122).

The synthetic BrainVision fixtures in ``test_bids_channels_units.py`` pin the
arithmetic; this pins the actual recording the issue was reported on, NEMAR
``nm000182`` (Nejedly et al. 2020 SEEG graphoelements, public). Before the fix,
``bids_channels="off"`` gave ``physical_dimension = "V"`` with a standard
deviation of 1.0e-6, and applying the sidecar gave ``"uV"`` with the **same**
1.0e-6 -- microvolts claimed over volt-valued samples, wrong by 10^6.

Opt in and run it
-----------------

The recording is ~40 MB and is **not** committed. Set ``BIOSIGIO_REAL_DATA=1``
to opt in; without it the test skips, so the default suite stays offline and
fast::

    BIOSIGIO_REAL_DATA=1 uv run pytest biosigio/tests/test_real_data_nm000182.py

Download, caching and the offline skip are all
:mod:`biosigio.tests.real_data`'s job (cache: ``~/.cache/biosigio/real_data``,
overridable with ``BIOSIGIO_REAL_DATA_CACHE``). A BrainVision recording is four
files -- the ``.vhdr`` names its ``.eeg`` and ``.vmrk`` siblings by bare
filename, and ``find_channels_tsv`` looks for the ``_channels.tsv`` next to the
header -- so all four are fetched under their own names into that one flat
cache directory, which is exactly the adjacency MNE and BIDS both expect. Each
is verified against its published sha256 (see ``FILES``), so a corrupted or
partially-served file fails loudly instead of quietly changing the numbers this
test asserts.

data.nemar.org returns an intermittent HTTP 500 on a small fraction of
requests, independent of file or client, so an occasional skip on a working
network is expected; re-run and the cached files make it moot.
"""

import numpy as np
import pytest

from biosigio import Recording
from biosigio.tests.real_data import fetch_real_recording

pytest.importorskip("mne", reason="BrainVision import requires the optional 'meg' extra (mne)")

# Pinned to a released version so the bytes cannot change under the assertions
# (the unversioned data.nemar.org path 404s).
BASE_URL = "https://data.nemar.org/nm000182/v1.0.2/sub-000/ieeg/"
STEM = "sub-000_task-MachineLearningEEG_run-001"
CHANNEL = "LMacro_01"
RESOLUTION_MICROVOLTS = 0.1  # "Ch1=LMacro_01,,0.1,µV" in the real header

# Suffix -> (size floor, sha256). The floor rejects a truncated download or an
# HTML error page cheaply; the digest is the real check.
#
# Every digest here describes bytes checked against nm000182's published v1.0.2
# manifest, not merely whatever this machine downloaded. The manifest carries two
# checksum kinds, and both were verified: `.vhdr` and `.eeg` are git-annex
# objects listed with sha256, which are these values verbatim; `.vmrk` and
# `channels.tsv` are git-tracked text listed with a git blob SHA-1, which
# `git hash-object` reproduced (7b04441..., 50cf2f6...) before their sha256 was
# taken from those same verified bytes.
FILES = {
    "_ieeg.vhdr": (700, "4f6759c67d128cdcde57f6ae7d8e80de07f687f3f48b040399be212d40b48864"),
    "_ieeg.vmrk": (25_000, "9879a6af5f5c0e37ecdaba7cdbd251277c008f6849f42c5f6d314be5e8c57c03"),
    "_channels.tsv": (100, "853cc22dc82ca72fb0bef02f04bd288bce7bfb76b673406959a85a4af63c2ee6"),
    "_ieeg.eeg": (42_000_000, "3900b96bb475c62d8a0057195c580f899af191b483d159b4f607d0d85c5221fa"),
}


@pytest.fixture(scope="module")
def vhdr():
    """The cached ``.vhdr``, with its ``.eeg``/``.vmrk``/``channels.tsv`` beside it."""
    paths = {
        suffix: fetch_real_recording(
            f"{BASE_URL}{STEM}{suffix}", min_bytes=min_bytes, sha256=digest
        )
        for suffix, (min_bytes, digest) in FILES.items()
    }
    return paths["_ieeg.vhdr"]


@pytest.fixture(scope="module")
def applied(vhdr) -> Recording:
    return Recording.from_file(str(vhdr))


@pytest.fixture(scope="module")
def ignored(vhdr) -> Recording:
    return Recording.from_file(str(vhdr), bids_channels="off")


def test_sidecar_puts_the_real_recording_on_a_microvolt_scale(applied, ignored):
    """The reported numbers: 1.0e-6 V becomes 1.0 uV, not "uV" over 1.0e-6."""
    assert ignored.channels[CHANNEL]["physical_dimension"] == "V"
    assert applied.channels[CHANNEL]["physical_dimension"] == "uV"
    assert applied.channels[CHANNEL]["channel_type"] == "SEEG"
    assert applied.channels[CHANNEL]["modality"] == "IEEG"

    volts_std = float(ignored.signals[CHANNEL].std())
    microvolts_std = float(applied.signals[CHANNEL].std())

    # The pre-fix state, kept explicit so a regression reads as what it is.
    assert volts_std == pytest.approx(1.0e-6, rel=1e-3)
    # An SEEG amplitude in microvolts is an order-1-to-100 number, never 1e-6.
    assert 1.0 <= microvolts_std <= 100.0
    assert microvolts_std == pytest.approx(volts_std * 1e6, rel=1e-12)


def test_the_sidecar_recovers_the_files_own_numbers(applied, vhdr):
    """The .eeg holds µV/resolution; the adopted values must match it exactly.

    Reads the raw float32 block off disk and applies the header's declared
    resolution by hand, so the assertion does not go through MNE at all.
    """
    on_disk = np.fromfile(vhdr.with_name(f"{STEM}_ieeg.eeg"), dtype="<f4")
    native_microvolts = on_disk.astype(np.float64) * RESOLUTION_MICROVOLTS

    values = applied.signals[CHANNEL].to_numpy()
    assert values.shape == native_microvolts.shape
    assert np.allclose(values, native_microvolts, rtol=1e-6, atol=1e-9)


def test_zarr_export_of_the_real_recording_carries_microvolts(applied, tmp_path):
    """The serving store's unit, scale and offset describe the converted values."""
    zarr = pytest.importorskip("zarr", reason="requires the optional 'zarr' extra")

    store_path = applied.to_zarr(str(tmp_path / "nm000182"), dtype="int16")
    root = zarr.open_group(store=zarr.storage.LocalStore(store_path), mode="r")

    assert "ieeg_1000hz" in list(root.group_keys())
    group = root["ieeg_1000hz"]
    meta = next(c for c in group.attrs["channels"] if c["label"] == CHANNEL)
    assert meta["unit"] == "uV"

    dequantized = group["0"][meta["row_index"], :] * meta["scale"] + meta["offset"]
    # Order 1e0 in microvolts, six orders of magnitude off the pre-fix 1e-6. The
    # band is loose on the low side because the store is anti-aliased down to
    # 1000 Hz, which trims a little variance (measured: 0.9996 vs 1.0000).
    assert 0.1 <= float(np.std(dequantized)) <= 100.0

    # The store is a 5000 -> 1000 Hz anti-aliased copy, so compare distributions
    # rather than samples: the amplitude scale is what the unit label claims.
    source = applied.signals[CHANNEL].to_numpy()
    assert float(np.std(dequantized)) == pytest.approx(float(np.std(source)), rel=0.1)
    assert float(np.mean(dequantized)) == pytest.approx(float(np.mean(source)), abs=0.05)
