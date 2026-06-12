"""Streaming Zarr export (bounded-memory large-recording path, NEMAR nemar-cli#737/#736).

Verifies that the streaming converter (lazy MNE read -> channel-major memmap ->
per-channel resample/quantize/write) reproduces what a full in-memory load would
produce, on a REAL multi-channel EDF (no mocks). The key property is
streamed == full-load for the same reader, so a 18 GB recording converts in
bounded RAM without changing the output.

Requires the zarr + meg (MNE) extras; skips cleanly otherwise.
"""

import os
import pathlib
import tempfile

import numpy as np
import pyedflib
import pytest

pytest.importorskip("zarr", reason="streaming Zarr export requires the 'zarr' extra")
pytest.importorskip("mne", reason="streaming Zarr export requires the 'meg' extra (mne)")

import zarr  # noqa: E402

from biosigio import stream_to_zarr  # noqa: E402
from biosigio.exporters.zarr import _DISCRETE_TYPES, _resample_channel  # noqa: E402
from biosigio.importers._mne_common import _MNE_TYPE_TO_biosigIO  # noqa: E402

# Real committed MEG fixtures (see test_meg_importer.py): a FIF file (305 ch,
# 100 Hz, 30 s) and a CTF .ds directory (244 ch, 1250 Hz, 4 s).
_REPO = pathlib.Path(__file__).resolve().parents[2]
MEG_FIF = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"
CTF_DS = _REPO / "examples/ctf/catch-alp-good-f.ds"


def _write_edf(path: str, rate: float, duration_s: float, n_ch: int) -> np.ndarray:
    """Write a real n-channel EDF at one rate; return the (n_ch, n_samples) data."""
    n = int(rate * duration_s)
    t = np.arange(n) / rate
    data = np.vstack([(30.0 + 5 * c) * np.sin(2 * np.pi * (3 + c) * t) for c in range(n_ch)])
    headers = []
    for c in range(n_ch):
        row = data[c]
        headers.append(
            {
                "label": f"EEG{c}",
                "dimension": "uV",
                "sample_frequency": rate,
                "physical_max": float(np.max(row)),
                "physical_min": float(np.min(row)),
                "digital_max": 32767,
                "digital_min": -32768,
                "prefilter": "n/a",
                "transducer": "n/a",
            }
        )
    w = pyedflib.EdfWriter(path, n_ch)
    try:
        w.setSignalHeaders(headers)
        w.writeSamples(list(data))
    finally:
        w.close()
    return data


class _TmpEDF:
    def __init__(self, rate, duration_s, n_ch):
        self.rate, self.duration_s, self.n_ch = rate, duration_s, n_ch

    def __enter__(self):
        fd, self.path = tempfile.mkstemp(suffix=".edf")
        os.close(fd)
        self.data = _write_edf(self.path, self.rate, self.duration_s, self.n_ch)
        return self.path

    def __exit__(self, *_exc):
        os.unlink(self.path)


def _mne_full_data(path):
    """Ground truth: load the whole EDF with MNE (preload) -> (n_ch, n_samples)."""
    import mne

    raw = mne.io.read_raw(path, preload=True, verbose="ERROR")
    return raw.get_data(), float(raw.info["sfreq"])


def _dequant(store_path, gname):
    g = zarr.open_group(store_path, mode="r")[gname]
    base = np.asarray(g["0"][:])
    scale = np.asarray(g["0"].attrs["scale"])[:, None]
    offset = np.asarray(g["0"].attrs["offset"])[:, None]
    return base * scale + offset


def test_stream_no_resample_reproduces_signal():
    """native <= cap: dequantized base matches the full-load signal within 1 LSB."""
    with _TmpEDF(rate=200.0, duration_s=20.0, n_ch=6) as path:
        full, _sfreq = _mne_full_data(path)
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "rec.zarr")
            stream_to_zarr(path, store, force_modality="EEG")
            deq = _dequant(store, "eeg_200hz")
            assert deq.shape == full.shape  # 200 Hz <= 250 cap -> no resample
            for i in range(full.shape[0]):
                rng = float(full[i].max() - full[i].min()) or 1.0
                assert np.max(np.abs(deq[i] - full[i])) <= 2 * rng / 65535


def test_stream_with_resample_matches_reference():
    """native > cap: streamed base matches resample-then-quantize of the full signal."""
    with _TmpEDF(rate=500.0, duration_s=20.0, n_ch=4) as path:
        full, _sfreq = _mne_full_data(path)
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "rec.zarr")
            stream_to_zarr(path, store, force_modality="EEG")  # 500 -> 250 cap
            deq = _dequant(store, "eeg_250hz")
            n_time = int(round(full.shape[1] * 250.0 / 500.0))
            assert deq.shape == (4, n_time)
            for i in range(full.shape[0]):
                ref = _resample_channel(full[i], 500.0, 250.0, discrete=False)
                rng = float(ref.max() - ref.min()) or 1.0
                # streamed goes through a float32 memmap; allow a few LSB of slack.
                assert np.max(np.abs(deq[i] - ref)) <= 5 * rng / 65535


def test_stream_store_structure_and_pyramid():
    """Root/group/array attrs and a min/max view pyramid are written."""
    with _TmpEDF(rate=200.0, duration_s=40.0, n_ch=3) as path:
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "rec.zarr")
            stream_to_zarr(path, store, force_modality="EEG")
            root = zarr.open_group(store, mode="r")
            ra = dict(root.attrs)
            assert ra["format"] == "biosigio-zarr"
            assert ra["channel_groups"] == ["eeg_200hz"]
            assert ra["dtype"] == "int16"
            g = root["eeg_200hz"]
            assert dict(g.attrs)["n_channels"] == 3
            assert dict(g.attrs)["rate"] == 200.0
            # 8000 samples (40 s @ 200 Hz) -> at least one view level (min_view 512).
            assert "1" in g["view"]
            v1 = g["view"]["1"]
            assert v1.shape[0] == 2 and v1.shape[1] == 3  # [min,max] x n_ch
            assert dict(v1.attrs)["kind"] == "minmax_envelope"


def test_stream_embeds_events():
    """An events_df is written as onset/duration/code arrays + label_map."""
    import pandas as pd

    ev = pd.DataFrame(
        {"onset": [0.5, 1.5, 2.5], "duration": [0.0, 0.0, 0.0], "description": ["a", "b", "a"]}
    )
    with _TmpEDF(rate=100.0, duration_s=5.0, n_ch=2) as path:
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "rec.zarr")
            stream_to_zarr(path, store, force_modality="EEG", events_df=ev)
            eg = zarr.open_group(store, mode="r")["events"]
            assert dict(eg.attrs)["n_events"] == 3
            assert set(dict(eg.attrs)["label_map"].values()) == {"a", "b"}
            assert np.asarray(eg["onset"][:]).tolist() == [0.5, 1.5, 2.5]


def test_stream_rejects_bad_dtype():
    with _TmpEDF(rate=100.0, duration_s=2.0, n_ch=2) as path:
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="dtype must be"):
                stream_to_zarr(path, os.path.join(d, "x.zarr"), dtype="int8")


# -- Real FIF MEG (the formats NEMAR actually streams: BrainVision/FIF) ----------


@pytest.mark.skipif(not MEG_FIF.exists(), reason="MEG FIF fixture missing")
def test_stream_real_fif_reproduces_full_load():
    """Streaming a real .fif reproduces the full-load signal within int16 quant."""
    import mne

    full = mne.io.read_raw_fif(str(MEG_FIF), preload=True, verbose="ERROR")
    full_data = full.get_data()  # (n_ch, n_samp); 100 Hz <= MEG cap 250 -> no resample
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "rec.zarr")
        stream_to_zarr(str(MEG_FIF), store, force_modality="MEG")
        assert dict(zarr.open_group(store, mode="r").attrs)["channel_groups"] == ["meg_100hz"]
        deq = _dequant(store, "meg_100hz")
        assert deq.shape == full_data.shape
        for i in range(full_data.shape[0]):
            rng = float(full_data[i].max() - full_data[i].min()) or 1.0
            assert np.max(np.abs(deq[i] - full_data[i])) <= 2 * rng / 65535


@pytest.mark.skipif(not CTF_DS.exists(), reason="CTF .ds fixture missing")
def test_stream_real_ctf_reproduces_full_load():
    """Streaming a real CTF .ds *directory* reproduces the resampled full-load
    signal within int16 quant.

    This is the path the NEMAR driver routes .ds recordings through. catch-alp
    is 1250 Hz > the 250 Hz MEG cap, so it also exercises the streaming resample
    on a directory-valued recording (not a single file). The lone TRIG channel
    resamples discretely; the 243 MEG/EEG/MISC channels continuously -- the
    reference mirrors that per-channel decision."""
    import mne

    raw = mne.io.read_raw_ctf(str(CTF_DS), preload=True, verbose="ERROR")
    full = raw.get_data()  # (244, 5000) in ch_names order
    native = float(raw.info["sfreq"])  # 1250 Hz
    mne_types = raw.get_channel_types()
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "rec.zarr")
        stream_to_zarr(str(CTF_DS), store, force_modality="MEG")
        groups = dict(zarr.open_group(store, mode="r").attrs)["channel_groups"]
        assert groups == ["meg_250hz"]  # 1250 -> 250 cap, all channels forced to MEG
        deq = _dequant(store, "meg_250hz")
        n_time = int(round(full.shape[1] * 250.0 / native))
        assert deq.shape == (244, n_time)
        for i in range(full.shape[0]):
            ctype = _MNE_TYPE_TO_biosigIO.get(mne_types[i], "OTHER")
            discrete = ctype in _DISCRETE_TYPES
            ref = _resample_channel(full[i], native, 250.0, discrete=discrete)
            rng = float(ref.max() - ref.min()) or 1.0
            # streamed goes through a float32 memmap; allow a few LSB of slack.
            assert np.max(np.abs(deq[i] - ref)) <= 5 * rng / 65535


@pytest.mark.skipif(not MEG_FIF.exists(), reason="MEG FIF fixture missing")
def test_stream_split_fif_covers_whole_chain():
    """Streaming the FIRST split of a multi-file FIF must convert the WHOLE
    recording, not just the first split's portion (regression guard for the
    on005261 split-FIF bug: read_raw follows the chain from split-01)."""
    import mne

    full = mne.io.read_raw_fif(str(MEG_FIF), preload=True, verbose="ERROR")
    n_full = int(full.n_times)
    with tempfile.TemporaryDirectory() as d:
        # The 3.6 MB fixture splits into a 10-part chain at 1.5 MB.
        full.save(
            os.path.join(d, "sub-01_task-mouse_meg.fif"),
            split_size="1.5MB",
            split_naming="bids",
            verbose="ERROR",
        )
        first = os.path.join(d, "sub-01_task-mouse_split-01_meg.fif")
        assert os.path.exists(first)
        assert os.path.exists(os.path.join(d, "sub-01_task-mouse_split-02_meg.fif"))
        store = os.path.join(d, "rec.zarr")
        stream_to_zarr(first, store, force_modality="MEG")
        g = zarr.open_group(store, mode="r")["meg_100hz"]
        # Whole-chain length, not the ~300-sample first split.
        assert dict(g.attrs)["n_samples"] == n_full
