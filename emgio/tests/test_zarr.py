"""Round-trip and contract tests for the biosigIO Zarr serving format.

NO MOCKS: real synthetic signals (sines/ramps, the same pattern the tabular and
neo tests use) give precise control to assert the per-modality rate caps,
anti-aliasing, int16 quantization tolerance, the min/max view-pyramid flags, and
event/metadata fidelity; one end-to-end test exports a real EMG EDF fixture.
Skips when the optional ``zarr`` extra is absent.
"""

import datetime
import pathlib

import numpy as np
import pytest

from emgio import Recording

zarr = pytest.importorskip("zarr", reason="Zarr serving format requires the optional 'zarr' extra")

from emgio.exporters.zarr import _DISCRETE_TYPES, _resample_channel  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
EMG_EDF = _REPO / "examples/bids/emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf"


def _open(path):
    return zarr.open_group(store=zarr.storage.LocalStore(str(path)), mode="r")


def test_zarr_float32_roundtrip_lossless(tmp_path):
    """float32, no resample (rate == cap): signals reconstruct exactly."""
    t = np.arange(2000)
    rec = Recording()
    rec.add_channel("C3", np.sin(t / 5.0).astype(np.float64), 250, "uV", "EEG")
    rec.add_channel("C4", np.cos(t / 8.0).astype(np.float64), 250, "uV", "EEG")

    out = rec.to_zarr(str(tmp_path / "r"), dtype="float32")
    assert out.endswith(".zarr")
    rt = Recording.from_file(out)

    assert rt.signals is not None
    assert list(rt.signals.columns) == ["C3", "C4"]
    assert np.allclose(rt.signals.to_numpy(), rec.signals.to_numpy())
    for ch in ("C3", "C4"):
        assert rt.channels[ch]["sample_frequency"] == 250.0
        assert rt.channels[ch]["physical_dimension"] == "uV"
        assert rt.channels[ch]["modality"] == "EEG"


def test_zarr_int16_within_quantization_step(tmp_path):
    """int16 reconstruction stays within one quantization step of the source."""
    t = np.arange(3000)
    data = (np.sin(t / 4.0) * 100.0).astype(np.float64)
    rec = Recording()
    rec.add_channel("EMG1", data, 1000, "uV", "EMG")  # 1000 == EMG cap, no resample

    rt = Recording.from_file(rec.to_zarr(str(tmp_path / "r")))
    step = (data.max() - data.min()) / 65535.0
    assert np.max(np.abs(rt.signals["EMG1"].to_numpy() - data)) <= step


def test_zarr_modality_cap_resamples_and_flags_antialias(tmp_path):
    """EEG above the 250 Hz cap is anti-alias downsampled; below it is untouched."""
    t = np.arange(4000)
    rec = Recording()
    rec.add_channel("hi", np.sin(t / 6.0).astype(np.float64), 500, "uV", "EEG")  # -> 250
    rec.add_channel("lo", np.sin(t / 6.0).astype(np.float64), 100, "uV", "EEG")  # stays 100

    root = _open(rec.to_zarr(str(tmp_path / "r")))
    names = [k for k in root.keys() if k != "events"]
    assert set(names) == {"eeg_250hz", "eeg_100hz"}
    assert root["eeg_250hz"].attrs["n_samples"] == 2000  # 4000 @ 500Hz -> 2000 @ 250Hz
    assert root["eeg_250hz"]["0"].attrs["anti_aliased"] is True
    assert root["eeg_100hz"].attrs["n_samples"] == 4000  # never upsampled, unchanged
    assert root["eeg_100hz"]["0"].attrs["anti_aliased"] is False


def test_zarr_view_pyramid_flagged_not_for_inference(tmp_path):
    """A view pyramid is built and marked render-only; level 0 is inference-usable."""
    rec = Recording()
    rec.add_channel("C1", np.sin(np.arange(8000) / 3.0).astype(np.float64), 250, "uV", "EEG")

    grp = _open(rec.to_zarr(str(tmp_path / "r")))["eeg_250hz"]
    assert grp["0"].attrs["usable_for_inference"] is True
    view = grp["view"]
    levels = sorted(view.keys(), key=int)
    assert levels, "expected at least one view pyramid level"
    for lvl in levels:
        a = view[lvl]
        assert a.attrs["kind"] == "minmax_envelope"
        assert a.attrs["usable_for_inference"] is False
        assert a.shape[0] == 2  # [min, max]


def test_zarr_discrete_channel_nearest_not_inference(tmp_path):
    """A TRIG channel is nearest-resampled (no anti-alias) and not inference-usable."""
    steps = np.repeat([0.0, 1.0, 0.0, 1.0], 1000)  # 4000 samples of square edges
    rec = Recording()
    rec.add_channel("trig", steps, 500, "n/a", "TRIG")  # EEG-less; modality MISC, 500->?

    root = _open(rec.to_zarr(str(tmp_path / "r")))
    name = next(k for k in root.keys() if k != "events")
    chan = root[name].attrs["channels"][0]
    assert chan["channel_type"] == "TRIG"
    assert chan["anti_aliased"] is False
    assert chan["usable_for_inference"] is False
    # A group of only discrete channels must not be flagged inference-usable.
    assert root[name]["0"].attrs["usable_for_inference"] is False


def test_zarr_int16_rejects_non_finite(tmp_path):
    """NaN/inf would silently decode to a midrange int16 value, so int16 export rejects it."""
    data = np.sin(np.arange(1000) / 5.0)
    data[100] = np.nan
    rec = Recording()
    rec.add_channel("C1", data, 250, "uV", "EEG")
    with pytest.raises(ValueError, match="non-finite"):
        rec.to_zarr(str(tmp_path / "r"))  # dtype="int16" default


def test_zarr_float32_preserves_nan(tmp_path):
    """float32 storage represents NaN, so the gap survives the round-trip."""
    data = np.sin(np.arange(1000) / 5.0)
    data[100:110] = np.nan
    rec = Recording()
    rec.add_channel("C1", data, 250, "uV", "EEG")

    rt = Recording.from_file(rec.to_zarr(str(tmp_path / "r"), dtype="float32"))
    got = rt.signals["C1"].to_numpy()
    assert np.array_equal(np.isnan(got), np.isnan(data))
    assert np.allclose(got[~np.isnan(got)], data[~np.isnan(data)])


def test_zarr_events_roundtrip(tmp_path):
    rec = Recording()
    rec.add_channel("C1", np.sin(np.arange(1000) / 5.0).astype(np.float64), 250, "uV", "EEG")
    rec.add_event(0.5, 0.1, "stim")
    rec.add_event(1.0, 0.0, "resp")
    rec.add_event(1.5, 0.2, "stim")  # repeated label exercises the code->label map

    rt = Recording.from_file(rec.to_zarr(str(tmp_path / "r")))
    events = rt.events.sort_values("onset").reset_index(drop=True)
    assert list(events["description"]) == ["stim", "resp", "stim"]
    assert np.allclose(events["onset"].to_numpy(), [0.5, 1.0, 1.5])
    assert np.allclose(events["duration"].to_numpy(), [0.1, 0.0, 0.2])


def test_zarr_metadata_types_preserved(tmp_path):
    """Recording metadata round-trips with types intact (datetime stays datetime)."""
    rec = Recording()
    rec.add_channel("C1", np.zeros(500), 250, "uV", "EEG")
    rec.set_metadata("startdate", datetime.datetime(2026, 5, 31, 12, 0, 0))
    rec.set_metadata("subject", "sub-01")

    rt = Recording.from_file(rec.to_zarr(str(tmp_path / "r")))
    assert rt.metadata["startdate"] == datetime.datetime(2026, 5, 31, 12, 0, 0)
    assert type(rt.metadata["startdate"]) is datetime.datetime
    assert rt.metadata["subject"] == "sub-01"


def test_zarr_multigroup_requires_group_selector(tmp_path):
    """A multi-rate store cannot collapse to one grid; group= picks one."""
    t = np.arange(2000)
    rec = Recording()
    rec.add_channel("E1", np.sin(t / 5.0).astype(np.float64), 250, "uV", "EEG")
    rec.add_channel("M1", np.cos(t / 5.0).astype(np.float64), 1000, "uV", "EMG")

    out = rec.to_zarr(str(tmp_path / "r"))
    with pytest.raises(ValueError, match="group="):
        Recording.from_file(out)
    rt = Recording.from_file(out, importer="zarr", group="emg_1000hz")
    assert list(rt.channels) == ["M1"]
    assert rt.channels["M1"]["modality"] == "EMG"


def test_zarr_infer_importer_and_no_signals():
    assert Recording._infer_importer("store.zarr") == "zarr"
    with pytest.raises(ValueError, match="No signals loaded"):
        Recording().to_zarr("unused")


@pytest.mark.skipif(not EMG_EDF.exists(), reason="EMG fixture missing")
def test_zarr_real_emg_fixture_exports_and_reconstructs(tmp_path):
    """End-to-end on a real EMG recording: export, then reconstruct one group.

    Reconstruction is compared against the exporter's own anti-aliased resample of
    the source (the serving copy is downsampled, not the full-rate original), so
    fidelity is the int16 quantization tolerance, not the raw signal.
    """
    rec = Recording.from_file(str(EMG_EDF))
    root = _open(rec.to_zarr(str(tmp_path / "emg")))
    assert root.attrs["format"] == "biosigio-zarr"
    groups = [k for k in root.keys() if k != "events"]
    assert groups

    # Reconstruct the first group and check a channel against the resampled source.
    gname = groups[0]
    rt = Recording.from_file(str(tmp_path / "emg.zarr"), importer="zarr", group=gname)
    label = list(rt.channels)[0]
    native = rec.channels[label]["sample_frequency"]
    target = rt.channels[label]["sample_frequency"]
    ctype = str(rec.channels[label].get("channel_type", "")).upper()
    expected = _resample_channel(
        rec.signals[label].to_numpy(), native, target, discrete=ctype in _DISCRETE_TYPES
    )
    got = rt.signals[label].to_numpy()
    n = min(len(expected), len(got))
    assert np.corrcoef(expected[:n], got[:n])[0, 1] > 0.99
