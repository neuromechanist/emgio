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

from biosigio import Recording

zarr = pytest.importorskip("zarr", reason="Zarr serving format requires the optional 'zarr' extra")

from biosigio.exporters.zarr import (  # noqa: E402
    _DISCRETE_TYPES,
    _resample_channel,
    _view_chunk_columns,
)
from biosigio.exporters.zarr_stream import _pyramid_level_lengths  # noqa: E402

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


def test_zarr_view_levels_chunked_at_constant_columns(tmp_path):
    """Every view level is chunked at a constant COLUMN count, and the group
    declares the pyramid and its geometry (issue #119).

    8 channels at 200 Hz for 600 s -> level 0 is 120000 samples and the pyramid is
    30000 / 7500 / 1875 / 468 (the last stops because 468 <= min_view_samples 512).
    The first three exceed the 1024-column default and are capped there; the fourth
    is shorter and becomes a single chunk. Under the old seconds-based rule the same
    levels would have been chunked at 200 / 50 / 12 / 3 columns, i.e. 150 / 150 /
    157 / 156 requests to read a level end to end instead of 30 / 8 / 2 / 1.
    """
    n = 120000
    rec = Recording()
    for c in range(8):
        rec.add_channel(f"C{c}", np.sin(np.arange(n) / (3.0 + c)), 200, "uV", "EEG")

    root = _open(rec.to_zarr(str(tmp_path / "r")))
    grp = root["eeg_200hz"]
    attrs = dict(grp.attrs)

    # The pyramid is declared, so a reader never probes view/1..n until a 404.
    assert attrs["n_view_levels"] == 4
    assert attrs["view_levels"] == [1, 2, 3, 4]
    assert sorted(grp["view"].keys(), key=int) == ["1", "2", "3", "4"]

    # Geometry is on the group, so a client can plan reads without opening arrays.
    assert attrs["view_downsample"] == 4
    assert attrs["view_chunk_columns"] == 1024
    assert attrs["chunk_seconds"] == 4.0
    assert attrs["shard_seconds"] == 300.0
    assert dict(root.attrs)["view_chunk_columns"] == 1024

    # Level 0 keeps its time-based chunk/shard, and now reports both in samples.
    a0 = grp["0"]
    assert a0.chunks == (8, 800)  # 4 s at 200 Hz
    assert a0.shards == (8, 60000)  # 300 s, a whole number of chunks
    assert dict(a0.attrs)["chunk_samples"] == 800
    assert dict(a0.attrs)["shard_samples"] == 60000
    assert dict(a0.attrs)["source_rate_hz"] == 200.0

    expected_lengths = [30000, 7500, 1875, 468]
    for level, length in zip([1, 2, 3, 4], expected_lengths, strict=True):
        av = grp["view"][str(level)]
        assert av.shape == (2, 8, length)
        assert av.chunks == (2, 8, min(length, 1024))
        assert dict(av.attrs)["chunk_columns"] == min(length, 1024)


def test_zarr_view_chunk_columns_override_changes_chunk_shape(tmp_path):
    """view_chunk_columns is honored end to end, on the arrays and in the attrs."""
    n = 120000
    rec = Recording()
    for c in range(8):
        rec.add_channel(f"C{c}", np.sin(np.arange(n) / (3.0 + c)), 200, "uV", "EEG")

    grp = _open(rec.to_zarr(str(tmp_path / "narrow"), view_chunk_columns=256))["eeg_200hz"]
    assert dict(grp.attrs)["view_chunk_columns"] == 256
    for level, length in zip([1, 2, 3, 4], [30000, 7500, 1875, 468], strict=True):
        av = grp["view"][str(level)]
        assert av.chunks == (2, 8, min(length, 256))
        assert dict(av.attrs)["chunk_columns"] == min(length, 256)


def test_zarr_short_view_level_is_a_single_chunk(tmp_path):
    """A level shorter than view_chunk_columns is one chunk, not a padded 1024."""
    rec = Recording()
    rec.add_channel("C1", np.sin(np.arange(3000) / 5.0), 250, "uV", "EEG")

    grp = _open(rec.to_zarr(str(tmp_path / "r")))["eeg_250hz"]
    assert dict(grp.attrs)["view_levels"] == [1, 2]
    for level, length in zip([1, 2], [750, 187], strict=True):  # both < 1024
        av = grp["view"][str(level)]
        assert av.shape[2] == length
        assert av.chunks[2] == length


def test_zarr_rejects_non_positive_view_chunk_columns(tmp_path):
    rec = Recording()
    rec.add_channel("C1", np.zeros(1000), 250, "uV", "EEG")
    with pytest.raises(ValueError, match="view_chunk_columns must be >= 1"):
        rec.to_zarr(str(tmp_path / "r"), view_chunk_columns=0)


# The 40-minute, 129-channel, 250 Hz EEG store the transfer-efficiency audit
# measured (nemarOrg/nemar-cli#1178): level 0 is 607586 samples.
_REFERENCE_N_TIME = 607586
_REFERENCE_LEVEL_LENGTHS = [151896, 37974, 9493, 2373, 593, 148]


def test_reference_store_view_chunk_counts():
    """Pure geometry over the audited reference store: the column rule turns a
    whole-recording read of level 4 from 594 requests into 3, and the level-6
    minimap from 148 into 1.

    Pins the two helpers the exporters share (`_pyramid_level_lengths` for the
    level lengths, `_view_chunk_columns` for the chunk width) against the numbers
    in the audit, so a change to either rule has to restate the acceptance case.
    """
    lengths = _pyramid_level_lengths(_REFERENCE_N_TIME, 4, 512, 12)
    assert lengths == _REFERENCE_LEVEL_LENGTHS

    def n_chunks(length: int, columns: int) -> int:
        return -(-length // _view_chunk_columns(length, columns))

    assert [n_chunks(length, 1024) for length in lengths] == [149, 38, 10, 3, 1, 1]
    assert n_chunks(lengths[3], 1024) == 3  # level 4, whole-recording view
    assert n_chunks(lengths[5], 1024) == 1  # level 6, the minimap

    # The old seconds-based rule, for the record: chunk_seconds * rate / 4**level.
    def old_columns(level: int) -> int:
        return max(1, int(round(4.0 * 250.0 / 4**level)))

    old = [-(-length // min(length, old_columns(i))) for i, length in enumerate(lengths, start=1)]
    assert old[3] == 594  # level 4
    assert old[5] == 148  # level 6


def test_view_chunk_columns_caps_at_level_length():
    """The shared helper never returns more columns than the level holds, nor 0."""
    assert _view_chunk_columns(5000, 1024) == 1024
    assert _view_chunk_columns(300, 1024) == 300
    assert _view_chunk_columns(1024, 1024) == 1024
    assert _view_chunk_columns(1, 1024) == 1


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


def test_zarr_int16_zerofills_and_flags_non_finite(tmp_path):
    """A NaN/inf in one channel must NOT sink the whole recording (a MoBI dataset's
    aux channel would otherwise take down every good EEG channel with it). int16
    export zero-fills the non-finite samples -- range computed over finite samples
    only -- and flags the channel (`nonfinite_samples`, `usable_for_inference` ->
    False). A clean channel in the same group is untouched."""
    good = np.sin(np.arange(1000) / 5.0)
    bad = np.sin(np.arange(1000) / 5.0)
    bad[100:110] = np.nan  # 10 NaN
    bad[200] = np.inf  # + 1 inf = 11 non-finite
    rec = Recording()
    rec.add_channel("C_good", good, 250, "uV", "EEG")
    rec.add_channel("C_bad", bad, 250, "uV", "EEG")

    root = _open(rec.to_zarr(str(tmp_path / "r")))  # dtype="int16" default; must not raise
    name = next(k for k in root.keys() if k != "events")
    chans = {c["label"]: c for c in root[name].attrs["channels"]}

    # Clean channel: untouched, inference-usable, unflagged.
    assert chans["C_good"]["usable_for_inference"] is True
    assert "nonfinite_samples" not in chans["C_good"]

    # Bad channel: present + viewable, but flagged and demoted.
    assert chans["C_bad"]["nonfinite_samples"] == 11
    assert chans["C_bad"]["usable_for_inference"] is False

    # Every decoded sample is finite (the fill never leaves NaN in int16), and the
    # finite samples still round-trip within one quantization step.
    arr = root[name]["0"][:]
    c = chans["C_bad"]
    decoded = arr[c["row_index"]] * c["scale"] + c["offset"]
    assert np.all(np.isfinite(decoded))
    finite_idx = np.isfinite(bad)
    assert np.allclose(decoded[finite_idx], bad[finite_idx], atol=2 * c["scale"])


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
