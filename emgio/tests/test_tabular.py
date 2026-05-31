"""Parquet / Arrow round-trip tests for the biosigIO tabular schema.

Columnar storage is lossless (no quantization like EDF), so the round-trip must
be bit-exact on signals and fully preserve the time index, per-channel metadata,
and events via the self-describing ``biosigio`` schema blob. NO MOCKS: real
fixture, real pyarrow files. Skips if the optional ``arrow`` extra is absent.
"""

import pathlib

import numpy as np
import pytest

from emgio import Recording

pytest.importorskip("pyarrow", reason="tabular serialization requires the optional 'arrow' extra")

_REPO = pathlib.Path(__file__).resolve().parents[2]
EMG_EDF = _REPO / "examples/bids/emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf"

requires_emg = pytest.mark.skipif(not EMG_EDF.exists(), reason="EMG fixture missing")


@requires_emg
@pytest.mark.parametrize("ext,method", [("parquet", "to_parquet"), ("feather", "to_arrow")])
def test_tabular_roundtrip_is_lossless(tmp_path, ext, method):
    rec = Recording.from_file(str(EMG_EDF))
    rec.add_event(onset=0.5, duration=0.1, description="m1")
    rec.add_event(onset=1.0, duration=0.0, description="m2")

    out = tmp_path / f"r.{ext}"
    assert getattr(rec, method)(str(out)) == str(out)
    rt = Recording.from_file(str(out))

    # Signals are bit-exact (columnar storage is lossless), with the time index.
    assert np.array_equal(rec.signals.to_numpy(), rt.signals.to_numpy())
    assert np.allclose(rec.signals.index.to_numpy(), rt.signals.index.to_numpy())
    assert list(rt.signals.columns) == list(rec.signals.columns)

    # Per-channel metadata preserved.
    for ch in rec.channels:
        for key in ("channel_type", "modality", "physical_dimension", "sample_frequency"):
            assert rt.channels[ch][key] == rec.channels[ch][key]

    # Events preserved (sorted, with values).
    assert list(rt.events["description"]) == ["m1", "m2"]
    assert np.allclose(rt.events["onset"].to_numpy(), [0.5, 1.0])
    assert rt.events["onset"].dtype == np.float64

    # Recording metadata preserved with TYPES intact (e.g. startdate stays a
    # datetime, not silently coerced to a string).
    for key, value in rec.metadata.items():
        assert key in rt.metadata, f"metadata key {key!r} dropped"
        assert type(rt.metadata[key]) is type(value), f"metadata[{key!r}] type changed"
        assert rt.metadata[key] == value, f"metadata[{key!r}] value changed"


@requires_emg
def test_tabular_explicit_importer_and_no_modality_creep(tmp_path):
    rec = Recording.from_file(str(EMG_EDF))
    out = tmp_path / "r.parquet"
    rec.to_parquet(str(out))
    rt = Recording.from_file(str(out), importer="tabular")
    assert len(rt.channels) == len(rec.channels)
    # The reconstructed object is a Recording with the same channel set.
    assert isinstance(rt, Recording)
    assert set(rt.channels) == set(rec.channels)


def test_tabular_roundtrip_with_no_events(tmp_path):
    """A recording with no events round-trips to an empty (typed) events frame."""
    rec = Recording()
    rec.add_channel("C1", np.sin(np.arange(500) / 10.0), 100, "uV", "EEG")
    out = tmp_path / "noevents.parquet"
    rec.to_parquet(str(out))
    rt = Recording.from_file(str(out))
    assert rt.events.empty
    assert list(rt.events.columns) == ["onset", "duration", "description"]
    assert np.array_equal(rec.signals.to_numpy(), rt.signals.to_numpy())


def test_non_serializable_metadata_raises(tmp_path):
    """Unexpected non-primitive metadata must raise, not be silently str()-ified."""
    rec = Recording()
    rec.add_channel("C1", np.zeros(10), 100, "uV", "EEG")
    rec.set_metadata("weird", object())  # not JSON/datetime/numpy serializable
    with pytest.raises(TypeError, match="not JSON-"):
        rec.to_parquet(str(tmp_path / "x.parquet"))


def test_plain_parquet_is_rejected(tmp_path):
    """A non-biosigIO parquet (no schema blob) must be rejected with a clear error."""
    import pandas as pd

    path = tmp_path / "plain.parquet"
    pd.DataFrame({"a": [1.0, 2.0, 3.0]}).to_parquet(path)
    with pytest.raises(ValueError, match="biosigIO tabular"):
        Recording.from_file(str(path))
