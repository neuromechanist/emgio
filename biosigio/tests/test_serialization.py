"""Cross-format serialization-consolidation tests (0.6.0).

NO MOCKS: real Recordings (synthetic real signals) and real files/stores. Covers
the behaviors shared by the Parquet/Arrow/Zarr serialization story: source_format
provenance, Zarr ``recording_metadata`` as a native object (with v1 JSON-string
back-compat), Zarr ``format_version`` validation, ``.zarr`` trailing-slash
inference, and the unified empty-signal error message.
"""

import datetime
import pathlib

import numpy as np
import pytest

from biosigio import Recording

_REPO = pathlib.Path(__file__).resolve().parents[2]
EMG_EDF = _REPO / "examples/bids/emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf"
_STARTDATE = datetime.datetime(2026, 5, 31, 9, 30, 0)


def _rec():
    rec = Recording()
    rec.add_channel("C1", np.sin(np.arange(500) / 7.0), 250, "uV", "EEG")
    rec.set_metadata("startdate", _STARTDATE)
    return rec


# --- source_format provenance (set centrally in from_file) ---


@pytest.mark.skipif(not EMG_EDF.exists(), reason="EMG fixture missing")
def test_from_file_sets_source_format():
    rec = Recording.from_file(str(EMG_EDF))
    assert rec.metadata["source_format"] == "edf"


def test_fresh_tabular_import_gets_tabular_source_format(tmp_path):
    pytest.importorskip("pyarrow")
    out = tmp_path / "r.parquet"
    _rec().to_parquet(str(out))
    rt = Recording.from_file(str(out))
    assert rt.metadata["source_format"] == "tabular"


def test_reimport_preserves_original_source_format(tmp_path):
    pytest.importorskip("pyarrow")
    rec = _rec()
    rec.set_metadata("source_format", "edf")  # as if loaded from EDF then re-serialized
    out = tmp_path / "r.parquet"
    rec.to_parquet(str(out))
    rt = Recording.from_file(str(out))
    assert rt.metadata["source_format"] == "edf"  # not relabeled to "tabular"


# --- Zarr recording_metadata as a native, browser-readable object ---


def test_zarr_recording_metadata_is_native_object(tmp_path):
    zarr = pytest.importorskip("zarr")
    out = _rec().to_zarr(str(tmp_path / "r"))
    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="r")
    md = dict(root.attrs)["recording_metadata"]
    assert isinstance(md, dict)  # a native object, not a JSON-encoded string
    # datetime is a typed envelope a zarrita/JS reader can detect and decode
    assert md["startdate"]["__biosigio_type__"] == "datetime"

    rt = Recording.from_file(out)
    assert rt.metadata["startdate"] == _STARTDATE
    assert type(rt.metadata["startdate"]) is datetime.datetime


def test_zarr_reads_legacy_v1_string_metadata(tmp_path):
    """A pre-0.6 store wrote recording_metadata as a JSON string; still reads."""
    zarr = pytest.importorskip("zarr")
    from biosigio.tabular_schema import metadata_to_json

    rec = _rec()
    out = rec.to_zarr(str(tmp_path / "r"))
    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="a")
    root.attrs["recording_metadata"] = metadata_to_json(rec.metadata)  # v1 form
    root.attrs["format_version"] = 1
    rt = Recording.from_file(out)
    assert rt.metadata["startdate"] == _STARTDATE


def test_zarr_rejects_future_format_version(tmp_path):
    zarr = pytest.importorskip("zarr")
    out = _rec().to_zarr(str(tmp_path / "r"))
    root = zarr.open_group(store=zarr.storage.LocalStore(out), mode="a")
    root.attrs["format_version"] = 99
    with pytest.raises(ValueError, match="Unsupported biosigIO Zarr store version"):
        Recording.from_file(out)


# --- .zarr directory inference robust to a trailing slash ---


def test_zarr_trailing_slash_infers_importer(tmp_path):
    pytest.importorskip("zarr")
    out = _rec().to_zarr(str(tmp_path / "r"))
    rt = Recording.from_file(out + "/")  # directory path with trailing separator
    assert isinstance(rt, Recording)
    assert rt.signals is not None and "C1" in rt.signals.columns


def test_infer_importer_zarr_trailing_slash():
    assert Recording._infer_importer("store.zarr/") == "zarr"
    assert Recording._infer_importer("store.zarr") == "zarr"


# --- unified empty-signal error message across the serialization writers ---


def test_empty_recording_to_parquet_unified_message(tmp_path):
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="No signals loaded"):
        Recording().to_parquet(str(tmp_path / "x.parquet"))


def test_empty_recording_to_arrow_unified_message(tmp_path):
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="No signals loaded"):
        Recording().to_arrow(str(tmp_path / "x.feather"))


def test_empty_recording_to_zarr_unified_message(tmp_path):
    pytest.importorskip("zarr")
    with pytest.raises(ValueError, match="No signals loaded"):
        Recording().to_zarr(str(tmp_path / "x"))
