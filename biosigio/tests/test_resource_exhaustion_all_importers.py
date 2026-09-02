"""Resource exhaustion must propagate unchanged out of EVERY guarded importer,
not just EDF (issue #123 follow-up review).

One parametrized test drives all eleven guarded catch-alls (see the audit in
biosigio/importers/*.py and biosigio/exceptions.py's is_resource_exhaustion):
edf, eeglab (classic v5/v7), brainvision, mef3, meg, csv, neo, tabular, wfdb,
otb, zarr. For each, a real or on-the-fly-synthesized fixture is loaded through
``Recording.from_file`` while the exact library call that importer's guard
wraps is monkeypatched to raise -- once with ``MemoryError``, once with
``OSError(errno.EMFILE, ...)`` -- and the test asserts the SAME exception
instance propagates out of ``Recording.from_file`` unchanged (never
FileReadError/ValueError/CorruptFileError/UnsupportedFormatError).

NO MOCKS of business logic: every fixture is a real file (either committed
under examples/, or written by a real writer -- pyedflib's EdfWriter, pymef,
neo's own NeoMatlabIO, or biosigio's own to_parquet/to_zarr against a real
committed EDF). Only the ONE library call each guard wraps is monkeypatched,
which simulates the operating system refusing a resource, not a stand-in for
any of biosigio's own read/parse logic. Cases needing an optional extra
(mne, pymef, pyarrow, neo, zarr) skip cleanly when it is absent.

Mutation-verified (see the commit/PR description for the full 11-case matrix):
with an importer's own `is_resource_exhaustion` binding monkeypatched to
always return False -- functionally identical to deleting that importer's
`if is_resource_exhaustion(e): raise` guard, without editing source -- the
corresponding case's assertion fails for the 6 importers that raise ValueError
directly (csv, neo, tabular, wfdb, otb, zarr). For the 5 that route through
`classify_read_error` (edf, eeglab, brainvision, mef3, meg), that alone does
NOT bite -- `classify_read_error`'s own defence-in-depth check saves it, by
design (the review explicitly asked for that second layer); disabling BOTH
`is_resource_exhaustion` bindings does bite. The wfdb-annotation,
meg-find_events, and csv-sniffing tests below were confirmed the same way.
"""

from __future__ import annotations

import errno
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyedflib
import pytest

from biosigio import Recording
from biosigio.exceptions import BiosigIOError

_REPO = Path(__file__).resolve().parents[2]
_EMG_EDF = _REPO / "examples/bids/emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf"
_EEGLAB_CLASSIC_SET = _REPO / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"
_BRAINVISION_VHDR = _REPO / "examples/brainvision/sub-01_task-rest_eeg.vhdr"
_MEG_FIF = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"
_WFDB_DIR = _REPO / "examples"
_OTB_FIXTURE = _REPO / "examples/one_sessantaquattro_truncated.otb+"


def _raiser(exc: BaseException) -> Callable[..., None]:
    """A callable that raises exactly ``exc`` (the same instance), regardless
    of how it is called -- used as a monkeypatch replacement for the library
    call each importer's guard wraps."""

    def _raise(*args, **kwargs):
        raise exc

    return _raise


def _write_minimal_edf(path: Path, n_channels: int = 2, n_samples: int = 500) -> None:
    """A tiny real EDF, written with pyedflib's own writer (no hand-crafted bytes)."""
    w = pyedflib.EdfWriter(str(path), n_channels)
    try:
        w.setSignalHeaders(
            [
                {
                    "label": f"EEG{i}",
                    "dimension": "uV",
                    "sample_frequency": 100.0,
                    "physical_max": 100.0,
                    "physical_min": -100.0,
                    "digital_max": 32767,
                    "digital_min": -32768,
                    "prefilter": "n/a",
                    "transducer": "n/a",
                }
                for i in range(n_channels)
            ]
        )
        w.writeSamples([np.zeros(n_samples) for _ in range(n_channels)])
    finally:
        w.close()


# -- one case-builder per guarded importer --------------------------------------
# Each returns (filepath, from_file kwargs) after writing/locating the fixture
# and monkeypatching the one library call that importer's guard wraps to raise
# `exc`. Skips (pytest.skip) when a needed optional extra/fixture is absent.


def _case_edf(tmp_path, monkeypatch, exc):
    path = tmp_path / "rec.edf"
    _write_minimal_edf(path, n_channels=2)
    monkeypatch.setattr(pyedflib.EdfReader, "readSignal", _raiser(exc))
    return str(path), {"importer": "edf"}


def _case_eeglab_classic(tmp_path, monkeypatch, exc):
    if not _EEGLAB_CLASSIC_SET.exists():
        pytest.skip("EEGLAB classic fixture missing")
    from biosigio.importers import eeglab as eeglab_mod

    # eeglab.py does `from scipy.io import loadmat` -- patching scipy.io.loadmat
    # would NOT affect that already-bound local name, so patch the importer
    # module's own binding instead.
    monkeypatch.setattr(eeglab_mod, "loadmat", _raiser(exc))
    return str(_EEGLAB_CLASSIC_SET), {"importer": "eeglab"}


def _case_brainvision(tmp_path, monkeypatch, exc):
    mne = pytest.importorskip("mne", reason="BrainVision case requires the optional 'meg' extra")
    if not _BRAINVISION_VHDR.exists():
        pytest.skip("BrainVision fixture missing")
    monkeypatch.setattr(mne.io, "read_raw_brainvision", _raiser(exc))
    return str(_BRAINVISION_VHDR), {"importer": "brainvision"}


def _case_mef3(tmp_path, monkeypatch, exc):
    pytest.importorskip("pymef", reason="MEF3 case requires the optional 'mef3' extra (pymef)")
    mne = pytest.importorskip("mne", reason="MEF3 case requires the optional 'mef3' extra (mne)")
    if not hasattr(mne.io, "read_raw_mef"):
        pytest.skip("MEF3 case needs mne>=1.12 (read_raw_mef)")
    # Deferred import: test_mef3_importer's own module-level skips have already
    # been re-checked above, so this will not itself skip/fail here.
    from biosigio.tests.test_mef3_importer import _write_mef3_session

    channels = {"ch01": np.zeros(200, dtype="int32"), "ch02": np.ones(200, dtype="int32")}
    path = tmp_path / "sub-01_task-test_ieeg.mefd"
    _write_mef3_session(path, channels, 200.0)
    monkeypatch.setattr(mne.io, "read_raw_mef", _raiser(exc))
    return str(path), {"importer": "mef3"}


def _case_meg(tmp_path, monkeypatch, exc):
    mne = pytest.importorskip("mne", reason="MEG case requires the optional 'meg' extra")
    if not _MEG_FIF.exists():
        pytest.skip("MEG FIF fixture missing")
    monkeypatch.setattr(mne.io, "read_raw_fif", _raiser(exc))
    return str(_MEG_FIF), {"importer": "meg"}


def _case_csv(tmp_path, monkeypatch, exc):
    import pandas as pd

    # Derived from the real committed EMG EDF (per review), not hand-typed rows.
    rec = Recording.from_file(str(_EMG_EDF))
    path = tmp_path / "emg.csv"
    rec.signals.to_csv(path)
    monkeypatch.setattr(pd, "read_csv", _raiser(exc))
    return str(path), {"importer": "csv"}


def _case_neo(tmp_path, monkeypatch, exc):
    neo = pytest.importorskip("neo", reason="neo case requires the optional 'neo' extra")
    pytest.importorskip("quantities", reason="neo case requires 'quantities' (pulled in by neo)")
    import quantities as pq

    from biosigio.tests.test_neo_importer import _write_block

    path = tmp_path / "stream.mat"
    data = np.zeros((100, 2), dtype="float64")
    _write_block(path, [("s1", data, 100.0, pq.uV, 0.0)])
    monkeypatch.setattr(neo.io.NeoMatlabIO, "read_block", _raiser(exc))
    return str(path), {"importer": "neo"}


def _case_tabular(tmp_path, monkeypatch, exc):
    pytest.importorskip("pyarrow", reason="tabular case requires the optional 'arrow' extra")
    import pyarrow.parquet as pq

    rec = Recording.from_file(str(_EMG_EDF))
    path = tmp_path / "emg.parquet"
    rec.to_parquet(str(path))
    monkeypatch.setattr(pq, "read_table", _raiser(exc))
    return str(path), {"importer": "tabular"}


def _case_wfdb(tmp_path, monkeypatch, exc):
    import wfdb

    for suffix in (".hea", ".dat", ".atr"):
        shutil.copy(_WFDB_DIR / f"100{suffix}", tmp_path / f"100{suffix}")
    monkeypatch.setattr(wfdb, "rdrecord", _raiser(exc))
    return str(tmp_path / "100.hea"), {"importer": "wfdb"}


def _case_otb(tmp_path, monkeypatch, exc):
    import subprocess

    if not _OTB_FIXTURE.exists():
        pytest.skip("OTB fixture missing")
    monkeypatch.setattr(subprocess, "run", _raiser(exc))
    return str(_OTB_FIXTURE), {"importer": "otb"}


def _case_zarr(tmp_path, monkeypatch, exc):
    zarr = pytest.importorskip("zarr", reason="zarr case requires the optional 'zarr' extra")

    rec = Recording.from_file(str(_EMG_EDF))
    path = tmp_path / "emg.zarr"
    written = rec.to_zarr(str(path))
    monkeypatch.setattr(zarr, "open_group", _raiser(exc))
    return written, {"importer": "zarr"}


CASES: list[tuple[str, Callable]] = [
    ("edf", _case_edf),
    ("eeglab_classic", _case_eeglab_classic),
    ("brainvision", _case_brainvision),
    ("mef3", _case_mef3),
    ("meg", _case_meg),
    ("csv", _case_csv),
    ("neo", _case_neo),
    ("tabular", _case_tabular),
    ("wfdb", _case_wfdb),
    ("otb", _case_otb),
    ("zarr", _case_zarr),
]

_EXCEPTIONS = [
    pytest.param(
        lambda: MemoryError(
            "Unable to allocate 24.8 MiB for an array with shape (3248000,) and data type float64"
        ),
        id="memory",
    ),
    pytest.param(
        lambda: OSError(errno.EMFILE, "Too many open files"),
        id="emfile",
    ),
]


@pytest.mark.parametrize("exc_factory", _EXCEPTIONS)
@pytest.mark.parametrize("case_id,case_fn", CASES, ids=[c[0] for c in CASES])
def test_resource_exhaustion_propagates_from_every_importer(
    tmp_path, monkeypatch, case_id, case_fn, exc_factory
):
    exc = exc_factory()
    filepath, load_kwargs = case_fn(tmp_path, monkeypatch, exc)

    with pytest.raises(type(exc)) as exc_info:
        Recording.from_file(filepath, **load_kwargs)

    # The exact same instance, not a re-typed copy -- every guard does a bare
    # `raise`, which preserves identity end to end.
    assert exc_info.value is exc
    assert not isinstance(exc_info.value, BiosigIOError)


# -- inner guards: annotation/events/sniffing helpers that used to swallow -----
# resource exhaustion into a "successful" Recording instead of failing the load
# (review items 1-3; the outer per-importer guards above do not exercise these
# nested try/except blocks, since they are not on the primary read path).


def test_wfdb_annotation_exhaustion_propagates(tmp_path, monkeypatch):
    """wfdb.rdann raising resource exhaustion must fail the load, not be
    written into an `annotation_error` metadata field on a "successful"
    Recording (the pre-fix behavior for issue #123 review item 1)."""
    import wfdb

    from biosigio.importers.wfdb import WFDBImporter

    for suffix in (".hea", ".dat", ".atr"):
        shutil.copy(_WFDB_DIR / f"100{suffix}", tmp_path / f"100{suffix}")

    exc = MemoryError("Unable to allocate array for WFDB annotations")
    monkeypatch.setattr(wfdb, "rdann", _raiser(exc))

    with pytest.raises(MemoryError) as exc_info:
        WFDBImporter().load(str(tmp_path / "100.hea"))
    assert exc_info.value is exc


def test_meg_find_events_exhaustion_propagates(monkeypatch):
    """mne.find_events raising a thread-exhaustion RuntimeError must fail the
    load, not be absorbed as "no stim channels found" -> empty events on a
    "successful" Recording (review item 2). RuntimeError (not MemoryError) to
    also exercise the except (ValueError, RuntimeError) clause's own type."""
    mne = pytest.importorskip("mne", reason="MEG case requires the optional 'meg' extra")
    if not _MEG_FIF.exists():
        pytest.skip("MEG FIF fixture missing")

    exc = RuntimeError("can't start new thread")
    monkeypatch.setattr(mne, "find_events", _raiser(exc))

    with pytest.raises(RuntimeError) as exc_info:
        Recording.from_file(str(_MEG_FIF), importer="meg")
    assert exc_info.value is exc


def test_csv_specialized_format_sniff_exhaustion_propagates(tmp_path, monkeypatch):
    """A resource-exhaustion OSError while sniffing for a specialized-format
    signature (_detect_specialized_format, review item 3's first site) must
    propagate, not be swallowed into "assume generic CSV, guess the rest"."""
    from biosigio.importers.csv import CSVImporter

    path = tmp_path / "generic.csv"
    path.write_text("a,b\n1,2\n3,4\n")

    exc = OSError(errno.EMFILE, "Too many open files")
    real_open = open

    def flaky_open(file, *args, **kwargs):
        if str(file) == str(path):
            raise exc
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", flaky_open)

    with pytest.raises(OSError) as exc_info:
        CSVImporter().load(str(path))
    assert exc_info.value is exc


def test_csv_structure_analysis_exhaustion_propagates(tmp_path, monkeypatch):
    """A resource-exhaustion error while analyzing CSV structure
    (_analyze_csv_structure, review item 3's second site) must propagate, not
    be swallowed into hard-coded parse defaults."""
    from biosigio.importers.csv import CSVImporter

    path = tmp_path / "generic.csv"
    path.write_text("a,b\n1,2\n3,4\n")

    exc = MemoryError("Unable to allocate array while analyzing CSV structure")
    monkeypatch.setattr(CSVImporter, "_is_numeric", _raiser(exc))

    with pytest.raises(MemoryError) as exc_info:
        CSVImporter().load(str(path))
    assert exc_info.value is exc
