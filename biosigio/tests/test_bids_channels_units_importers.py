"""BIDS unit adoption across the real importers (issue #122).

``test_bids_channels_units.py`` pins the rule on BrainVision; this pins it on
every other importer whose native unit differs from what a sidecar may declare,
because the whole defect was an importer-specific assumption (pyedflib already
reports the sidecar's unit, MNE never does) that held for the format the feature
was written against and for no other.

NO MOCKS. Every recording here comes off disk through the real importer: the
committed BIDS fixtures under ``examples/`` where one exists, a real ``.fif``
written by MNE's own writer where the committed fixtures have no planar
gradiometer, and a real EDF written by ``pyedflib`` and byte-patched for the
tolerant-fallback path (the same construction ``test_edf_fallback.py`` uses,
whose helpers are reused rather than re-derived).
"""

import os
import pathlib
import shutil

import numpy as np
import pytest

from biosigio import Recording
from biosigio.bids import apply_channels_tsv

_REPO = pathlib.Path(__file__).resolve().parents[2]
MEG_FIF = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"
MEG_TSV = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_channels.tsv"
CTF_DS = _REPO / "examples/ctf/catch-alp-good-f.ds"
EEG_SET = _REPO / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"
IEEG_EDF = (
    _REPO
    / "examples/bids/ieeg/sub-01/ses-postimp/ieeg/sub-01_ses-postimp_task-stim_run-08_ieeg.edf"
)


def link_beside(source: pathlib.Path, directory: pathlib.Path, name: str) -> pathlib.Path:
    """Put ``source`` in ``directory`` under ``name`` without copying MBs of fixture.

    A symlink keeps a 3.6 MB ``.fif`` (or a CTF directory) out of every tmp_path,
    and the importers only ever read it. Falls back to a real copy where symlinks
    are unavailable.
    """
    target = directory / name
    try:
        os.symlink(source, target)
    except (OSError, NotImplementedError):
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return target


def rewrite_units(
    source_tsv: pathlib.Path, destination: pathlib.Path, replacements: dict[str, str]
) -> pathlib.Path:
    """Copy a real ``channels.tsv``, overriding the ``units`` cell of named rows.

    Keeps every other column and row of the committed sidecar intact, so what is
    under test is one edited declaration against a real file, not a hand-built
    table that might differ from the real one in some other way.

    Args:
        source_tsv: The committed sidecar to start from.
        destination: Path to write the edited sidecar to.
        replacements: ``{channel name: units}``; a name absent here is untouched.

    Returns:
        ``destination``.
    """
    lines = source_tsv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    name_at, units_at = header.index("name"), header.index("units")
    out = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        if cells[name_at] in replacements:
            cells[units_at] = replacements[cells[name_at]]
        out.append("\t".join(cells))
    destination.write_text("\n".join(out) + "\n", encoding="utf-8")
    return destination


# --------------------------------------------------------------------------
# MEG family: MNE reports SI teslas; a sidecar in fT is a 10^15 conversion, and
# the stim channels sharing the file must not move at all.
# --------------------------------------------------------------------------

pytest.importorskip("mne", reason="the MEG/BrainVision importers need the 'meg' extra (mne)")


@pytest.fixture(scope="module")
def meg_native():
    if not MEG_FIF.exists():
        pytest.skip("MEG BIDS fixture missing")
    return Recording.from_file(str(MEG_FIF), bids_channels="off")


def test_meg_sidecar_in_femtotesla_converts_and_spares_the_triggers(meg_native, tmp_path):
    """The committed sidecar says T and V; declaring fT must move only the T rows."""
    fif = link_beside(MEG_FIF, tmp_path, "sub-01_task-mouse_meg.fif")
    magnetometers = [
        label
        for label, info in meg_native.channels.items()
        if info["channel_type"] in {"MEGMAG", "MEGREFMAG"}
    ]
    rewrite_units(
        MEG_TSV,
        tmp_path / "sub-01_task-mouse_channels.tsv",
        dict.fromkeys(magnetometers, "fT"),
    )

    rec = Recording.from_file(str(fif))

    # Every magnetometer: T -> fT, values up by 1e15, label updated.
    for label in magnetometers:
        assert meg_native.channels[label]["physical_dimension"] == "T"
        assert rec.channels[label]["physical_dimension"] == "fT"
        assert "bids_unit" not in rec.channels[label]
    sample = magnetometers[0]
    assert np.allclose(
        rec.signals[sample].to_numpy(), meg_native.signals[sample].to_numpy() * 1e15, rtol=1e-9
    )
    assert float(rec.signals[sample].std()) == pytest.approx(
        float(meg_native.signals[sample].std()) * 1e15, rel=1e-9
    )

    # The two TRIG rows still declare V while MNE also reports V: untouched.
    for label in ("UPPT001", "UPPT002"):
        assert rec.channels[label]["physical_dimension"] == "V"
        assert "bids_unit" not in rec.channels[label]
        assert np.array_equal(rec.signals[label].to_numpy(), meg_native.signals[label].to_numpy())

    report = rec.metadata["channels_tsv_units"]
    assert report["units_column_present"] is True
    assert report["converted"] == len(magnetometers)
    assert report["kept_importer_unit"] == 0


def test_a_millivolt_sidecar_cannot_corrupt_meg_trigger_codes(meg_native, tmp_path):
    """The reproduction from review: codes 0/11/22/33/44/55 must not become 0/11000/...

    ``UPPT001`` is a real trigger channel in the committed MEG fixture, typed
    TRIG and labelled V by MNE's FIFF unit code while holding integer codes.
    """
    fif = link_beside(MEG_FIF, tmp_path, "sub-01_task-mouse_meg.fif")
    rewrite_units(MEG_TSV, tmp_path / "sub-01_task-mouse_channels.tsv", {"UPPT001": "mV"})

    rec = Recording.from_file(str(fif))

    codes = np.unique(meg_native.signals["UPPT001"].to_numpy())
    assert len(codes) > 1, "fixture no longer carries distinct trigger codes"
    assert np.array_equal(np.unique(rec.signals["UPPT001"].to_numpy()), codes)
    assert rec.channels["UPPT001"]["physical_dimension"] == "V"
    assert rec.channels["UPPT001"]["bids_unit"] == "mV"
    assert rec.metadata["channels_tsv_units"]["kept_importer_unit"] == 1


def test_planar_gradiometers_convert_from_tesla_per_metre(tmp_path):
    """T/m -> fT/cm, the one MEG conversion no committed fixture exercises.

    The committed ``.fif`` and CTF fixtures hold only axial magnetometers (MNE
    type ``mag``), so a real planar-gradiometer file is written here with MNE's
    own writer and read back through biosigIO's importer.
    """
    import mne

    rng = np.random.default_rng(122)
    data = rng.normal(0, 5e-12, (3, 200))  # T/m for the grads, V for the EEG row
    info = mne.create_info(["MEG0112", "MEG0113", "EEG001"], 500.0, ["grad", "grad", "eeg"])
    fif = tmp_path / "sub-01_task-grad_meg.fif"
    mne.io.RawArray(data, info, verbose="ERROR").save(str(fif), verbose="ERROR")

    native = Recording.from_file(str(fif), bids_channels="off")
    assert native.channels["MEG0112"]["physical_dimension"] == "T/m"
    assert native.channels["EEG001"]["physical_dimension"] == "V"

    (tmp_path / "sub-01_task-grad_channels.tsv").write_text(
        "name\ttype\tunits\n"
        "MEG0112\tMEGGRADPLANAR\tfT/cm\n"
        "MEG0113\tMEGGRADPLANAR\tfT/cm\n"
        "EEG001\tEEG\tV\n",
        encoding="utf-8",
    )

    rec = Recording.from_file(str(fif))

    # 1 T/m = 1e15 fT / 1e2 cm = 1e13 fT/cm.
    for label in ("MEG0112", "MEG0113"):
        assert rec.channels[label]["physical_dimension"] == "fT/cm"
        assert np.allclose(
            rec.signals[label].to_numpy(), native.signals[label].to_numpy() * 1e13, rtol=1e-9
        )
    # The EEG row already agreed with the importer: untouched, and not counted.
    assert rec.channels["EEG001"]["physical_dimension"] == "V"
    assert np.array_equal(rec.signals["EEG001"].to_numpy(), native.signals["EEG001"].to_numpy())
    assert rec.metadata["channels_tsv_units"]["converted"] == 2


@pytest.mark.skipif(not CTF_DS.exists(), reason="CTF fixture missing")
def test_ctf_meg_and_eeg_channels_move_independently(tmp_path):
    """A mixed CTF recording: teslas convert, the volts rows in the same file do not."""
    # A CTF .ds names its internal files after the directory stem, so it cannot
    # be renamed to a BIDS stem; the sidecar takes the directory's own name
    # instead, which find_channels_tsv resolves through its full-stem candidate.
    ds = link_beside(CTF_DS, tmp_path, CTF_DS.name)
    native = Recording.from_file(str(ds), bids_channels="off")

    rows = ["name\ttype\tunits"]
    for label, info in native.channels.items():
        unit = "fT" if info["channel_type"] in {"MEGMAG", "MEGREFMAG"} else "V"
        rows.append(f"{label}\t{info['channel_type']}\t{unit}")
    (tmp_path / f"{CTF_DS.stem}_channels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rec = Recording.from_file(str(ds))

    magnetometer = next(
        label for label, info in native.channels.items() if info["channel_type"] == "MEGMAG"
    )
    eeg = next(label for label, info in native.channels.items() if info["channel_type"] == "EEG")
    assert rec.channels[magnetometer]["physical_dimension"] == "fT"
    assert np.allclose(
        rec.signals[magnetometer].to_numpy(),
        native.signals[magnetometer].to_numpy() * 1e15,
        rtol=1e-9,
    )
    assert rec.channels[eeg]["physical_dimension"] == "V"
    assert np.array_equal(rec.signals[eeg].to_numpy(), native.signals[eeg].to_numpy())


# --------------------------------------------------------------------------
# EEGLAB: hardcodes "uV" (a .set carries no unit field) and loads at float32.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not EEG_SET.exists(), reason="EEG BIDS fixture missing")
def test_eeglab_sidecar_in_millivolts_scales_down_and_keeps_float32(tmp_path):
    """uV -> mV is 1e-3, and the memory-saving float32 must survive it."""
    set_file = link_beside(EEG_SET, tmp_path, "sub-01_task-eyesopen_eeg.set")
    native = Recording.from_file(str(EEG_SET), bids_channels="off")
    label = next(iter(native.channels))
    assert native.channels[label]["physical_dimension"] == "uV"
    assert native.signals[label].dtype == np.float32

    rewrite_units(
        EEG_SET.with_name("sub-01_task-eyesopen_channels.tsv"),
        tmp_path / "sub-01_task-eyesopen_channels.tsv",
        dict.fromkeys(native.channels, "mV"),
    )

    rec = Recording.from_file(str(set_file))

    assert rec.channels[label]["physical_dimension"] == "mV"
    assert rec.signals[label].dtype == np.float32
    assert np.allclose(
        rec.signals[label].to_numpy(), native.signals[label].to_numpy() * 1e-3, rtol=1e-5
    )
    assert rec.metadata["channels_tsv_units"]["converted"] == len(native.channels)


# --------------------------------------------------------------------------
# CSV: unit comes from a caller argument or a per-type default, and a column of
# whole numbers arrives as int64.
# --------------------------------------------------------------------------


def write_csv(directory: pathlib.Path, stem: str, values: np.ndarray) -> pathlib.Path:
    """Write a real generic CSV with one integer-valued channel column."""
    path = directory / f"{stem}_eeg.csv"
    lines = ["time,EEG1"] + [f"{i / 100.0},{int(v)}" for i, v in enumerate(values)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def csv_recording(tmp_path):
    """A real CSV read through the CSV importer, with an int64 signal column."""
    counts = np.array([10, -20, 30, 0, -40, 127], dtype=np.int64)
    path = write_csv(tmp_path, "sub-01_task-rest", counts)
    rec = Recording.from_file(
        str(path), importer="csv", force_csv=True, physical_dimensions={"EEG1": "µV"}
    )
    return rec, counts, path


def test_csv_spelling_only_sidecar_relabels_an_integer_column(csv_recording, tmp_path):
    """µV -> uV is ratio 1: the label moves, the int64 column does not."""
    rec, counts, path = csv_recording
    assert rec.channels["EEG1"]["physical_dimension"] == "µV"
    assert rec.signals["EEG1"].dtype == np.int64

    (tmp_path / "sub-01_task-rest_channels.tsv").write_text(
        "name\ttype\tunits\nEEG1\tEEG\tuV\n", encoding="utf-8"
    )
    assert apply_channels_tsv(rec, str(tmp_path / "sub-01_task-rest_channels.tsv")) == 1

    assert rec.channels["EEG1"]["physical_dimension"] == "uV"
    assert rec.signals["EEG1"].dtype == np.int64
    assert np.array_equal(rec.signals["EEG1"].to_numpy(), counts)
    assert rec.metadata["channels_tsv_units"]["relabelled"] == 1
    assert rec.metadata["channels_tsv_units"]["converted"] == 0


def test_csv_millivolt_sidecar_promotes_the_integer_column(csv_recording, tmp_path):
    """µV -> mV is 1e-3, so int64 must become float64 rather than truncate."""
    rec, counts, _ = csv_recording

    (tmp_path / "sub-01_task-rest_channels.tsv").write_text(
        "name\ttype\tunits\nEEG1\tEEG\tmV\n", encoding="utf-8"
    )
    assert apply_channels_tsv(rec, str(tmp_path / "sub-01_task-rest_channels.tsv")) == 1

    assert rec.channels["EEG1"]["physical_dimension"] == "mV"
    assert rec.signals["EEG1"].dtype == np.float64
    assert np.allclose(rec.signals["EEG1"].to_numpy(), counts * 1e-3, rtol=1e-12)
    assert rec.metadata["channels_tsv_units"]["converted"] == 1


# --------------------------------------------------------------------------
# EDF: the format the feature was written against, and the one place a unit
# string can arrive mojibaked.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not IEEG_EDF.exists(), reason="iEEG BIDS fixture missing")
def test_the_pyedflib_mojibake_micro_sign_relabels_without_rescaling(tmp_path):
    """A non-UTF-8 EDF header yields "\\x83\\xcaV"; it means uV and must not rescale.

    The committed EDF fixtures all carry clean ASCII dimensions, so the mangled
    string is set on a channel of a genuinely EDF-imported recording rather than
    read from a fixture that reproduces the encoding -- the parsing, the ratio
    and the relabel are all still the production path.
    """
    rec = Recording.from_file(str(IEEG_EDF), bids_channels="off")
    label = next(iter(rec.channels))
    before = rec.signals[label].to_numpy().copy()
    rec.set_channel(label, physical_dimension="\x83\xcaV")

    (tmp_path / "sub-01_channels.tsv").write_text(
        f"name\ttype\tunits\n{label}\tSEEG\tuV\n", encoding="utf-8"
    )
    assert apply_channels_tsv(rec, str(tmp_path / "sub-01_channels.tsv")) == 1

    assert rec.channels[label]["physical_dimension"] == "uV"
    assert "bids_unit" not in rec.channels[label]
    assert np.array_equal(rec.signals[label].to_numpy(), before)
    assert rec.metadata["channels_tsv_units"]["relabelled"] == 1


def test_tolerant_edf_fallback_is_a_no_op_when_the_sidecar_matches(tmp_path):
    """The MNE-backed EDF fallback rescales to the header unit, so uV/uV moves nothing.

    Reuses ``test_edf_fallback``'s real-file construction (pyedflib writes a
    compliant EDF, then one physical_max field is patched to make pyedflib refuse
    it) so this exercises the same bytes the fallback tests do.
    """
    from biosigio.tests.test_edf_fallback import _channel, _patch_signal_field, _write_edf

    path = str(tmp_path / "sub-01_task-rest_eeg.edf")
    rng = np.random.default_rng(122)
    channels = [_channel("C0", 100.0, -200.0, 200.0), _channel("REF", 100.0, -200.0, 200.0)]
    data = [rng.uniform(-200, 200, 300), rng.uniform(-200, 200, 300)]
    _write_edf(path, channels, data)
    _patch_signal_field(path, 1, "physical_max", b"-200")  # forces the tolerant read

    native = Recording.from_file(path, importer="edf", bids_channels="off")
    assert native.metadata["edf_tolerant_read"] is True
    assert native.channels["C0"]["physical_dimension"] == "uV"

    (tmp_path / "sub-01_task-rest_channels.tsv").write_text(
        "name\ttype\tunits\nC0\tEEG\tuV\nREF\tEEG\tuV\n", encoding="utf-8"
    )
    rec = Recording.from_file(path, importer="edf")

    assert rec.metadata["edf_tolerant_read"] is True
    assert rec.channels["C0"]["physical_dimension"] == "uV"
    assert np.array_equal(rec.signals["C0"].to_numpy(), native.signals["C0"].to_numpy())
    assert rec.metadata["channels_tsv_units"]["converted"] == 0
    assert rec.metadata["channels_tsv_units"]["kept_importer_unit"] == 0
