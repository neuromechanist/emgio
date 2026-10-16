"""BrainVision reader scenarios on a real one-minute exemplar.

``examples/one_minute/brainvision/one_minute.{vhdr,vmrk,eeg}`` is a 60 second, 8 channel,
250 Hz, multiplexed ``IEEE_FLOAT_32`` excerpt of the repository's EEG fixture, with three
markers (provenance, license and derivation are in ``NOTICE.md`` beside it).
Every test reads the real files, or a ``tmp_path`` copy of them that a scenario has broken
on disk; nothing is mocked and no stand-in data is synthesized.

What is pinned here is what the reader does today, verified by running it:

* the exemplar reads exactly as MNE reads it, markers included;
* a missing data file is a typed ``FileReadError`` that names the file;
* a stale ``DataFile=`` is recovered from a same-stem sibling (issue #132), and a different
  stem is never guessed;
* a missing marker file reads the data without events on MNE >= 1.13 and is a typed error
  below that;
* a data file cut mid-frame is refused with ``CorruptFileError`` (code ``corrupt_or_truncated``)
  on both read paths, the in-memory importer and the streaming source, naming the file;
* an extension that differs from the header only in case is resolved by the OS on a
  case-insensitive filesystem and by biosigio's sibling lookup on a case-sensitive one.

Skips cleanly if the optional ``meg`` extra (mne) is not installed.
"""

import pathlib
import shutil

import numpy as np
import pytest

mne = pytest.importorskip(
    "mne", reason="BrainVision import requires the optional 'meg' extra (mne)"
)

from biosigio import Recording  # noqa: E402
from biosigio.exceptions import CorruptFileError, FileReadError  # noqa: E402
from biosigio.exporters.zarr_stream import _open_stream_source  # noqa: E402
from biosigio.importers.brainvision import HEADER_RECOVERED_KEY  # noqa: E402

EXEMPLAR_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples/one_minute/brainvision"
VHDR = EXEMPLAR_DIR / "one_minute.vhdr"
TRIPLET = ("one_minute.vhdr", "one_minute.vmrk", "one_minute.eeg")

SFREQ = 250.0
N_SAMPLES = 15000  # 60 s at 250 Hz
CHANNELS = ["FP1", "AF7", "FPZ", "FP2", "AF8", "F9", "F7", "F5"]
BYTES_PER_FRAME = len(CHANNELS) * 4  # IEEE_FLOAT_32, multiplexed
MARKERS = [
    (0.5, "Stimulus/eyes_open_start"),
    (20.0, "Stimulus/fixation_check"),
    (40.0, "Stimulus/rest_marker"),
]

# MNE >= 1.13 reads a recording whose marker file is absent; 1.12.x raises FileNotFoundError.
_MNE_TOLERATES_MISSING_MARKER = tuple(int(p) for p in mne.__version__.split(".")[:2]) >= (1, 13)


@pytest.fixture(scope="module")
def exemplar() -> Recording:
    """The exemplar read once through biosigio's public entry point."""
    return Recording.from_file(str(VHDR))


@pytest.fixture
def triplet(tmp_path) -> pathlib.Path:
    """A private copy of the exemplar triplet that a test is free to break."""
    for name in TRIPLET:
        shutil.copy2(EXEMPLAR_DIR / name, tmp_path / name)
    return tmp_path


def _read(directory: pathlib.Path) -> Recording:
    return Recording.from_file(str(directory / "one_minute.vhdr"))


def _set_data_file(directory: pathlib.Path, name: str) -> None:
    """Rewrite the header's ``DataFile=`` to ``name``, every other byte kept."""
    vhdr = directory / "one_minute.vhdr"
    text = vhdr.read_text(encoding="utf-8")
    assert "DataFile=one_minute.eeg\n" in text
    vhdr.write_text(
        text.replace("DataFile=one_minute.eeg\n", f"DataFile={name}\n"),
        encoding="utf-8",
        newline="",
    )


def _snapshot(directory: pathlib.Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir()}


def _case_insensitive(directory: pathlib.Path) -> bool:
    """Whether ``directory`` sits on a filesystem that folds case (macOS and Windows defaults)."""
    probe = directory / "CaseProbe"
    probe.write_bytes(b"")
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.unlink()


def _assert_same_recording(rec: Recording, expected: Recording) -> None:
    assert list(rec.signals.columns) == list(expected.signals.columns)
    np.testing.assert_array_equal(rec.signals.to_numpy(), expected.signals.to_numpy())


def test_exemplar_is_a_minute_of_eight_channels(exemplar):
    assert list(exemplar.signals.columns) == CHANNELS
    assert exemplar.signals.shape == (N_SAMPLES, len(CHANNELS))
    assert {info["sample_frequency"] for info in exemplar.channels.values()} == {SFREQ}
    assert exemplar.signals.shape[0] / SFREQ == 60.0
    # Every channel carries signal, so a read that returns zeros cannot pass.
    assert np.all(np.ptp(exemplar.signals.to_numpy(), axis=0) > 0)
    assert exemplar.events is not None
    assert list(zip(exemplar.events["onset"], exemplar.events["description"], strict=True)) == (
        MARKERS
    )


def test_exemplar_reads_like_mne(exemplar):
    """Channel names, rate, length, every sample and every marker match MNE's own read."""
    raw = mne.io.read_raw_brainvision(str(VHDR), preload=True, verbose="ERROR")
    assert list(exemplar.signals.columns) == raw.ch_names
    assert {info["sample_frequency"] for info in exemplar.channels.values()} == {
        float(raw.info["sfreq"])
    }
    assert exemplar.signals.shape == (raw.n_times, len(raw.ch_names))
    np.testing.assert_array_equal(exemplar.signals.to_numpy(), raw.get_data().T)
    np.testing.assert_array_equal(exemplar.events["onset"].to_numpy(), raw.annotations.onset)
    np.testing.assert_array_equal(exemplar.events["duration"].to_numpy(), raw.annotations.duration)
    assert list(exemplar.events["description"]) == list(raw.annotations.description)


def test_exemplar_samples_are_the_interleaved_floats_in_the_data_file(exemplar):
    """An oracle that never touches MNE: de-interleave the binary file with NumPy.

    The header gives a resolution of 1 microvolt, and biosigio reports volts.
    """
    frames = np.fromfile(EXEMPLAR_DIR / "one_minute.eeg", dtype="<f4")
    frames = frames.reshape(N_SAMPLES, len(CHANNELS)).astype(np.float64)
    np.testing.assert_allclose(exemplar.signals.to_numpy(), frames * 1e-6, rtol=1e-12, atol=0)


def test_missing_data_file_is_a_typed_error_naming_the_file(triplet):
    (triplet / "one_minute.eeg").unlink()
    with pytest.raises(FileReadError) as info:
        _read(triplet)
    assert type(info.value) is FileReadError
    # The message carries the file name. It also embeds the directory (the reader quotes the
    # header's and MNE's own path), so the name is asserted and the absence of a path is not.
    assert "one_minute.eeg" in str(info.value)
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_stale_data_reference_is_read_from_the_same_stem_eeg(triplet, exemplar):
    """BIDS renames the triplet but not the header's ``DataFile=`` (issue #132)."""
    _set_data_file(triplet, "stale_name.eeg")
    before = _snapshot(triplet)
    rec = _read(triplet)
    _assert_same_recording(rec, exemplar)
    assert len(rec.events) == len(MARKERS)
    assert rec.metadata[HEADER_RECOVERED_KEY] == {
        "DataFile": {"referenced": "stale_name.eeg", "used": "one_minute.eeg"}
    }
    assert _snapshot(triplet) == before  # the dataset files are never touched


def test_same_stem_dat_wins_over_a_differently_named_eeg(triplet, exemplar):
    """The header still names the original ``.eeg``, which now carries another stem.

    A same-stem ``.dat`` sits beside it. The differently named ``.eeg`` is a decoy of the
    right size but all zeros, so reading it by mistake cannot reproduce the exemplar.
    """
    eeg = triplet / "one_minute.eeg"
    (triplet / "one_minute.dat").write_bytes(eeg.read_bytes())
    eeg.write_bytes(bytes(eeg.stat().st_size))
    eeg.rename(triplet / "renamed.eeg")
    before = _snapshot(triplet)
    rec = _read(triplet)
    _assert_same_recording(rec, exemplar)
    assert rec.metadata[HEADER_RECOVERED_KEY] == {
        "DataFile": {"referenced": "one_minute.eeg", "used": "one_minute.dat"}
    }
    assert _snapshot(triplet) == before


def test_differently_named_eeg_alone_is_not_guessed(triplet):
    (triplet / "one_minute.eeg").rename(triplet / "renamed.eeg")
    with pytest.raises(FileReadError, match="one_minute.eeg"):
        _read(triplet)


@pytest.mark.skipif(
    not _MNE_TOLERATES_MISSING_MARKER,
    reason=f"MNE {mne.__version__} raises on a missing marker file; see the test below",
)
def test_missing_marker_file_reads_the_data_without_events(triplet, exemplar):
    """MNE >= 1.13 reads the data without the markers; the three events are gone."""
    (triplet / "one_minute.vmrk").unlink()
    rec = _read(triplet)
    _assert_same_recording(rec, exemplar)
    assert rec.events is not None and rec.events.empty


@pytest.mark.skipif(
    _MNE_TOLERATES_MISSING_MARKER,
    reason=f"MNE {mne.__version__} reads a missing marker file; see the test above",
)
def test_missing_marker_file_is_a_typed_error_on_old_mne(triplet):
    (triplet / "one_minute.vmrk").unlink()
    with pytest.raises(FileReadError, match="one_minute.vmrk") as info:
        _read(triplet)
    assert type(info.value) is FileReadError
    assert isinstance(info.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("offset", [3, -1], ids=["half-plus-3-bytes", "half-minus-1-byte"])
def test_truncated_data_file_is_refused_on_both_read_paths(triplet, offset):
    """A data file cut to about half its length, mid-frame, is a typed corrupt-file error.

    A frame is 32 bytes here (8 channels of 4 bytes), and the header declares no sample count,
    so the size alone shows a partial trailing frame. Both the in-memory importer and the
    streaming source refuse it, and the message names the file, not its directory. A cut
    exactly on a frame boundary cannot be seen from this header and is not detected
    (see docs/formats/brainvision.md).
    """
    eeg = triplet / "one_minute.eeg"
    eeg.write_bytes(eeg.read_bytes()[: (N_SAMPLES // 2) * BYTES_PER_FRAME + offset])
    with pytest.raises(CorruptFileError) as in_memory:
        _read(triplet)
    with pytest.raises(CorruptFileError) as streamed:
        _open_stream_source(str(triplet / "one_minute.vhdr"), None)
    for err in (in_memory.value, streamed.value):
        assert err.code == "corrupt_or_truncated"
        assert "one_minute.eeg" in str(err)
        assert str(triplet) not in str(err)


def test_empty_data_file_is_a_typed_error(triplet):
    (triplet / "one_minute.eeg").write_bytes(b"")
    with pytest.raises(FileReadError) as info:
        _read(triplet)
    assert type(info.value) is FileReadError


@pytest.mark.parametrize("case", ["header-upper", "file-upper"])
def test_extension_case_mismatch(triplet, exemplar, case):
    """The header and the file differ only in the case of the extension.

    A case-insensitive filesystem (macOS and Windows defaults) hands the file over under
    either name, so there is nothing to recover. A case-sensitive one (Linux) cannot, and
    biosigio's sibling lookup finds the file and records the substitution.
    """
    if case == "header-upper":
        _set_data_file(triplet, "one_minute.EEG")
        referenced, used = "one_minute.EEG", "one_minute.eeg"
    else:
        (triplet / "one_minute.eeg").rename(triplet / "one_minute.EEG")
        referenced, used = "one_minute.eeg", "one_minute.EEG"
    before = _snapshot(triplet)
    rec = _read(triplet)
    _assert_same_recording(rec, exemplar)
    if _case_insensitive(triplet):
        assert HEADER_RECOVERED_KEY not in rec.metadata
    else:
        assert rec.metadata[HEADER_RECOVERED_KEY] == {
            "DataFile": {"referenced": referenced, "used": used}
        }
    assert _snapshot(triplet) == before
