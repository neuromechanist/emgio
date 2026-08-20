"""4D Neuroimaging/BTi MEG importer (read_raw_bti) on a real vendored fixture,
plus synthetic-layout tests for the content-based directory detection.

Uses a small vendored 4D/BTi recording (examples/bti/sub-01_task-test_meg/, ~480
KB total, from MNE's BSD-licensed test suite; see examples/bti/README.md).
Validates that biosigIO reads BTi data through Recording.from_file's
content-based directory detection (BTi ships with no file extension, so it
can't be recognized by extension the way CTF .ds/KIT .con/.sqd/.kdf are).

Detection/dispatch itself is covered by synthetic (placeholder-content)
directory layouts -- no real BTi binary data is needed to prove a directory is
or isn't recognized as BTi, only the right filenames in the right place. This
also covers the negative case explicitly warned about in the format facts: a
directory holding only a nested ``.datalad/config`` (present in almost every
datalad-tracked dataset repo) must NOT be mistaken for a BTi sidecar.

Skips cleanly if the optional 'meg' extra (mne) is not installed.
"""

import pathlib
from collections import Counter

import pytest

from biosigio import Recording

pytest.importorskip("mne", reason="BTi import requires the optional 'meg' extra (mne)")

from biosigio.importers.meg import _find_bti_pdf  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
BTI = _REPO / "examples/bti/sub-01_task-test_meg"

pytestmark = pytest.mark.skipif(not BTI.exists(), reason="BTi fixture missing")


@pytest.fixture(scope="module")
def bti_rec():
    return Recording.from_file(str(BTI))


def test_bti_channel_types_preserved(bti_rec):
    """Sensor types stay distinct (mag vs reference vs stim/misc), not collapsed."""
    types = Counter(i["channel_type"] for i in bti_rec.channels.values())
    assert types["MEGMAG"] == 248
    assert types["MEGREFMAG"] == 23
    assert types["TRIG"] == 2
    assert types["MISC"] == 7
    assert sum(types.values()) == 280
    assert bti_rec.get_n_channels() == 280


def test_bti_units(bti_rec):
    """Magnetometers/reference magnetometers carry tesla; the 5 reference
    gradiometers among the reference sensors carry tesla-per-meter; stim/misc
    carry volts."""
    dims = Counter((i["channel_type"], i["physical_dimension"]) for i in bti_rec.channels.values())
    assert dims[("MEGMAG", "T")] == 248
    assert dims[("MEGREFMAG", "T")] == 18
    assert dims[("MEGREFMAG", "T/m")] == 5
    assert dims[("TRIG", "V")] == 2
    assert dims[("MISC", "V")] == 7


def test_bti_sampling_and_shape(bti_rec):
    rates = {i["sample_frequency"] for i in bti_rec.channels.values()}
    assert rates == {1017.25}
    assert bti_rec.signals.shape == (305, 280)


def test_bti_stim_events(bti_rec):
    """Stim-channel triggers are read into events (onsets in seconds, sorted)."""
    assert len(bti_rec.events) > 0
    onsets = bti_rec.events["onset"].to_numpy()
    assert (onsets[:-1] <= onsets[1:]).all()  # sorted
    assert onsets.min() >= 0.0


def test_bti_source_format_recorded(bti_rec):
    assert bti_rec.metadata.get("source_format") == "meg"
    assert bti_rec.metadata.get("source_file") == str(BTI)


def test_bti_explicit_importer_matches_auto_detect():
    """importer='meg' (skipping the content sniff) reads the same recording."""
    rec = Recording.from_file(str(BTI), importer="meg")
    assert rec.get_n_channels() == 280


# -- Content-based detection / dispatch (synthetic layouts, no real BTi data) ----


def test_find_bti_pdf_recognizes_conventional_layout(tmp_path):
    d = tmp_path / "sub-01_task-rest_meg"
    d.mkdir()
    (d / "c,rfDC").write_bytes(b"\x00")
    (d / "config").write_bytes(b"\x00")
    (d / "hs_file").write_bytes(b"\x00")
    assert _find_bti_pdf(str(d)) == str(d / "c,rfDC")


def test_find_bti_pdf_recognizes_filtered_pdf_variant(tmp_path):
    """A filtered copy (e.g. c,rfDC,fn50,o) also matches the c,rf* prefix."""
    d = tmp_path / "sub-01_task-rest_meg"
    d.mkdir()
    (d / "c,rfDC,fn50,o").write_bytes(b"\x00")
    (d / "config").write_bytes(b"\x00")
    assert _find_bti_pdf(str(d)) == str(d / "c,rfDC,fn50,o")


def test_find_bti_pdf_optional_hs_file(tmp_path):
    """hs_file is optional in BIDS; detection must not require it."""
    d = tmp_path / "sub-01_task-rest_meg"
    d.mkdir()
    (d / "c,rfDC").write_bytes(b"\x00")
    (d / "config").write_bytes(b"\x00")
    assert _find_bti_pdf(str(d)) is not None


def test_find_bti_pdf_requires_config_sibling(tmp_path):
    """A c,rf* file with no sibling config is not enough (two-signal check)."""
    d = tmp_path / "sub-01_task-rest_meg"
    d.mkdir()
    (d / "c,rfDC").write_bytes(b"\x00")
    assert _find_bti_pdf(str(d)) is None


def test_find_bti_pdf_rejects_datalad_config_only(tmp_path):
    """The explicit negative case: a directory holding only .datalad/config (every
    datalad-tracked dataset has one) must NOT be detected as BTi -- matching on the
    basename 'config' alone, or recursing into subdirectories, would misfire here."""
    d = tmp_path / "some-dataset"
    d.mkdir()
    datalad_dir = d / ".datalad"
    datalad_dir.mkdir()
    (datalad_dir / "config").write_bytes(b"[datalad]\n")
    assert _find_bti_pdf(str(d)) is None


def test_find_bti_pdf_empty_or_missing_directory(tmp_path):
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    assert _find_bti_pdf(str(empty)) is None
    assert _find_bti_pdf(str(tmp_path / "does_not_exist")) is None


def test_bti_directory_auto_detects_to_meg_importer(tmp_path):
    """Recording._infer_importer routes a synthetic BTi-layout directory to 'meg'."""
    d = tmp_path / "sub-01_task-rest_meg"
    d.mkdir()
    (d / "c,rfDC").write_bytes(b"\x00")
    (d / "config").write_bytes(b"\x00")
    assert Recording._infer_importer(str(d)) == "meg"


def test_non_bti_extensionless_directory_raises_unsupported(tmp_path):
    """An extension-less directory that isn't BTi-shaped raises a clear,
    typed error instead of silently falling through to the FIF reader."""
    from biosigio.exceptions import UnsupportedFormatError

    d = tmp_path / "some-dataset"
    d.mkdir()
    (d / "readme.txt").write_bytes(b"not bti")
    with pytest.raises(UnsupportedFormatError):
        Recording._infer_importer(str(d))


def test_datalad_dataset_directory_not_misdetected_end_to_end(tmp_path):
    """End-to-end version of the negative case through Recording.from_file:
    a dataset directory whose only 'config' lives under .datalad/ must raise
    UnsupportedFormatError, not silently misread as BTi (or crash)."""
    from biosigio.exceptions import UnsupportedFormatError

    d = tmp_path / "some-dataset"
    d.mkdir()
    (d / ".datalad").mkdir()
    (d / ".datalad" / "config").write_bytes(b"[datalad]\n")
    with pytest.raises(UnsupportedFormatError):
        Recording.from_file(str(d))
