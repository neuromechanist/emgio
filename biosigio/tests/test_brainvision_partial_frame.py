"""A BrainVision data file with a partial trailing frame is refused, not read short.

MNE reads ``size // frame`` samples and drops the rest without a word, together with
any marker in the dropped span. biosigIO refuses such a file with ``CorruptFileError``
(code ``corrupt_or_truncated``) on both read paths: the in-memory importer and the
streaming Zarr source. Every case copies the committed fixture triplet
(``examples/brainvision``) into ``tmp_path`` and cuts the copy. NO MOCKS.
"""

import os
import pathlib
import shutil

import numpy as np
import pytest

mne = pytest.importorskip(
    "mne", reason="BrainVision import requires the optional 'meg' extra (mne)"
)

from biosigio import Recording  # noqa: E402
from biosigio.exceptions import CorruptFileError  # noqa: E402
from biosigio.exporters.zarr_stream import _open_stream_source  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_DIR = _REPO / "examples/brainvision"
STEM = "sub-01_task-rest_eeg"

pytestmark = pytest.mark.skipif(
    not (FIXTURE_DIR / f"{STEM}.vhdr").exists(), reason="BrainVision fixture missing"
)

# The fixture header declares 8 channels of IEEE_FLOAT_32, so a frame is 32 bytes.
N_CHANNELS = 8
FRAME_BYTES = N_CHANNELS * 4
DATA_NAME = f"{STEM}.eeg"


def _copy_triplet(directory: pathlib.Path) -> pathlib.Path:
    """Copy the fixture triplet into ``directory``; return the copied header path."""
    for ext in (".vhdr", ".vmrk", ".eeg"):
        shutil.copy(FIXTURE_DIR / f"{STEM}{ext}", directory / f"{STEM}{ext}")
    return directory / f"{STEM}.vhdr"


def _cut(path: pathlib.Path, n_bytes: int) -> None:
    """Remove the last ``n_bytes`` bytes of ``path``."""
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - n_bytes])


def _assert_refusal_names_file_only(excinfo, directory: pathlib.Path, name: str = DATA_NAME):
    err = excinfo.value
    message = str(err)
    assert err.code == "corrupt_or_truncated"
    assert name in message
    assert str(directory) not in message
    assert os.sep not in message


def test_fixture_triplet_still_reads_as_mne_reads_it(tmp_path):
    """The whole-frame fixture is untouched by the check: same channels, rate and data."""
    vhdr = _copy_triplet(tmp_path)
    assert (tmp_path / DATA_NAME).stat().st_size % FRAME_BYTES == 0

    rec = Recording.from_file(str(vhdr))
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")

    assert len(rec.channels) == raw.info["nchan"] == N_CHANNELS
    assert {info["sample_frequency"] for info in rec.channels.values()} == {raw.info["sfreq"]}
    assert rec.signals.shape == (raw.n_times, N_CHANNELS)
    np.testing.assert_array_equal(rec.signals.to_numpy().T, raw.get_data())
    assert rec.events is not None and len(rec.events) == 3


@pytest.mark.parametrize(
    "cut",
    [pytest.param(1, id="one-byte-short"), pytest.param(FRAME_BYTES // 2, id="half-a-frame-short")],
)
def test_partial_frame_is_refused(tmp_path, cut):
    vhdr = _copy_triplet(tmp_path)
    _cut(tmp_path / DATA_NAME, cut)

    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr))
    _assert_refusal_names_file_only(excinfo, tmp_path)


@pytest.mark.parametrize(
    "cut",
    [pytest.param(1, id="one-byte-short"), pytest.param(FRAME_BYTES // 2, id="half-a-frame-short")],
)
def test_partial_frame_is_refused_by_the_streaming_source(tmp_path, cut):
    vhdr = _copy_triplet(tmp_path)
    _cut(tmp_path / DATA_NAME, cut)

    with pytest.raises(CorruptFileError) as excinfo:
        _open_stream_source(str(vhdr), None)
    _assert_refusal_names_file_only(excinfo, tmp_path)


def test_whole_fixture_still_opens_on_the_streaming_source(tmp_path):
    vhdr = _copy_triplet(tmp_path)

    src = _open_stream_source(str(vhdr), None)
    try:
        assert len(src.channels) == N_CHANNELS
    finally:
        src.close()


@pytest.mark.parametrize("opener", ["importer", "stream"])
def test_partial_frame_of_a_header_recovered_from_a_sibling_names_the_dataset_file(
    tmp_path, opener
):
    """The header names a file that is gone, so the sibling is read through a staged
    copy; the refusal names the sibling, not the staged name or the temporary directory."""
    vhdr = _copy_triplet(tmp_path)
    text = vhdr.read_text(encoding="utf-8")
    assert f"DataFile={DATA_NAME}" in text
    vhdr.write_text(text.replace(f"DataFile={DATA_NAME}", "DataFile=pre-rename.eeg"), "utf-8")
    _cut(tmp_path / DATA_NAME, 1)

    with pytest.raises(CorruptFileError) as excinfo:
        if opener == "importer":
            Recording.from_file(str(vhdr))
        else:
            _open_stream_source(str(vhdr), None)
    _assert_refusal_names_file_only(excinfo, tmp_path)
    message = str(excinfo.value)
    assert "pre-rename" not in message
    assert "data.eeg" not in message
    assert "biosigio-vhdr-" not in message


def test_frame_size_follows_the_declared_binary_format(tmp_path):
    """With 16-bit samples a frame is 16 bytes: half a frame (8 bytes) is refused,
    while 16 bytes off is a whole number of 16-bit frames and is left alone."""
    vhdr = _copy_triplet(tmp_path)
    data_file = tmp_path / DATA_NAME
    floats = np.frombuffer(data_file.read_bytes(), dtype="<f4")
    ints = np.clip(np.rint(floats), -32768, 32767).astype("<i2")
    data_file.write_bytes(ints.tobytes())
    text = vhdr.read_text(encoding="utf-8")
    assert "BinaryFormat=IEEE_FLOAT_32" in text
    vhdr.write_text(text.replace("IEEE_FLOAT_32", "INT_16"), encoding="utf-8")

    assert Recording.from_file(str(vhdr)).signals.shape == (floats.size // N_CHANNELS, N_CHANNELS)

    _cut(data_file, 16)
    assert Recording.from_file(str(vhdr)).signals.shape[0] == floats.size // N_CHANNELS - 1

    _cut(data_file, 8)
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr))
    _assert_refusal_names_file_only(excinfo, tmp_path)
