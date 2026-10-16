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
import warnings

import numpy as np
import pytest

mne = pytest.importorskip(
    "mne", reason="BrainVision import requires the optional 'meg' extra (mne)"
)

from biosigio import Recording  # noqa: E402
from biosigio.exceptions import CorruptFileError, FileReadError  # noqa: E402
from biosigio.exporters.zarr_stream import _open_stream_source  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_DIR = _REPO / "examples/brainvision"
STEM = "sub-01_task-rest_eeg"

pytestmark = pytest.mark.skipif(
    not (FIXTURE_DIR / f"{STEM}.vhdr").exists(), reason="BrainVision fixture missing"
)

# The fixture header declares 8 channels of IEEE_FLOAT_32, so the frame size derived
# from it is 32 bytes. That is a frame size, not a file size.
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
    """With 16-bit samples the derived frame size is 16 bytes (8 channels of 2 bytes).

    A cut of 16 bytes removes one whole 16-bit frame, which cannot be detected and is
    read; a further cut of 8 bytes (half the derived frame size) leaves a partial
    frame and is refused. These are cut lengths and frame sizes, not file sizes.
    """
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


def _declare_data_points(vhdr: pathlib.Path, value) -> None:
    """Write ``DataPoints=<value>`` into the copied header's [Common Infos]."""
    text = vhdr.read_text(encoding="utf-8")
    anchor = f"NumberOfChannels={N_CHANNELS}\n"
    assert text.count(anchor) == 1 and "DataPoints" not in text
    vhdr.write_text(text.replace(anchor, f"{anchor}DataPoints={value}\n"), encoding="utf-8")


def _read_with(opener: str, vhdr: pathlib.Path):
    """Read ``vhdr`` through the in-memory importer or the streaming source."""
    if opener == "importer":
        return Recording.from_file(str(vhdr))
    src = _open_stream_source(str(vhdr), None)
    src.close()
    return src


@pytest.mark.parametrize("opener", ["importer", "stream"])
def test_data_points_equal_to_the_true_count_reads(tmp_path, opener):
    vhdr = _copy_triplet(tmp_path)
    true_points = (tmp_path / DATA_NAME).stat().st_size // FRAME_BYTES
    _declare_data_points(vhdr, true_points)

    result = _read_with(opener, vhdr)
    if opener == "importer":
        assert result.signals.shape == (true_points, N_CHANNELS)
    else:
        assert result.n_samples == true_points


@pytest.mark.parametrize("opener", ["importer", "stream"])
@pytest.mark.parametrize(
    ("declared_offset", "cut"),
    [
        pytest.param(1, 0, id="declares-one-sample-more-than-the-data"),
        pytest.param(0, FRAME_BYTES, id="data-cut-by-one-whole-frame"),
    ],
)
def test_file_smaller_than_data_points_is_refused(tmp_path, opener, declared_offset, cut):
    vhdr = _copy_triplet(tmp_path)
    true_points = (tmp_path / DATA_NAME).stat().st_size // FRAME_BYTES
    _declare_data_points(vhdr, true_points + declared_offset)
    _cut(tmp_path / DATA_NAME, cut)

    with pytest.raises(CorruptFileError) as excinfo:
        _read_with(opener, vhdr)
    _assert_refusal_names_file_only(excinfo, tmp_path)


@pytest.mark.parametrize("opener", ["importer", "stream"])
def test_file_longer_than_data_points_is_read_as_mne_reads_it(tmp_path, opener):
    """MNE sizes the recording from the file and ignores DataPoints for binary data."""
    vhdr = _copy_triplet(tmp_path)
    true_points = (tmp_path / DATA_NAME).stat().st_size // FRAME_BYTES
    _declare_data_points(vhdr, true_points - 1)

    result = _read_with(opener, vhdr)
    if opener == "importer":
        assert result.signals.shape == (true_points, N_CHANNELS)
    else:
        assert result.n_samples == true_points


@pytest.mark.parametrize("opener", ["importer", "stream"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("0", id="zero"),
        pytest.param("", id="empty"),
        pytest.param("-5", id="negative"),
        pytest.param("1250.5", id="decimal"),
        pytest.param("1e3", id="exponent"),
        pytest.param("abc", id="text"),
    ],
)
def test_data_points_that_is_not_a_positive_integer_keeps_the_frame_check_only(
    tmp_path, opener, value
):
    vhdr = _copy_triplet(tmp_path)
    true_points = (tmp_path / DATA_NAME).stat().st_size // FRAME_BYTES
    _declare_data_points(vhdr, value)

    result = _read_with(opener, vhdr)
    if opener == "importer":
        assert result.signals.shape == (true_points, N_CHANNELS)
    else:
        assert result.n_samples == true_points

    # Such a value is no count: a file cut by a whole frame still reads, as before.
    _cut(tmp_path / DATA_NAME, FRAME_BYTES)
    result = _read_with(opener, vhdr)
    if opener == "importer":
        assert result.signals.shape == (true_points - 1, N_CHANNELS)
    else:
        assert result.n_samples == true_points - 1


def test_ahdr_frame_counts_the_extra_channel_mne_adds(tmp_path):
    """MNE counts one channel more than an .ahdr header names, and so does the frame.

    The derived header names 7 of the fixture's 8 channels. Its data file is a whole
    number of 8-channel frames (a derived frame size of 32 bytes, not a file size) but
    not of 7-channel frames, so it must read, and a partial cut must still be refused.
    """
    vhdr = _copy_triplet(tmp_path)
    text = vhdr.read_text(encoding="utf-8")
    assert "NumberOfChannels=8" in text and "Ch8=F5,,0.1,\u00b5V\n" in text
    ahdr = tmp_path / f"{STEM}.ahdr"
    ahdr.write_text(
        text.replace("NumberOfChannels=8", "NumberOfChannels=7").replace(
            "Ch8=F5,,0.1,\u00b5V\n", ""
        ),
        encoding="utf-8",
    )
    data_file = tmp_path / DATA_NAME
    size = data_file.stat().st_size
    assert size % (7 * 4) != 0

    rec = Recording.from_file(str(ahdr), importer="brainvision")
    assert rec.signals.shape == (size // FRAME_BYTES, 7)

    _cut(data_file, 1)
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(ahdr), importer="brainvision")
    _assert_refusal_names_file_only(excinfo, tmp_path)


def test_int32_frame_follows_the_declared_binary_format(tmp_path):
    """INT_32 samples are 4 bytes, so the derived frame size is 32 bytes (a frame size,
    not a file size): half of it, 16 bytes, is a partial frame and is refused."""
    vhdr = _copy_triplet(tmp_path)
    data_file = tmp_path / DATA_NAME
    floats = np.frombuffer(data_file.read_bytes(), dtype="<f4")
    ints = np.clip(np.rint(floats.astype("<f8")), -(2**31), 2**31 - 1).astype("<i4")
    data_file.write_bytes(ints.tobytes())
    text = vhdr.read_text(encoding="utf-8")
    assert "BinaryFormat=IEEE_FLOAT_32" in text
    vhdr.write_text(text.replace("IEEE_FLOAT_32", "INT_32"), encoding="utf-8")

    assert Recording.from_file(str(vhdr)).signals.shape == (floats.size // N_CHANNELS, N_CHANNELS)

    _cut(data_file, FRAME_BYTES // 2)
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr))
    _assert_refusal_names_file_only(excinfo, tmp_path)


def _make_vectorized(vhdr: pathlib.Path) -> None:
    """Rewrite the copy as VECTORIZED data: each channel stored whole, one after another."""
    data_file = vhdr.with_name(DATA_NAME)
    samples = np.frombuffer(data_file.read_bytes(), dtype="<f4").reshape(-1, N_CHANNELS)
    data_file.write_bytes(np.ascontiguousarray(samples.T).tobytes())
    text = vhdr.read_text(encoding="utf-8")
    assert "DataOrientation=MULTIPLEXED" in text
    vhdr.write_text(text.replace("MULTIPLEXED", "VECTORIZED"), encoding="utf-8")


def test_vectorized_copy_reads_the_samples_of_the_multiplexed_fixture(tmp_path):
    expected = Recording.from_file(str(FIXTURE_DIR / f"{STEM}.vhdr")).signals.to_numpy()
    vhdr = _copy_triplet(tmp_path)
    _make_vectorized(vhdr)

    np.testing.assert_array_equal(Recording.from_file(str(vhdr)).signals.to_numpy(), expected)


@pytest.mark.parametrize("opener", ["importer", "stream"])
@pytest.mark.parametrize(
    "cut",
    [pytest.param(1, id="one-byte-short"), pytest.param(FRAME_BYTES // 2, id="half-a-frame-short")],
)
def test_vectorized_partial_frame_is_refused(tmp_path, opener, cut):
    vhdr = _copy_triplet(tmp_path)
    _make_vectorized(vhdr)
    _cut(tmp_path / DATA_NAME, cut)

    with pytest.raises(CorruptFileError) as excinfo:
        _read_with(opener, vhdr)
    _assert_refusal_names_file_only(excinfo, tmp_path)


def test_vectorized_whole_frame_cut_without_data_points_is_not_detected(tmp_path):
    """The undetected case: with no DataPoints there is nothing to compare the size to.

    The cut file still reads, and because vectorized data stores each channel whole,
    the channel boundaries move: the first channel is intact and the rest are corrupted.
    A declared DataPoints makes the same cut detectable.
    """
    vhdr = _copy_triplet(tmp_path)
    _make_vectorized(vhdr)
    intact = Recording.from_file(str(vhdr)).signals.to_numpy()
    _cut(tmp_path / DATA_NAME, FRAME_BYTES)

    cut = Recording.from_file(str(vhdr)).signals.to_numpy()

    assert cut.shape == (intact.shape[0] - 1, N_CHANNELS)
    np.testing.assert_array_equal(cut[:, 0], intact[:-1, 0])
    assert not np.array_equal(cut[:, 1], intact[:-1, 1])

    _declare_data_points(vhdr, intact.shape[0])
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr))
    _assert_refusal_names_file_only(excinfo, tmp_path)


def test_header_with_windows_line_endings_is_checked(tmp_path):
    vhdr = _copy_triplet(tmp_path)
    original = vhdr.read_bytes()
    assert b"\r\n" not in original
    vhdr.write_bytes(original.replace(b"\n", b"\r\n"))
    size = (tmp_path / DATA_NAME).stat().st_size

    assert Recording.from_file(str(vhdr)).signals.shape == (size // FRAME_BYTES, N_CHANNELS)

    _cut(tmp_path / DATA_NAME, 1)
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr))
    _assert_refusal_names_file_only(excinfo, tmp_path)


def test_free_text_in_the_comment_section_does_not_hide_the_check(tmp_path):
    """Comment text has no ``key=value`` shape; it must stay out of the parsed header."""
    vhdr = _copy_triplet(tmp_path)
    text = vhdr.read_text(encoding="utf-8")
    assert text.rstrip().endswith("[Comment]")
    vhdr.write_text(
        text.rstrip() + "\nRecorded during a pilot session.\nNo delimiter on these lines\n",
        encoding="utf-8",
    )
    size = (tmp_path / DATA_NAME).stat().st_size
    assert Recording.from_file(str(vhdr)).signals.shape == (size // FRAME_BYTES, N_CHANNELS)

    _cut(tmp_path / DATA_NAME, 1)
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr))
    _assert_refusal_names_file_only(excinfo, tmp_path)


def test_lowercase_common_infos_section_is_checked(tmp_path):
    """Some exporters write ``[Common infos]``; MNE reads it, so the check must too."""
    vhdr = _copy_triplet(tmp_path)
    text = vhdr.read_text(encoding="utf-8")
    assert "[Common Infos]" in text
    vhdr.write_text(text.replace("[Common Infos]", "[Common infos]"), encoding="utf-8")
    size = (tmp_path / DATA_NAME).stat().st_size
    assert Recording.from_file(str(vhdr)).signals.shape == (size // FRAME_BYTES, N_CHANNELS)

    _cut(tmp_path / DATA_NAME, 1)
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr))
    _assert_refusal_names_file_only(excinfo, tmp_path)


def test_percent_sign_in_the_data_file_name_is_not_interpolated(tmp_path):
    """A ``%`` in ``DataFile=`` must reach the check as written, as it reaches MNE."""
    vhdr = _copy_triplet(tmp_path)
    name = f"{STEM}_50%.eeg"
    (tmp_path / DATA_NAME).rename(tmp_path / name)
    text = vhdr.read_text(encoding="utf-8")
    vhdr.write_text(text.replace(f"DataFile={DATA_NAME}", f"DataFile={name}"), encoding="utf-8")
    size = (tmp_path / name).stat().st_size
    assert Recording.from_file(str(vhdr)).signals.shape == (size // FRAME_BYTES, N_CHANNELS)

    _cut(tmp_path / name, 1)
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr))
    _assert_refusal_names_file_only(excinfo, tmp_path, name=name)


@pytest.mark.parametrize("opener", ["importer", "stream"])
def test_ascii_data_has_no_fixed_frame_and_is_not_refused(tmp_path, opener):
    """ASCII data is text, so its size is no multiple of a frame. The header keeps a
    ``BinaryFormat=`` line, as converted headers do, so only the ``DataFormat=`` gate
    keeps the frame check off it."""
    vhdr = _copy_triplet(tmp_path)
    data_file = tmp_path / DATA_NAME
    samples = np.frombuffer(data_file.read_bytes(), dtype="<f4").reshape(-1, N_CHANNELS)
    data_file.write_text(
        "".join(" ".join(f"{v:.6g}" for v in row) + "\n" for row in samples), encoding="ascii"
    )
    text = vhdr.read_text(encoding="utf-8")
    assert "DataFormat=BINARY" in text and "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n" in text
    ascii_infos = "[ASCII Infos]\nDecimalSymbol=.\nSkipLines=0\nSkipColumns=0\n"
    vhdr.write_text(
        text.replace("DataFormat=BINARY", "DataFormat=ASCII").replace(
            "BinaryFormat=IEEE_FLOAT_32\n", f"BinaryFormat=IEEE_FLOAT_32\n\n{ascii_infos}"
        ),
        encoding="utf-8",
    )
    assert data_file.stat().st_size % FRAME_BYTES != 0  # a binary frame check would refuse it

    result = _read_with(opener, vhdr)
    if opener == "importer":
        assert result.signals.shape == (samples.shape[0], N_CHANNELS)
    else:
        assert result.n_samples == samples.shape[0]


def test_header_naming_no_channels_is_left_to_mne(tmp_path):
    """A frame of zero bytes cannot be checked; MNE raises its own typed error."""
    vhdr = _copy_triplet(tmp_path)
    text = vhdr.read_text(encoding="utf-8")
    vhdr.write_text(text.replace("NumberOfChannels=8", "NumberOfChannels=0"), encoding="utf-8")

    with pytest.raises(FileReadError) as excinfo:
        Recording.from_file(str(vhdr))
    assert not isinstance(excinfo.value, CorruptFileError)


def test_unreadable_header_and_missing_data_file_are_left_to_mne(tmp_path):
    with pytest.raises(FileReadError) as missing_header:
        Recording.from_file(str(tmp_path / f"{STEM}.vhdr"), importer="brainvision")
    assert not isinstance(missing_header.value, CorruptFileError)

    vhdr = _copy_triplet(tmp_path)
    (tmp_path / DATA_NAME).unlink()
    with pytest.raises(FileReadError) as missing_data:
        Recording.from_file(str(vhdr))
    assert not isinstance(missing_data.value, CorruptFileError)


def _append(path: pathlib.Path, n_bytes: int) -> None:
    """Append ``n_bytes`` zero bytes to ``path``."""
    path.write_bytes(path.read_bytes() + b"\x00" * n_bytes)


@pytest.mark.parametrize("opener", ["importer", "stream"])
@pytest.mark.parametrize(
    "extra",
    [pytest.param(1, id="one-extra-byte"), pytest.param(FRAME_BYTES - 1, id="frame-minus-one")],
)
def test_vectorized_file_with_a_partial_frame_is_refused_even_when_longer_than_data_points(
    tmp_path, opener, extra
):
    """A vectorized file is channels x points x bytes long whatever DataPoints says, so
    a size that is not a whole number of frames is refused when it exceeds the count."""
    vhdr = _copy_triplet(tmp_path)
    _make_vectorized(vhdr)
    true_points = (tmp_path / DATA_NAME).stat().st_size // FRAME_BYTES
    _declare_data_points(vhdr, true_points)
    _append(tmp_path / DATA_NAME, extra)

    with pytest.raises(CorruptFileError) as excinfo:
        _read_with(opener, vhdr)
    _assert_refusal_names_file_only(excinfo, tmp_path)


@pytest.mark.parametrize("opener", ["importer", "stream"])
def test_vectorized_file_of_whole_frames_with_matching_data_points_reads(tmp_path, opener):
    expected = Recording.from_file(str(FIXTURE_DIR / f"{STEM}.vhdr")).signals.to_numpy()
    vhdr = _copy_triplet(tmp_path)
    _make_vectorized(vhdr)
    _declare_data_points(vhdr, expected.shape[0])

    result = _read_with(opener, vhdr)
    if opener == "importer":
        np.testing.assert_array_equal(result.signals.to_numpy(), expected)
    else:
        assert result.n_samples == expected.shape[0]


@pytest.mark.parametrize("opener", ["importer", "stream"])
def test_multiplexed_file_with_a_partial_frame_beyond_data_points_reads_as_mne_reads_it(
    tmp_path, opener
):
    """MNE drops the trailing partial bytes of multiplexed data without moving any sample."""
    vhdr = _copy_triplet(tmp_path)
    true_points = (tmp_path / DATA_NAME).stat().st_size // FRAME_BYTES
    _declare_data_points(vhdr, true_points)
    _append(tmp_path / DATA_NAME, 1)

    result = _read_with(opener, vhdr)
    if opener == "importer":
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
        np.testing.assert_array_equal(result.signals.to_numpy().T, raw.get_data())
        assert result.signals.shape == (true_points, N_CHANNELS)
    else:
        assert result.n_samples == true_points


@pytest.mark.parametrize("opener", ["importer", "stream"])
def test_vectorized_whole_extra_frame_beyond_data_points_is_a_known_undetected_gap(
    tmp_path, opener
):
    """KNOWN GAP, not desired behavior. If DataPoints is right, this read is silently
    misaligned. The test pins current behavior; closing the gap means changing it to
    expect CorruptFileError.

    Whole extra frames cannot be told from a stale DataPoints. MNE sizes the channel
    blocks from the file, so the first channel is intact and channels 1 to 7 differ.
    """
    vhdr = _copy_triplet(tmp_path)
    _make_vectorized(vhdr)
    intact = Recording.from_file(str(vhdr)).signals.to_numpy()
    n_points = intact.shape[0]
    _declare_data_points(vhdr, n_points)
    _append(tmp_path / DATA_NAME, FRAME_BYTES)

    if opener == "importer":
        longer = Recording.from_file(str(vhdr)).signals.to_numpy()
        assert longer.shape == (n_points + 1, N_CHANNELS)
        first_samples = longer[:n_points].T
    else:
        src = _open_stream_source(str(vhdr), None)
        try:
            assert src.n_samples == n_points + 1
            first_samples = src.read(list(range(N_CHANNELS)), 0, n_points)
        finally:
            src.close()

    np.testing.assert_array_equal(first_samples[0], intact[:, 0])
    for channel in range(1, N_CHANNELS):
        assert not np.array_equal(first_samples[channel], intact[:, channel]), channel


# ---------------------------------------------------------------------------
# allow_partial_frame=True: the opt-in for a multiplexed file that ends in a partial frame
# ---------------------------------------------------------------------------

DROPPED_BYTES_KEY = "brainvision_dropped_bytes"

# (extra bytes appended, bytes cut, whole frames MNE reads, bytes it drops). Derived
# frame size is 32 bytes (8 channels of 4), so a cut of 1 byte leaves 31 bytes of
# partial frame and a cut of half the frame size leaves 16.
PARTIAL_CASES = [
    pytest.param(5, 0, 1250, 5, id="five-extra-bytes"),
    pytest.param(0, 1, 1249, 31, id="one-byte-short"),
    pytest.param(0, FRAME_BYTES // 2, 1249, FRAME_BYTES // 2, id="half-a-frame-short"),
]


def _partial_copy(directory: pathlib.Path, extra: int, cut: int) -> pathlib.Path:
    """A triplet copy whose data file ends in a partial frame."""
    vhdr = _copy_triplet(directory)
    if extra:
        _append(directory / DATA_NAME, extra)
    if cut:
        _cut(directory / DATA_NAME, cut)
    return vhdr


def _mne_whole_frames(n_frames: int) -> np.ndarray:
    """MNE's read of the unmodified whole-frame fixture, first ``n_frames`` samples."""
    raw = mne.io.read_raw_brainvision(
        str(FIXTURE_DIR / f"{STEM}.vhdr"), preload=True, verbose="ERROR"
    )
    return raw.get_data()[:, :n_frames]


def test_partial_frame_is_refused_unless_the_flag_is_on(tmp_path):
    vhdr = _partial_copy(tmp_path, 0, 1)

    with pytest.raises(CorruptFileError):
        Recording.from_file(str(vhdr))
    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(vhdr), allow_partial_frame=False)
    _assert_refusal_names_file_only(excinfo, tmp_path)


@pytest.mark.parametrize(("extra", "cut", "frames", "dropped"), PARTIAL_CASES)
def test_flag_on_reads_the_whole_frames_as_mne_reads_them(tmp_path, extra, cut, frames, dropped):
    vhdr = _partial_copy(tmp_path, extra, cut)

    with pytest.warns(UserWarning):
        rec = Recording.from_file(str(vhdr), allow_partial_frame=True)

    assert rec.signals.shape == (frames, N_CHANNELS)
    np.testing.assert_array_equal(rec.signals.to_numpy().T, _mne_whole_frames(frames))
    assert rec.events is not None and len(rec.events) == 3


@pytest.mark.parametrize(("extra", "cut", "frames", "dropped"), PARTIAL_CASES)
def test_flag_on_warns_naming_the_file_and_the_dropped_bytes(tmp_path, extra, cut, frames, dropped):
    vhdr = _partial_copy(tmp_path, extra, cut)

    with pytest.warns(UserWarning) as record:
        Recording.from_file(str(vhdr), allow_partial_frame=True)

    messages = [str(w.message) for w in record if "partial frame" in str(w.message)]
    assert len(messages) == 1
    assert DATA_NAME in messages[0]
    assert f"last {dropped} bytes" in messages[0]
    assert str(tmp_path) not in messages[0]
    assert os.sep not in messages[0]


@pytest.mark.parametrize(("extra", "cut", "frames", "dropped"), PARTIAL_CASES)
def test_flag_on_records_the_dropped_byte_count(tmp_path, extra, cut, frames, dropped):
    vhdr = _partial_copy(tmp_path, extra, cut)

    with pytest.warns(UserWarning):
        rec = Recording.from_file(str(vhdr), allow_partial_frame=True)

    assert rec.get_metadata(DROPPED_BYTES_KEY) == dropped


def test_flag_on_leaves_a_whole_frame_file_alone(tmp_path):
    """No warning and no metadata key when there is nothing to drop."""
    vhdr = _copy_triplet(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec = Recording.from_file(str(vhdr), allow_partial_frame=True)

    assert DROPPED_BYTES_KEY not in rec.metadata
    np.testing.assert_array_equal(rec.signals.to_numpy().T, _mne_whole_frames(1250))


@pytest.mark.parametrize("opener", ["importer", "stream"])
@pytest.mark.parametrize(
    ("extra", "cut"),
    [pytest.param(1, 0, id="one-extra-byte"), pytest.param(0, 1, id="one-byte-short")],
)
def test_flag_on_still_refuses_vectorized_data(tmp_path, opener, extra, cut):
    vhdr = _copy_triplet(tmp_path)
    _make_vectorized(vhdr)
    if extra:
        _append(tmp_path / DATA_NAME, extra)
    if cut:
        _cut(tmp_path / DATA_NAME, cut)

    with pytest.raises(CorruptFileError) as excinfo:
        if opener == "importer":
            Recording.from_file(str(vhdr), allow_partial_frame=True)
        else:
            _open_stream_source(str(vhdr), None, allow_partial_frame=True)
    _assert_refusal_names_file_only(excinfo, tmp_path)
    assert "does not apply to vectorized data" in str(excinfo.value)


@pytest.mark.parametrize("opener", ["importer", "stream"])
def test_flag_on_still_refuses_a_file_smaller_than_data_points(tmp_path, opener):
    vhdr = _copy_triplet(tmp_path)
    true_points = (tmp_path / DATA_NAME).stat().st_size // FRAME_BYTES
    _declare_data_points(vhdr, true_points)
    _cut(tmp_path / DATA_NAME, 1)

    with pytest.raises(CorruptFileError) as excinfo:
        if opener == "importer":
            Recording.from_file(str(vhdr), allow_partial_frame=True)
        else:
            _open_stream_source(str(vhdr), None, allow_partial_frame=True)
    _assert_refusal_names_file_only(excinfo, tmp_path)
    assert "smaller than" in str(excinfo.value)


def test_flag_is_ignored_by_other_importers():
    """Like mixed_rate outside EDF/BDF, the keyword reaches only the BrainVision reader."""
    wfdb_header = _REPO / "examples/100.hea"
    if not wfdb_header.exists():
        pytest.skip("WFDB fixture missing")

    rec = Recording.from_file(str(wfdb_header), allow_partial_frame=True)

    assert DROPPED_BYTES_KEY not in rec.metadata
    assert rec.signals.shape[0] > 0


@pytest.mark.parametrize(("extra", "cut", "frames", "dropped"), PARTIAL_CASES)
def test_streaming_source_honors_the_flag(tmp_path, extra, cut, frames, dropped):
    vhdr = _partial_copy(tmp_path, extra, cut)

    with pytest.raises(CorruptFileError):
        _open_stream_source(str(vhdr), None)

    with pytest.warns(UserWarning, match=f"last {dropped} bytes"):
        src = _open_stream_source(str(vhdr), None, allow_partial_frame=True)
    try:
        assert src.n_samples == frames
        assert src.extra_metadata[DROPPED_BYTES_KEY] == dropped
        np.testing.assert_array_equal(
            src.read(list(range(N_CHANNELS)), 0, frames), _mne_whole_frames(frames)
        )
    finally:
        src.close()


def test_stream_to_zarr_honors_the_flag_and_stores_the_dropped_count(tmp_path):
    zarr = pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")
    from biosigio import stream_to_zarr

    vhdr = _partial_copy(tmp_path, 0, 1)
    with pytest.raises(CorruptFileError):
        stream_to_zarr(str(vhdr), str(tmp_path / "refused.zarr"))

    with pytest.warns(UserWarning, match="last 31 bytes"):
        store = stream_to_zarr(str(vhdr), str(tmp_path / "accepted.zarr"), allow_partial_frame=True)

    root = zarr.open_group(store, mode="r")
    assert root.attrs["recording_metadata"][DROPPED_BYTES_KEY] == 31
