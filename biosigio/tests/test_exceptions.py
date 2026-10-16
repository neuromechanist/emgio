"""Typed read-error classification (biosigio.exceptions).

A caller (the NEMAR Zarr pipeline) needs to know *why* a recording could not be
read so it can surface a specific reason. These tests pin the classifier and the
importers' typed raises on REAL files (no mocks):

* an unknown extension -> UnsupportedFormatError
* a real MNE evoked (``*-ave.fif``: averaged, no continuous raw) -> NotContinuousRecordingError
* a truncated EDF -> CorruptFileError
* a FIF cut inside its data -> CorruptFileError
* mixed-rate EDF -> MixedSamplingRateError (still a ValueError, for back-compat)

The string-matching unit tests cover the reader phrasings we map without needing
each format on hand.
"""

import errno
import importlib.util
import os
import pathlib
import shutil
import struct
import tempfile

import numpy as np
import pyedflib
import pytest

from biosigio import Recording
from biosigio.exceptions import (
    REASONS,
    BiosigIOError,
    CorruptFileError,
    FileReadError,
    MixedSamplingRateError,
    NotContinuousRecordingError,
    UnsupportedFormatError,
    classify_read_error,
    is_host_condition,
    is_resource_exhaustion,
)

_HAS_MNE = importlib.util.find_spec("mne") is not None
_REPO = pathlib.Path(__file__).resolve().parents[2]
_MEG_FIF = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"
_CTF_DS = _REPO / "examples/ctf/catch-alp-good-f.ds"
_KIT_SQD = _REPO / "examples/kit/sub-01_task-test_meg.sqd"
_BTI_DIR = _REPO / "examples/bti/sub-01_task-test_meg"


# -- classify_read_error: reader-phrasing -> typed error -----------------------


def test_classify_no_raw_data_is_not_continuous():
    err = classify_read_error(ValueError("No raw data in /x/sub-01_task-a-ave.fif"))
    assert isinstance(err, NotContinuousRecordingError)
    assert err.code == "not_continuous"


def test_classify_not_a_raw_file_is_not_continuous():
    err = classify_read_error(RuntimeError("This is not a raw file"))
    assert isinstance(err, NotContinuousRecordingError)


def test_classify_filesize_is_corrupt():
    err = classify_read_error(OSError("the file is not EDF(+) or BDF(+) compliant (Filesize)"))
    assert isinstance(err, CorruptFileError)
    assert err.code == "corrupt_or_truncated"


def test_classify_incomplete_split_is_corrupt():
    err = classify_read_error(
        ValueError("Split raw file detected but next file ..._split-02_meg.fif does not exist")
    )
    assert isinstance(err, CorruptFileError)


def test_classify_ctf_trial_truncation_is_corrupt():
    err = classify_read_error(
        ValueError("The number of samples is not an even multiple of the trial size")
    )
    assert isinstance(err, CorruptFileError)


def test_classify_numpy_partial_element_buffer_is_corrupt():
    """The text numpy raises when a tag read returns a partial element: a FIF
    cut mid-sample leaves a buffer that is not a whole number of elements."""
    with pytest.raises(ValueError) as caught:
        np.frombuffer(b"\x00" * 7, dtype=">f4")
    err = classify_read_error(caught.value)
    assert isinstance(err, CorruptFileError)
    assert err.code == "corrupt_or_truncated"


@pytest.mark.parametrize(
    "text",
    ["buffer size must be a multiple of element size", "cannot reshape array of size 3"],
    ids=["buffer", "reshape"],
)
def test_classify_host_condition_wins_over_a_truncation_match(text):
    """The host check runs before the message match: an EIO whose text happens to
    be a truncation message stays the OSError, so a caller retries the host
    instead of recording a corrupt file."""
    exc = OSError(errno.EIO, text)
    with pytest.raises(OSError) as caught:
        classify_read_error(exc)
    assert caught.value is exc


def test_classify_numpy_reshape_size_mismatch_is_corrupt():
    """The text numpy raises when a FIF data buffer holds fewer samples than its
    header declares: the whole samples that remain no longer fill the block."""
    with pytest.raises(ValueError) as caught:
        np.zeros(7, dtype=np.float32).reshape((2, 4))
    err = classify_read_error(caught.value)
    assert isinstance(err, CorruptFileError)
    assert err.code == "corrupt_or_truncated"


def test_classify_unrelated_value_error_is_not_corrupt():
    """The truncation mappings are specific messages, not a catch-all for
    ``ValueError``: another numpy error stays the generic read failure."""
    with pytest.raises(ValueError) as caught:
        np.ones(2) @ np.ones(3)  # shape mismatch, a different numpy message
    err = classify_read_error(caught.value)
    assert type(err) is FileReadError
    assert err.code == "file_read_error"


@pytest.mark.parametrize(
    "text",
    [
        "12 values are needed to reshape it into the grid",
        "the read buffer was empty",
    ],
    ids=["reshape", "buffer"],
)
def test_classify_near_miss_text_is_not_corrupt(text):
    """Text that shares a word with a matched message ("reshape", "buffer") but
    not the message itself stays the generic read failure. Widening either match
    to that single word would type these as corrupt."""
    err = classify_read_error(ValueError(text))
    assert type(err) is FileReadError
    assert err.code == "file_read_error"


def test_classify_unknown_falls_back_to_file_read_error():
    err = classify_read_error(RuntimeError("something unexpected"))
    assert type(err) is FileReadError
    assert err.code == "file_read_error"


def test_classify_passes_through_existing_typed_error():
    original = NotContinuousRecordingError("already typed")
    assert classify_read_error(original) is original


def test_classify_includes_filepath_when_given():
    err = classify_read_error(ValueError("No raw data"), "/data/sub-01_task-a-ave.fif")
    assert "sub-01_task-a-ave.fif" in str(err)


def test_classify_truncated_fif_names_the_file_without_its_directory():
    """A truncated-FIF message carries the file name only, for a file and for a
    directory recording (a CTF ``.ds`` path may end in a separator)."""
    with pytest.raises(ValueError) as caught:
        np.frombuffer(b"\x00" * 7, dtype=">f4")
    for path in ("/scratch/run7/sub-01_meg.fif", "/scratch/run7/sub-01_meg.ds/"):
        err = classify_read_error(caught.value, path)
        assert isinstance(err, CorruptFileError)
        assert pathlib.PurePath(path).name in str(err)
        assert "scratch" not in str(err) and "run7" not in str(err)


# -- is_resource_exhaustion: resource exhaustion is never a property of the file
# (issue #123) ------------------------------------------------------------------


def test_is_resource_exhaustion_memory_error():
    assert is_resource_exhaustion(MemoryError("out of memory")) is True


def test_is_resource_exhaustion_numpy_array_memory_error():
    """numpy's ``_ArrayMemoryError`` (a MemoryError subclass) is the exact shape
    the on008083 failures logged: "Unable to allocate 24.8 MiB for an array
    with shape (3248000,) and data type float64", raised inside a per-channel
    readSignal loop when the host is saturated. Provoked for real (not
    hand-constructed) so the assertion is about the actual numpy type -- numpy
    decorates it ``@_display_as_base`` so ``str()``/``type().__name__`` both
    read "MemoryError" (by design, for a friendlier traceback), so the type
    is confirmed via its module instead of its display name."""
    with pytest.raises(MemoryError) as exc_info:
        np.zeros(int(1e18), dtype=np.float64)
    exc = exc_info.value
    assert type(exc) is not MemoryError  # genuinely numpy's subclass, not the builtin
    assert type(exc).__module__.startswith("numpy")
    assert is_resource_exhaustion(exc) is True


def test_is_resource_exhaustion_thread_start_failure():
    assert is_resource_exhaustion(RuntimeError("can't start new thread")) is True


def test_is_resource_exhaustion_enomem_oserror():
    assert is_resource_exhaustion(OSError(errno.ENOMEM, "Cannot allocate memory")) is True


def test_is_resource_exhaustion_eagain_oserror():
    """EAGAIN is kept for fork()-under-RLIMIT_NPROC (thread/process exhaustion),
    not because this codebase does nonblocking I/O -- it does not, so a real
    EAGAIN here always means the exhaustion reading. See the docstring on
    _RESOURCE_EXHAUSTION_ERRNOS."""
    assert is_resource_exhaustion(OSError(errno.EAGAIN, "Resource temporarily unavailable")) is True


def test_is_resource_exhaustion_emfile_oserror():
    """Per-process file-descriptor exhaustion (e.g. many parallel EDF/HDF5/Zarr
    readers open at once) is the same "host is saturated" condition as
    MemoryError, just a different resource."""
    assert is_resource_exhaustion(OSError(errno.EMFILE, "Too many open files")) is True


def test_is_resource_exhaustion_enfile_oserror():
    """System-wide (not just per-process) file-descriptor exhaustion."""
    assert is_resource_exhaustion(OSError(errno.ENFILE, "Too many open files in system")) is True


def test_is_resource_exhaustion_cannot_allocate_memory_message_only():
    """No errno set (e.g. re-raised across a process boundary, or constructed
    by a library from a bare message) -- the substring match must still fire."""
    assert is_resource_exhaustion(OSError("cannot allocate memory")) is True


def test_is_resource_exhaustion_too_many_open_files_message_only():
    assert is_resource_exhaustion(OSError("Too many open files")) is True


def test_is_resource_exhaustion_negative_case_genuine_missing_file():
    """A real ENOENT is a file problem, not resource exhaustion -- it must still
    classify as a read error (not be treated as retryable)."""
    assert is_resource_exhaustion(OSError(errno.ENOENT, "No such file or directory")) is False


def test_is_resource_exhaustion_negative_case_different_errno_not_matched():
    """A real, unrelated errno (permission denied) on an OSError whose message
    happens to say nothing exhaustion-flavored must not be matched -- proves
    the errno check is a specific allowlist, not "any OSError with an errno"."""
    assert is_resource_exhaustion(OSError(errno.EACCES, "Permission denied")) is False


def test_is_resource_exhaustion_negative_case_unrelated_error():
    assert is_resource_exhaustion(ValueError("not EDF(+) or BDF(+) compliant (Filesize)")) is False


# -- is_resource_exhaustion walks __cause__/__context__ so cleanup can't mask ---
# resource exhaustion (a finally/__exit__/close() that itself raises while a
# MemoryError is propagating would otherwise hide it from a caller's `except`).


def test_is_resource_exhaustion_walks_explicit_cause_chain():
    inner = MemoryError("Unable to allocate 24.8 MiB for an array")
    outer = RuntimeError("cleanup failed")
    outer.__cause__ = inner  # explicit `raise outer from inner` shape
    assert is_resource_exhaustion(outer) is True


def test_is_resource_exhaustion_walks_implicit_context_chain():
    """The shape Python itself produces when a `finally`/`__exit__` raises while
    another exception is already propagating (no `from` needed -- this is
    automatic, see test_eeglab_v73_masked_memory_error_via_h5py_exit for the
    real h5py-context-manager version of this)."""
    try:
        try:
            raise MemoryError("Unable to allocate 24.8 MiB for an array")
        finally:
            raise OSError("unable to synchronously close file, id_type: 0x1")
    except OSError as outer:
        assert outer.__cause__ is None
        assert isinstance(outer.__context__, MemoryError)
        assert is_resource_exhaustion(outer) is True


def test_is_resource_exhaustion_chain_walk_stops_at_unrelated_root():
    """A chain that never bottoms out in resource exhaustion is correctly False,
    even several links deep -- the walk does not just always return True once
    it starts walking."""
    root = ValueError("not EDF(+) or BDF(+) compliant (Filesize)")
    mid = RuntimeError("re-raised during cleanup")
    mid.__cause__ = root
    outer = OSError("outermost wrapper")
    outer.__cause__ = mid
    assert is_resource_exhaustion(outer) is False


def test_is_resource_exhaustion_chain_depth_is_bounded():
    """A MemoryError buried past _MAX_CHAIN_DEPTH links is NOT found -- the walk
    is bounded, not an unbounded chain traversal (guards against a pathological
    or cyclic __context__ chain hanging the check)."""
    from biosigio.exceptions import _MAX_CHAIN_DEPTH

    memory_error = MemoryError("buried too deep")
    exc: BaseException = memory_error
    # One link past the bound: depth 0 is `outer` itself, so placing the
    # MemoryError at exactly _MAX_CHAIN_DEPTH links away is one step too far.
    for _ in range(_MAX_CHAIN_DEPTH):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = exc
        exc = wrapper
    assert is_resource_exhaustion(exc) is False

    # One link closer (within the bound) IS found.
    exc2: BaseException = memory_error
    for _ in range(_MAX_CHAIN_DEPTH - 1):
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = exc2
        exc2 = wrapper
    assert is_resource_exhaustion(exc2) is True


# -- classify_read_error re-raises resource exhaustion, never FileReadError -----


def test_classify_read_error_reraises_memory_error():
    exc = MemoryError("Unable to allocate 24.8 MiB for an array")
    with pytest.raises(MemoryError) as exc_info:
        classify_read_error(exc, "/data/sub-001_ses-01_task-HierPrior_eeg.edf")
    assert exc_info.value is exc


def test_classify_read_error_reraises_numpy_array_memory_error():
    try:
        np.zeros(int(1e18), dtype=np.float64)
    except MemoryError as exc:
        with pytest.raises(MemoryError) as exc_info:
            classify_read_error(exc, "/data/sub-001.edf")
        assert exc_info.value is exc
    else:
        pytest.fail("expected numpy to raise MemoryError for an absurd allocation")


def test_classify_read_error_reraises_thread_exhaustion_runtime_error():
    exc = RuntimeError("can't start new thread")
    with pytest.raises(RuntimeError) as exc_info:
        classify_read_error(exc)
    assert exc_info.value is exc


def test_classify_read_error_reraises_enomem_oserror():
    exc = OSError(errno.ENOMEM, "Cannot allocate memory")
    with pytest.raises(OSError) as exc_info:
        classify_read_error(exc)
    assert exc_info.value is exc


def test_classify_read_error_negative_case_enoent_still_classifies_as_read_error():
    """A genuine "file not found"-flavored OSError is a real read problem, not
    resource exhaustion, and must still fall through to FileReadError."""
    err = classify_read_error(OSError(errno.ENOENT, "No such file or directory"), "/x/missing.edf")
    assert type(err) is FileReadError
    assert err.code == "file_read_error"


# -- host I/O errors are the host, not the file: re-raised, never typed --------

_HOST_IO = [
    errno.EACCES,
    errno.EPERM,
    errno.EIO,
    errno.ENOSPC,
    errno.EROFS,
    errno.ETIMEDOUT,
    *(code for code in (getattr(errno, "EDQUOT", None), getattr(errno, "ESTALE", None)) if code),
]


@pytest.mark.parametrize("code", _HOST_IO, ids=[errno.errorcode[c] for c in _HOST_IO])
def test_classify_read_error_reraises_host_io_errors(code):
    exc = OSError(code, os.strerror(code), "/data/sub-01_eeg.edf")
    assert is_host_condition(exc) is True
    with pytest.raises(OSError) as info:
        classify_read_error(exc, "/data/sub-01_eeg.edf")
    assert info.value is exc


def test_host_condition_is_found_behind_a_wrapper():
    """A reader that wraps the OS error (``raise ValueError(...) from err``) is
    still a host condition, and the wrapper is re-raised as-is."""
    outer = ValueError("could not open recording")
    outer.__cause__ = PermissionError(errno.EACCES, "Permission denied")
    assert is_host_condition(outer) is True
    with pytest.raises(ValueError) as info:
        classify_read_error(outer)
    assert info.value is outer


def test_host_condition_includes_resource_exhaustion():
    assert is_host_condition(MemoryError()) is True


@pytest.mark.parametrize(
    "exc",
    [
        OSError(errno.ENOENT, "No such file or directory"),
        OSError("the file is not EDF(+) or BDF(+) compliant (Filesize)"),  # no errno
        ValueError("header is corrupt"),
    ],
    ids=["enoent", "errno-less-oserror", "valueerror"],
)
def test_file_problems_are_not_host_conditions(exc):
    assert is_host_condition(exc) is False
    assert isinstance(classify_read_error(exc), FileReadError)


# -- typed errors are ValueErrors (back-compat) + have stable codes ------------


def test_typed_errors_are_valueerrors():
    for exc in (
        UnsupportedFormatError,
        FileReadError,
        NotContinuousRecordingError,
        CorruptFileError,
        MixedSamplingRateError,
    ):
        assert issubclass(exc, BiosigIOError)
        assert issubclass(exc, ValueError)


def test_every_code_has_a_reason():
    for exc in (
        BiosigIOError,
        UnsupportedFormatError,
        FileReadError,
        NotContinuousRecordingError,
        CorruptFileError,
        MixedSamplingRateError,
    ):
        assert exc.code in REASONS and REASONS[exc.code]


# -- real files ----------------------------------------------------------------


def test_unknown_extension_raises_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".xyz") as fh:
        with pytest.raises(UnsupportedFormatError):
            Recording.from_file(fh.name)


@pytest.mark.skipif(not _HAS_MNE, reason="MEG path needs the 'meg' extra (mne)")
def test_real_evoked_ave_fif_is_not_continuous():
    """A trial-averaged MNE evoked (``*-ave.fif``) is valid data but carries no
    continuous raw recording -> NotContinuousRecordingError (the on005261 case)."""
    import mne

    info = mne.create_info(["MEG0011", "MEG0021"], sfreq=100.0, ch_types="mag")
    evoked = mne.EvokedArray(np.zeros((2, 50), dtype=float), info, tmin=0.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub-01_task-x_ave.fif")  # MNE wants *-ave.fif
        evoked.save(path)
        with pytest.raises(NotContinuousRecordingError):
            Recording.from_file(path)


def test_truncated_edf_is_corrupt():
    """An EDF cut short of its declared length fails pyedflib's Filesize check ->
    CorruptFileError."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rec.edf")
        rate, n = 100.0, 1000
        w = pyedflib.EdfWriter(path, 1)
        try:
            w.setSignalHeaders(
                [
                    {
                        "label": "EEG1",
                        "dimension": "uV",
                        "sample_frequency": rate,
                        "physical_max": 100.0,
                        "physical_min": -100.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    }
                ]
            )
            w.writeSamples([np.zeros(n)])
        finally:
            w.close()
        # Lop off the back half of the data records -> declared > actual size.
        full = os.path.getsize(path)
        with open(path, "r+b") as fh:
            fh.truncate(full // 2)
        with pytest.raises(CorruptFileError):
            Recording.from_file(path)


def test_mixed_rate_edf_raises_typed_but_valueerror():
    """The mixed-rate policy error is now MixedSamplingRateError yet still a
    ValueError, so existing callers keep catching it."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "psg.edf")
        w = pyedflib.EdfWriter(path, 2)
        try:
            w.setSignalHeaders(
                [
                    {
                        "label": "EEG",
                        "dimension": "uV",
                        "sample_frequency": 200.0,
                        "physical_max": 100.0,
                        "physical_min": -100.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    },
                    {
                        "label": "SpO2",
                        "dimension": "%",
                        "sample_frequency": 12.5,
                        "physical_max": 100.0,
                        "physical_min": 0.0,
                        "digital_max": 32767,
                        "digital_min": -32768,
                        "prefilter": "n/a",
                        "transducer": "n/a",
                    },
                ]
            )
            w.writeSamples([np.zeros(200), np.zeros(13)])
        finally:
            w.close()
        with pytest.raises(MixedSamplingRateError):
            Recording.from_file(path, mixed_rate="error")
        with pytest.raises(ValueError):  # back-compat
            Recording.from_file(path, mixed_rate="error")


@pytest.mark.skipif(not _HAS_MNE or not _MEG_FIF.exists(), reason="needs mne + the FIF fixture")
def test_incomplete_split_fif_chain_is_corrupt():
    """A split FIF whose chain is missing a member (split-01 present, split-02
    gone) -> CorruptFileError, via the real MEG importer (the on005261 split
    failure mode). read_raw_fif(split-01) follows the chain and raises when the
    next file is absent."""
    import mne

    full = mne.io.read_raw_fif(str(_MEG_FIF), preload=True, verbose="ERROR")
    with tempfile.TemporaryDirectory() as d:
        full.save(
            os.path.join(d, "sub-01_task-mouse_meg.fif"),
            split_size="1.5MB",
            split_naming="bids",
            verbose="ERROR",
        )
        first = os.path.join(d, "sub-01_task-mouse_split-01_meg.fif")
        second = os.path.join(d, "sub-01_task-mouse_split-02_meg.fif")
        assert os.path.exists(first) and os.path.exists(second)
        os.remove(second)  # break the chain
        with pytest.raises(CorruptFileError):
            Recording.from_file(first)


@pytest.mark.skipif(not _HAS_MNE or not _CTF_DS.exists(), reason="needs mne + the CTF fixture")
def test_truncated_ctf_meg4_is_corrupt():
    """A CTF .ds whose .meg4 is chopped to a non-multiple of its record size
    (the on004398 truncation) -> CorruptFileError, via the real CTF reader."""
    with tempfile.TemporaryDirectory() as d:
        ds = os.path.join(d, "catch-alp-good-f.ds")
        shutil.copytree(_CTF_DS, ds)
        meg4 = os.path.join(ds, "catch-alp-good-f.meg4")
        # Cut a few hundred bytes off the tail: the .meg4 is now (size-8) not a
        # clean int32*nchan*nsamp multiple, which read_raw_ctf rejects.
        with open(meg4, "r+b") as fh:
            fh.truncate(os.path.getsize(meg4) - 333)
        with pytest.raises(CorruptFileError):
            Recording.from_file(ds)


def _cut_copy(src, tmp_path, *, cut_bytes=0, keep_fraction=1.0):
    """Copy the fixture ``src`` into ``tmp_path`` and return the copy's path.

    The copy keeps ``keep_fraction`` of the file, less ``cut_bytes`` taken from
    the end of what remains. With the defaults it is intact.
    """
    dest = tmp_path / src.name
    shutil.copyfile(src, dest)
    keep = int(dest.stat().st_size * keep_fraction) - cut_bytes
    with open(dest, "r+b") as fh:
        fh.truncate(keep)
    return dest


@pytest.mark.skipif(not _HAS_MNE or not _MEG_FIF.exists(), reason="needs mne + the FIF fixture")
def test_intact_fif_still_reads_as_mne_reads_it(tmp_path):
    """Regression guard for the truncation mappings: an intact FIF still loads,
    with the channels, sampling rate and samples MNE itself reads from it."""
    import mne

    raw = mne.io.read_raw_fif(str(_MEG_FIF), preload=True, verbose="ERROR")
    rec = Recording.from_file(str(_cut_copy(_MEG_FIF, tmp_path)))

    assert len(rec.channels) == len(raw.ch_names)
    assert {info["sample_frequency"] for info in rec.channels.values()} == {raw.info["sfreq"]}
    assert rec.signals.shape == raw.get_data().T.shape
    np.testing.assert_allclose(rec.signals, raw.get_data().T)


# The fixture stores 4-byte (float32) samples. Cutting a multiple of 4 bytes from
# its last data buffer leaves a whole number of samples that no longer fill the
# buffer's declared block (a reshape failure); any other cut leaves a partial
# sample (a frombuffer failure). Halving the file is the plain truncated part.
@pytest.mark.skipif(not _HAS_MNE or not _MEG_FIF.exists(), reason="needs mne + the FIF fixture")
@pytest.mark.parametrize(
    ("cut", "numpy_text"),
    [
        pytest.param({"keep_fraction": 0.5}, None, id="half"),
        pytest.param(
            {"cut_bytes": 1002},
            "buffer size must be a multiple of element size",
            id="partial_sample",
        ),
        pytest.param({"cut_bytes": 1000}, "cannot reshape array of size", id="whole_samples"),
    ],
)
def test_truncated_fif_data_is_corrupt(tmp_path, cut, numpy_text):
    """A FIF cut inside its data raises CorruptFileError (corrupt_or_truncated),
    not the generic FileReadError, through the real MEG importer. The message
    names the file and not the directory it sits in; the numpy error MNE raised
    stays attached as the cause."""
    path = _cut_copy(_MEG_FIF, tmp_path, **cut)
    with pytest.raises(CorruptFileError) as caught:
        Recording.from_file(str(path))

    err = caught.value
    assert err.code == "corrupt_or_truncated"
    assert path.name in str(err)
    assert str(tmp_path) not in str(err) and tmp_path.name not in str(err)
    assert isinstance(err.__cause__, ValueError)
    if numpy_text is not None:
        assert numpy_text in str(err.__cause__)


@pytest.mark.skipif(not _HAS_MNE or not _KIT_SQD.exists(), reason="needs mne + the KIT fixture")
@pytest.mark.parametrize("keep_fraction", [0.7, 0.9])
def test_truncated_kit_data_is_corrupt(tmp_path, keep_fraction):
    """The truncation match is on numpy's message, so it applies to any reader:
    a KIT/Yokogawa file cut inside its data also raises CorruptFileError, and the
    message names the file and not its directory."""
    path = _cut_copy(_KIT_SQD, tmp_path, keep_fraction=keep_fraction)
    with pytest.raises(CorruptFileError) as caught:
        Recording.from_file(str(path))

    err = caught.value
    assert err.code == "corrupt_or_truncated"
    assert path.name in str(err)
    assert str(tmp_path) not in str(err) and tmp_path.name not in str(err)
    assert "cannot reshape array of size" in str(err.__cause__)


@pytest.mark.skipif(not _HAS_MNE or not _BTI_DIR.exists(), reason="needs mne + the BTi fixture")
@pytest.mark.parametrize("name", ["config", "hs_file"])
@pytest.mark.parametrize("keep_fraction", [0.5, 0.9])
def test_truncated_bti_metadata_is_corrupt(tmp_path, name, keep_fraction):
    """The 4D/BTi config and head-shape files raise the message MNE gives for a cut inside
    a data tag, so a cut one raises CorruptFileError naming the dataset directory. The
    4D data file does not raise that message and keeps its generic FileReadError."""
    bti = tmp_path / _BTI_DIR.name
    shutil.copytree(_BTI_DIR, bti)
    target = bti / name
    data = target.read_bytes()
    target.write_bytes(data[: int(len(data) * keep_fraction)])

    with pytest.raises(CorruptFileError) as caught:
        Recording.from_file(str(bti), importer="meg")

    err = caught.value
    assert err.code == "corrupt_or_truncated"
    assert _BTI_DIR.name in str(err)
    assert str(tmp_path) not in str(err)
    assert "buffer size must be a multiple of element size" in str(err)


@pytest.mark.skipif(not _HAS_MNE or not _BTI_DIR.exists(), reason="needs mne + the BTi fixture")
@pytest.mark.parametrize("keep_fraction", [0.5, 0.9])
def test_truncated_bti_data_keeps_the_generic_read_error(tmp_path, keep_fraction):
    """The 4D/BTi data file raises a message the truncation match does not cover, so it
    keeps the generic FileReadError rather than CorruptFileError."""
    bti = tmp_path / _BTI_DIR.name
    shutil.copytree(_BTI_DIR, bti)
    target = bti / "c,rfDC"
    data = target.read_bytes()
    target.write_bytes(data[: int(len(data) * keep_fraction)])

    with pytest.raises(FileReadError) as caught:
        Recording.from_file(str(bti), importer="meg")

    assert type(caught.value) is FileReadError
    assert caught.value.code == "file_read_error"


@pytest.mark.skipif(not _HAS_MNE or not _KIT_SQD.exists(), reason="needs mne + the KIT fixture")
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="#164: a KIT file cut to about 23 to 60 percent of its data reads as all zeros",
)
def test_kit_cut_that_reads_as_zeros_is_refused(tmp_path):
    """The known gap, pinned: a KIT file cut inside the range where it reads silently as
    zeros should be refused. Strict, so the mark is removed when #164 is fixed."""
    path = _cut_copy(_KIT_SQD, tmp_path, keep_fraction=0.4)
    with pytest.raises(CorruptFileError):
        Recording.from_file(str(path))


_FIFF_DATA_BUFFER = 300  # FIFF_DATA_BUFFER: the tag kind of one block of samples


def _fif_data_buffer_ends(path):
    """Byte offset at which each data-buffer tag of a FIF file ends.

    A FIF file is a chain of tags, each a 16-byte big-endian header (kind, type,
    size, next) followed by ``size`` bytes of payload; ``next`` is 0 when the
    following tag starts right after the payload, positive for an explicit
    position, and negative at the end of the chain.
    """
    blob = path.read_bytes()
    ends, pos = [], 0
    while pos + 16 <= len(blob):
        kind, _type, size, nxt = struct.unpack(">iIii", blob[pos : pos + 16])
        if kind == _FIFF_DATA_BUFFER:
            ends.append(pos + 16 + size)
        if nxt == 0:
            pos += 16 + size
        elif nxt > 0:
            pos = nxt
        else:
            break
    return ends


def _fif_cut_after_buffer(tmp_path, extra_bytes):
    """Copy the FIF fixture, cut it ``extra_bytes`` past the end of its middle
    data buffer, and return ``(path, buffers_kept)``."""
    path = _cut_copy(_MEG_FIF, tmp_path)
    ends = _fif_data_buffer_ends(path)
    assert len(ends) > 1
    kept = len(ends) // 2
    with open(path, "r+b") as fh:
        fh.truncate(ends[kept - 1] + extra_bytes)
    return path, kept


# A tag header is 16 bytes. MNE drops an incomplete header at the end of the file
# silently, so a cut within the 15 bytes after a buffer reads the same as a cut
# exactly at its end: as a shorter recording, with no error. That is data loss
# reported as success, so this asserts the expected behavior (a typed error) and
# is an expected failure until it is fixed.
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,  # "DID NOT RAISE"; any other error is a real failure
    reason="biosigio #163: a FIF cut exactly at a data-buffer boundary reads silently shorter",
)
@pytest.mark.skipif(not _HAS_MNE or not _MEG_FIF.exists(), reason="needs mne + the FIF fixture")
@pytest.mark.parametrize("extra_bytes", [0, 15], ids=["at_boundary", "partial_next_header"])
def test_fif_cut_at_a_data_buffer_boundary_is_corrupt(tmp_path, extra_bytes):
    """A FIF cut exactly where a data buffer ends, or within the 15 bytes after
    it, should raise CorruptFileError like any other cut of the data."""
    path, _ = _fif_cut_after_buffer(tmp_path, extra_bytes)
    with pytest.raises(CorruptFileError):
        Recording.from_file(str(path))


@pytest.mark.skipif(not _HAS_MNE or not _MEG_FIF.exists(), reason="needs mne + the FIF fixture")
def test_fif_cut_just_inside_a_data_buffer_is_corrupt(tmp_path):
    """The edge of the expected failure above: once the next buffer's complete
    header is present without its samples, the cut is inside a buffer and is
    detected."""
    path, _ = _fif_cut_after_buffer(tmp_path, 16)
    with pytest.raises(CorruptFileError) as caught:
        Recording.from_file(str(path))
    assert caught.value.code == "corrupt_or_truncated"
