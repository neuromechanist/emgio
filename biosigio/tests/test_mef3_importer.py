"""MEF3 iEEG importer (mne.io.read_raw_mef, new in MNE 1.12) on REAL sessions
written on the fly with pymef -- no mocks.

pymef can WRITE MEF3 sessions (it wraps the MEF3 C reference library used to
read them too), so every test here synthesizes a real ``.mefd`` session in
``tmp_path`` with known int32 sample values and reads it back through
``Recording.from_file``, exactly as a real acquisition would be read. This
exercises real MEF3 file I/O (universal headers, per-channel ``.tidx``/``.tdat``
segments, unit-scaling metadata), not a stand-in for it.

Skips cleanly if the optional ``mef3`` extra (mne>=1.12 + pymef) is not
installed, or if the installed MNE predates 1.12 (``read_raw_mef`` does not
exist yet).
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymef", reason="MEF3 import requires the optional 'mef3' extra (pymef)")
mne = pytest.importorskip("mne", reason="MEF3 import requires the optional 'mef3' extra (mne)")
if not hasattr(mne.io, "read_raw_mef"):
    pytest.skip(
        "MEF3 import needs mne>=1.12 (read_raw_mef); installed MNE predates it",
        allow_module_level=True,
    )

from biosigio import Recording  # noqa: E402
from biosigio.importers.mef3 import _mne_version_tuple, require_mne_mef  # noqa: E402

_START_TIME = 1_700_000_000_000_000  # arbitrary uUTC microseconds
_SECTION3 = {
    "recording_time_offset": 0,
    "DST_start_time": 0,
    "DST_end_time": 0,
    "GMT_offset": 0,
    "subject_name_1": b"Test",
    "subject_name_2": b"Subject",
    "subject_ID": b"sub-01",
    "recording_location": b"biosigio-test",
}


def _section2(
    raw: np.ndarray, sfreq: float, start_sample: int, units: bytes, factor: float
) -> dict:
    """A minimal, real MEF3 section-2 (per-segment) metadata dict.

    Fields that pymef/MEF3 recompute from the actual data on write (block
    counts/sizes, discontinuity counts, ...) are left at 0/placeholder, mirroring
    pymef's own test suite (``tests/test_pymef.py``).
    """
    return {
        "channel_description": b"Test_channel",
        "session_description": b"Test_session",
        "recording_duration": 1,
        "reference_description": b"n/a",
        "acquisition_channel_number": 0,
        "sampling_frequency": sfreq,
        "notch_filter_frequency_setting": 0.0,
        "low_frequency_filter_setting": 0.0,
        "high_frequency_filter_setting": 0.0,
        "AC_line_frequency": 60,
        "units_conversion_factor": factor,
        "units_description": units,
        "maximum_native_sample_value": float(raw.max()) if raw.size else 0.0,
        "minimum_native_sample_value": float(raw.min()) if raw.size else 0.0,
        "start_sample": start_sample,
        "number_of_blocks": 0,
        "maximum_block_bytes": 0,
        "maximum_block_samples": 0,
        "maximum_difference_bytes": 0,
        "block_interval": 0,
        "number_of_discontinuities": 0,
        "maximum_contiguous_blocks": 0,
        "maximum_contiguous_block_bytes": 0,
        "maximum_contiguous_samples": 0,
        "number_of_samples": int(raw.size),
    }


def _write_mef3_session(
    session_path,
    channels: dict[str, np.ndarray],
    sfreq: float,
    *,
    units: bytes = b"uV",
    units_conversion_factor: float = 1.0,
    password: str = "",
    password_1: str | None = None,
    password_2: str | None = None,
) -> None:
    """Write a real, single-segment-per-channel MEF3 session via pymef.

    Args:
        session_path: Target ``.mefd`` directory (created here; must not exist).
        channels: label -> int32 sample array (same length/rate for every
            channel, matching RawMEF's uniform-grid requirement).
        sfreq: Sampling rate shared by every channel.
        units: MEF3 ``units_description`` (e.g. ``b"uV"``).
        units_conversion_factor: MEF3 ``units_conversion_factor``.
        password: Level 1 AND level 2 password when both are the same; "" (the
            default) for an unencrypted session. For a genuinely encrypted
            session pass DISTINCT ``password_1``/``password_2`` instead --
            MEF3 requires the two levels to differ (using the same value for
            both breaks decryption; verified empirically: a session written
            with password_1 == password_2 fails to read back at all, even with
            the correct password). Reading an encrypted session needs the
            level-2 (owner) password.
        password_1: Level 1 password; overrides ``password`` when given.
        password_2: Level 2 (owner) password; overrides ``password`` when given.
    """
    from pymef.mef_session import MefSession

    pw1 = password if password_1 is None else password_1
    pw2 = password if password_2 is None else password_2

    session_path.mkdir(parents=True, exist_ok=False)
    n_samples = {len(data) for data in channels.values()}
    if len(n_samples) != 1:
        raise ValueError("all channels must share the same sample count")
    n = n_samples.pop()
    end_time = int(_START_TIME + 1e6 * n / sfreq)

    ms = MefSession(str(session_path), pw2, read_metadata=False)
    try:
        for label, data in channels.items():
            data = np.asarray(data, dtype="int32")
            sec2 = _section2(data, sfreq, 0, units, units_conversion_factor)
            ms.write_mef_ts_segment_metadata(
                label, 0, pw1, pw2, _START_TIME, end_time, sec2, _SECTION3
            )
            ms.write_mef_ts_segment_data(label, 0, pw1, pw2, int(sfreq), data)
    finally:
        ms.close()


@pytest.fixture
def mef3_session(tmp_path):
    """A real 3-channel, 500 Hz, 2 s MEF3 session with known ramp/sine values."""
    sfreq = 500.0
    n = int(sfreq * 2)
    t = np.arange(n)
    channels = {
        "ch01": (t - n // 2).astype("int32"),  # deterministic ramp, incl. negatives
        "ch02": (200 * np.sin(2 * np.pi * 3 * t / sfreq)).astype("int32"),
        "ch03": np.zeros(n, dtype="int32"),
    }
    path = tmp_path / "sub-01_task-test_ieeg.mefd"
    _write_mef3_session(path, channels, sfreq)
    return path, channels, sfreq


def test_mef3_channel_count_and_rate(mef3_session):
    path, channels, sfreq = mef3_session
    rec = Recording.from_file(str(path))
    assert rec.get_n_channels() == len(channels)
    assert rec.get_sampling_frequency() == sfreq
    assert rec.get_n_samples() == len(next(iter(channels.values())))
    assert set(rec.channels.keys()) == set(channels.keys())


def test_mef3_channel_type_and_modality(mef3_session):
    """MEF3 channels default to SEEG (no per-channel modality in the format itself)."""
    path, _channels, _sfreq = mef3_session
    rec = Recording.from_file(str(path))
    for info in rec.channels.values():
        assert info["channel_type"] == "SEEG"
        assert info["modality"] == "IEEG"


def test_mef3_units_are_volts(mef3_session):
    """uV samples convert to the FIFF-volt convention every MNE-backed importer uses."""
    path, _channels, _sfreq = mef3_session
    rec = Recording.from_file(str(path))
    for info in rec.channels.values():
        assert info["physical_dimension"] == "V"


def test_mef3_sample_values_roundtrip(mef3_session):
    """The exact int32 samples written survive the write -> read_raw_mef pipeline.

    units_description=uV, units_conversion_factor=1.0 -> MNE scales by 1e-6 to
    reach volts, so undoing that (x 1e6) must reproduce the original integers.
    """
    path, channels, _sfreq = mef3_session
    rec = Recording.from_file(str(path))
    for label, original in channels.items():
        recovered_uv = rec.signals[label].to_numpy() * 1e6
        np.testing.assert_allclose(recovered_uv, original.astype(np.float64), atol=1e-6)


def test_mef3_source_format_recorded(mef3_session):
    path, _channels, _sfreq = mef3_session
    rec = Recording.from_file(str(path))
    assert rec.metadata.get("source_format") == "mef3"
    assert rec.metadata.get("source_file") == str(path)


def test_mef3_different_units_conversion_factor(tmp_path):
    """units_conversion_factor scales the recovered value, not just units_description."""
    sfreq = 200.0
    n = 400
    data = (np.arange(n) - 100).astype("int32")
    path = tmp_path / "sub-02_task-test_ieeg.mefd"
    _write_mef3_session(path, {"ch01": data}, sfreq, units=b"uV", units_conversion_factor=2.5)
    rec = Recording.from_file(str(path))
    # physical (V) = digital * units_conversion_factor * (uV->V scale 1e-6)
    expected_v = data.astype(np.float64) * 2.5 * 1e-6
    np.testing.assert_allclose(rec.signals["ch01"].to_numpy(), expected_v, atol=1e-12)


def test_mef3_toc_gap_becomes_event(tmp_path):
    """A real inter-segment gap (two segments, 1 s apart) surfaces as a
    BAD_ACQ_SKIP event -- MEF3's table-of-contents gaps are read as MNE
    annotations, which biosigio reads into ``rec.events`` (mirrors how
    BrainVisionImporter reads ``.vmrk`` markers)."""
    from pymef.mef_session import MefSession

    sfreq = 250.0
    secs1, gap_s, secs2 = 2, 1, 2
    n1, n2 = int(sfreq * secs1), int(sfreq * secs2)
    seg1_end = int(_START_TIME + 1e6 * secs1)
    seg2_start = int(seg1_end + 1e6 * gap_s)
    seg2_end = int(seg2_start + 1e6 * secs2)

    path = tmp_path / "sub-03_task-test_ieeg.mefd"
    path.mkdir()
    ms = MefSession(str(path), "", read_metadata=False)
    try:
        raw1 = np.arange(n1, dtype="int32")
        sec2_1 = _section2(raw1, sfreq, 0, b"uV", 1.0)
        ms.write_mef_ts_segment_metadata(
            "ch01", 0, "", "", _START_TIME, seg1_end, sec2_1, _SECTION3
        )
        ms.write_mef_ts_segment_data("ch01", 0, "", "", int(sfreq), raw1)

        raw2 = np.arange(n2, dtype="int32")
        start_sample = n1 + int(gap_s * sfreq)
        sec2_2 = _section2(raw2, sfreq, start_sample, b"uV", 1.0)
        ms.write_mef_ts_segment_metadata("ch01", 1, "", "", seg2_start, seg2_end, sec2_2, _SECTION3)
        ms.write_mef_ts_segment_data("ch01", 1, "", "", int(sfreq), raw2)
    finally:
        ms.close()

    rec = Recording.from_file(str(path))
    assert isinstance(rec.events, pd.DataFrame)
    assert len(rec.events) == 1
    row = rec.events.iloc[0]
    assert row["description"] == "BAD_ACQ_SKIP"
    assert row["onset"] == pytest.approx(secs1)
    assert row["duration"] == pytest.approx(gap_s)


def test_mef3_password_roundtrips_when_empty(mef3_session):
    """The default password="" (unencrypted) load path works end to end."""
    path, channels, _sfreq = mef3_session
    rec = Recording.from_file(str(path), importer="mef3", password="")
    assert rec.get_n_channels() == len(channels)


# -- Encrypted sessions (real pymef password_1/password_2, no mocks) -------------
#
# password="" above only exercises the DEFAULT, already covered by every other
# test in this file; it says nothing about the encrypted branch. pymef writes
# real encrypted MEF3 sessions (level 1 + level 2 passwords), so these tests
# write one and cover both directions: the correct password reads and
# round-trips exactly, and a wrong/empty password against an encrypted session
# is rejected.


def test_mef3_encrypted_session_correct_password_roundtrips(tmp_path):
    sfreq = 250.0
    n = 500
    data = (np.arange(n, dtype="int32") - 250).astype("int32")
    path = tmp_path / "sub-04_task-test_ieeg.mefd"
    _write_mef3_session(
        path, {"ch01": data}, sfreq, password_1="levelonepw", password_2="leveltwopw"
    )

    # The level-2 (owner) password is the "correct" one for a full read.
    rec = Recording.from_file(str(path), importer="mef3", password="leveltwopw")
    assert rec.get_n_channels() == 1
    recovered_uv = rec.signals["ch01"].to_numpy() * 1e6
    np.testing.assert_allclose(recovered_uv, data.astype(np.float64), atol=1e-6)


def test_mef3_encrypted_session_wrong_password_raises_typed_error(tmp_path):
    """A wrong password raises a typed BiosigIOError (specifically FileReadError,
    the generic fallback -- classify_read_error has no rule matching "password"
    or "invalid", so it does not get its own error code). Verified by executing
    the real pymef/MNE path: pymef raises RuntimeError("MEF password is
    invalid"), which classify_read_error wraps as FileReadError rather than
    leaving it a bare RuntimeError.

    Judgement call: NOT adding a dedicated error code for this. It is a narrow,
    low-frequency failure mode that is already correctly typed (a stable
    BiosigIOError/.code="file_read_error"); a new taxonomy entry has downstream
    effects on how NEMAR surfaces failure reasons and is out of scope here.
    """
    from biosigio.exceptions import FileReadError

    sfreq = 250.0
    n = 500
    data = (np.arange(n, dtype="int32") - 250).astype("int32")
    path = tmp_path / "sub-05_task-test_ieeg.mefd"
    _write_mef3_session(
        path, {"ch01": data}, sfreq, password_1="levelonepw", password_2="leveltwopw"
    )

    with pytest.raises(FileReadError, match="(?i)password"):
        Recording.from_file(str(path), importer="mef3", password="wrongpassword")


def test_mef3_encrypted_session_empty_password_raises_typed_error(tmp_path):
    """An empty password against a session that IS encrypted is rejected the
    same way an outright wrong password is (both surface as pymef's "MEF
    password is invalid", classified as FileReadError)."""
    from biosigio.exceptions import FileReadError

    sfreq = 250.0
    n = 500
    data = (np.arange(n, dtype="int32") - 250).astype("int32")
    path = tmp_path / "sub-06_task-test_ieeg.mefd"
    _write_mef3_session(
        path, {"ch01": data}, sfreq, password_1="levelonepw", password_2="leveltwopw"
    )

    with pytest.raises(FileReadError, match="(?i)password"):
        Recording.from_file(str(path), importer="mef3", password="")


def test_mef3_nonmonotonic_channel_labels_do_not_get_reordered(tmp_path):
    """Labels that sort differently alphabetically than they were inserted
    (ch1, ch2, ch3, ch10, ch20 alphabetically vs. the insertion order below)
    must not get silently reshuffled anywhere in the pipeline. Each channel
    carries a distinct constant value equal to its own numeric suffix, so a
    label/data mismatch (e.g. an accidental alphabetical sort somewhere)
    shows up immediately as a wrong value under a channel's own name, not just
    as a count/shape mismatch. The real on006392 dataset has 194 channels, so
    ordering at scale is a real condition, not a hypothetical one."""
    sfreq = 200.0
    n = 100
    labels = ["ch10", "ch2", "ch1", "ch20", "ch3"]  # deliberately non-monotonic
    channels = {label: np.full(n, int(label[2:]), dtype="int32") for label in labels}
    path = tmp_path / "sub-07_task-test_ieeg.mefd"
    _write_mef3_session(path, channels, sfreq)

    rec = Recording.from_file(str(path))
    assert set(rec.channels.keys()) == set(labels)
    for label in labels:
        expected_uv = float(label[2:])
        recovered_uv = rec.signals[label].to_numpy() * 1e6
        np.testing.assert_allclose(
            recovered_uv, np.full(n, expected_uv), atol=1e-6, err_msg=f"channel {label}"
        )


# -- Format dispatch (no fixture needed; just the extension -> importer mapping) --


def test_mef3_extension_dispatches_to_mef3_importer():
    assert Recording._infer_importer("sub-01/ieeg/sub-01_task-x_ieeg.mefd") == "mef3"


def test_mef3_trailing_slash_dispatches_to_mef3():
    """A .mefd passed as a directory path (trailing slash) still resolves."""
    assert Recording._infer_importer("sub-01/ieeg/sub-01_task-x_ieeg.mefd/") == "mef3"


# -- MNE/pymef version gate (require_mne_mef) ------------------------------------


def test_mne_version_tuple_parses_plain_and_prerelease():
    assert _mne_version_tuple("1.12.1") == (1, 12)
    assert _mne_version_tuple("1.6") == (1, 6)
    assert _mne_version_tuple("1.13.0.dev0") == (1, 13)


def test_mne_version_tuple_unparseable_is_lowest():
    assert _mne_version_tuple("not-a-version") == (0, 0)


def test_mne_version_tuple_compares_numerically_not_lexicographically():
    """ "1.9" < "1.12" as strings is FALSE ("1.9" > "1.12" lexicographically,
    since '9' > '1'); the parsed tuple comparison must get this right."""
    assert _mne_version_tuple("1.9") < _mne_version_tuple("1.12")
    assert not ("1.9" < "1.12")  # sanity check: confirms the string trap is real


def test_require_mne_mef_accepts_exact_floor_version(monkeypatch):
    """ "1.12.0" is the exact floor -- it must NOT raise (a strictly-greater-than
    check would wrongly reject the floor version itself)."""
    monkeypatch.setattr(mne, "__version__", "1.12.0")
    mne_mod = require_mne_mef()
    assert mne_mod is mne


def test_require_mne_mef_succeeds_with_real_environment():
    """The actual installed mne/pymef in this test environment satisfy the MEF3
    floor (mne>=1.12); this exercises the real gate, not a simulated one."""
    mne_mod = require_mne_mef()
    assert mne_mod.__name__ == "mne"
    assert hasattr(mne_mod.io, "read_raw_mef")


def test_mef3_missing_file_raises_typed_error(tmp_path):
    """A nonexistent .mefd path is classified into a typed biosigIO error, not a
    bare pymef/MNE exception."""
    from biosigio.exceptions import BiosigIOError

    missing = tmp_path / "does_not_exist.mefd"
    with pytest.raises(BiosigIOError):
        Recording.from_file(str(missing))


def test_require_mne_mef_raises_for_old_mne_version(monkeypatch):
    """The version gate fires a clear, actionable error when MNE predates 1.12.

    Patches only the ``mne.__version__`` string mne itself exposes (reverted
    automatically by monkeypatch); ``require_mne_mef``'s own comparison logic
    runs for real against that value -- nothing about the gate itself is faked."""
    monkeypatch.setattr(mne, "__version__", "1.11.0")
    with pytest.raises(ImportError, match="mne>=1.12"):
        require_mne_mef()


def test_require_mne_mef_raises_when_pymef_missing(monkeypatch):
    """The pymef-missing branch fires when mne is new enough but pymef isn't
    importable. ``sys.modules["pymef"] = None`` is the standard way to make
    Python's own import machinery raise ImportError for a module without
    actually uninstalling it (see the import system docs); the code under test
    (``require_mne_mef``) still runs its real try/except for real.

    Matches on "uv sync --extra mef3" (our own install-hint text), NOT on the
    bare word "pymef": Python's own ModuleNotFoundError message for a
    sys.modules-halted import ("import of pymef halted; None in sys.modules")
    already contains "pymef", so a match="pymef" assertion would pass even if
    biosigio's own try/except-and-reraise around the import were deleted
    entirely -- it needs to prove OUR message fired, not just that some
    ImportError mentioning "pymef" propagated.
    """
    import sys

    monkeypatch.setitem(sys.modules, "pymef", None)
    with pytest.raises(ImportError, match="uv sync --extra mef3"):
        require_mne_mef()
