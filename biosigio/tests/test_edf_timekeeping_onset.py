"""A late first time-keeping onset in an EDF+/BDF+ file (issue #145).

An EDF+ or BDF+ file stores a time-stamped annotation list (TAL) in every data
record, and the first TAL of a record is the record's "time-keeping" onset. The
specification asks the first record to start at 0 s, and EDFlib (inside pyedflib)
refuses a file whose first time-keeping onset is 1 s or more, reporting only the
generic "(it contains format errors)". A recording cut out of a longer session
at a data-record boundary keeps its session-relative TALs, so it starts late.

Every fixture is a REAL file on disk, written with ``pyedflib.EdfWriter``. A
"block" is a byte-exact cut of the session's records ``[first, first + count)``
with the record count patched: the late-onset shape the issue reports, here
with a first onset of 40 s. No mocks, no fake readers.
"""

from __future__ import annotations

import errno
import os
import re
import signal
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
import pytest

from biosigio.exceptions import BiosigIOError, CorruptFileError, MixedSamplingRateError
from biosigio.exporters.zarr_stream import _open_stream_source
from biosigio.importers._edf_tolerant import probe_timekeeping_onset
from biosigio.importers.edf import EDFImporter

_ONSET_AT_BLOCK_START = 40
# Chosen so that a first onset of 40 s carries over a minute, an hour and a day.
_SESSION_START = datetime(2026, 10, 15, 23, 59, 45)
_TAL_ONSET = re.compile(rb"([+-])([0-9]+(?:\.[0-9]+)?)(?=[\x14\x15])")


@dataclass(frozen=True)
class _Layout:
    """Where the annotation signals sit in a file, read from the raw bytes."""

    header_bytes: int
    n_records: int
    record_bytes: int
    # (offset inside a data record, width in bytes) of each annotation signal
    annotation_blocks: tuple[tuple[int, int], ...]


def _layout(path: str | Path, bdf: bool) -> _Layout:
    """Parse the fixed-width header directly, independent of the code under test."""
    raw = Path(path).read_bytes()[:65536]
    n_signals = int(raw[252:256])
    sample_size = 3 if bdf else 2
    labels = [raw[256 + 16 * i : 256 + 16 * (i + 1)] for i in range(n_signals)]
    spr_at = 256 + n_signals * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    samples = [int(raw[spr_at + 8 * i : spr_at + 8 * (i + 1)]) for i in range(n_signals)]
    wanted = b"BDF Annotations " if bdf else b"EDF Annotations "
    blocks = []
    offset = 0
    for label, n in zip(labels, samples, strict=True):
        if label == wanted:
            blocks.append((offset, n * sample_size))
        offset += n * sample_size
    return _Layout(
        header_bytes=int(raw[184:192]),
        n_records=int(raw[236:244]),
        record_bytes=offset,
        annotation_blocks=tuple(blocks),
    )


def _write_session(
    path: str,
    *,
    bdf: bool,
    n_records: int,
    record_duration: float = 1.0,
    rates: tuple[float, ...] = (32.0, 32.0),
    annotation_signals: int = 1,
    start: datetime = _SESSION_START,
    seed: int = 145,
) -> None:
    """Write a compliant EDF+C/BDF+C session with one annotation per annotation signal per record."""
    rng = np.random.default_rng(seed)
    file_type = pyedflib.FILETYPE_BDFPLUS if bdf else pyedflib.FILETYPE_EDFPLUS
    dmax = 8388607 if bdf else 32767
    headers = [
        {
            "label": f"EMG{i}",
            "dimension": "uV",
            "sample_frequency": rate,
            "physical_max": 500.0,
            "physical_min": -500.0,
            "digital_max": dmax,
            "digital_min": -dmax - 1,
            "transducer": "",
            "prefilter": "",
        }
        for i, rate in enumerate(rates)
    ]
    writer = pyedflib.EdfWriter(path, len(rates), file_type=file_type)
    try:
        writer.setSignalHeaders(headers)
        writer.setStartdatetime(start)
        if record_duration != 1.0:
            with warnings.catch_warnings():
                # pyedflib warns that a forced record duration may alter the
                # sample frequencies it reads back; here they are what we set.
                warnings.simplefilter("ignore", UserWarning)
                writer.setDatarecordDuration(record_duration)
        if annotation_signals > 1:
            writer.set_number_of_annotation_signals(annotation_signals)
        # The writer fills each record's annotation signals in turn, so with N
        # signals the j-th event lands in record j // N.
        for j in range(n_records * annotation_signals):
            onset = (j + 0.25) * record_duration / annotation_signals
            writer.writeAnnotation(onset, 0.1, f"event {j}")
        writer.writeSamples(
            [rng.uniform(-400, 400, round(n_records * record_duration * rate)) for rate in rates]
        )
    finally:
        writer.close()


def _cut_block(src: str, dst: str, bdf: bool, *, first: int, count: int) -> None:
    """Cut records ``[first, first + count)`` out byte for byte; patch the count."""
    layout = _layout(src, bdf)
    raw = Path(src).read_bytes()
    start = layout.header_bytes + first * layout.record_bytes
    end = start + count * layout.record_bytes
    header = bytearray(raw[: layout.header_bytes])
    header[236:244] = str(count).ljust(8).encode("ascii")
    Path(dst).write_bytes(bytes(header) + raw[start:end])


def _make_block(
    tmp_path: Path,
    bdf: bool,
    *,
    first: int = _ONSET_AT_BLOCK_START,
    count: int = 10,
    record_duration: float = 1.0,
    rates: tuple[float, ...] = (32.0, 32.0),
    annotation_signals: int = 1,
    start: datetime = _SESSION_START,
) -> tuple[str, str]:
    """A compliant session and the late-onset block cut from it."""
    ext = "bdf" if bdf else "edf"
    session = str(tmp_path / f"session.{ext}")
    block = str(tmp_path / f"block.{ext}")
    _write_session(
        session,
        bdf=bdf,
        n_records=first + count,
        record_duration=record_duration,
        rates=rates,
        annotation_signals=annotation_signals,
        start=start,
    )
    _cut_block(session, block, bdf, first=first, count=count)
    return session, block


def _shift_block(block: bytes, delta: Decimal, *, timekeeping_only: bool) -> bytes:
    """Add ``delta`` seconds to the TAL onsets of one annotation block.

    Written for the writer's own layout: whole TALs separated by single NULs,
    zero-padded to the block width. ``timekeeping_only`` leaves every TAL that
    carries an annotation untouched.
    """
    out = bytearray()
    position = 0
    for match in re.finditer(rb"[^\x00]+", block):
        out += block[position : match.start()]
        tal = match.group()
        onset = _TAL_ONSET.match(tal)
        assert onset is not None, tal
        if timekeeping_only and not tal.endswith(b"\x14\x14"):
            out += tal
        else:
            fraction = onset.group(2).partition(b".")[2]
            value = Decimal(onset.group(2).decode())
            shifted = (-value if onset.group(1) == b"-" else value) + delta
            sign = "-" if shifted < 0 else "+"
            text = f"{abs(shifted):.{len(fraction)}f}"
            out += (sign + text).encode("ascii") + tal[onset.end() :]
        position = match.end()
    assert len(out) <= len(block), "the shifted TALs no longer fit the block"
    return bytes(out) + b"\x00" * (len(block) - len(out))


def _shift_onsets(
    src: str, dst: str, bdf: bool, delta: int, *, timekeeping_only: bool = False
) -> None:
    """Copy ``src`` to ``dst`` with every TAL onset moved by ``delta`` seconds."""
    layout = _layout(src, bdf)
    data = bytearray(Path(src).read_bytes())
    for record in range(layout.n_records):
        for offset, width in layout.annotation_blocks:
            at = layout.header_bytes + record * layout.record_bytes + offset
            data[at : at + width] = _shift_block(
                bytes(data[at : at + width]), Decimal(delta), timekeeping_only=timekeeping_only
            )
    Path(dst).write_bytes(bytes(data))


def _refusal(path: str) -> OSError:
    """The exception pyedflib raises for ``path``."""
    with pytest.raises(OSError) as info:
        pyedflib.EdfReader(path)
    return info.value


def _patch(path: str, bdf: bool, record: int, old: bytes, new: bytes) -> None:
    """Replace ``old`` with ``new`` in a record's first annotation block, keeping its width.

    A longer ``new`` takes the room from the NULs that pad the block's end.
    """
    layout = _layout(path, bdf)
    offset, width = layout.annotation_blocks[0]
    at = layout.header_bytes + record * layout.record_bytes + offset
    data = bytearray(Path(path).read_bytes())
    block = bytes(data[at : at + width])
    assert old in block, (old, block)
    patched = block.replace(old, new, 1)
    if len(patched) > width:
        assert patched[width:] == b"\x00" * (len(patched) - width), "no room to grow"
        patched = patched[:width]
    data[at : at + width] = patched.ljust(width, b"\x00")
    Path(path).write_bytes(bytes(data))


def _stamped_from(tmp_path: Path, bdf: bool, digits: str, *, gap_at: int | None = None) -> str:
    """A block whose time-keeping TALs read ``digits`` seconds, then +1, +2, ... per record.

    Rebuilt from a writer block with the annotation signal widened to hold the
    digits. ``digits`` is a string because the longest ones exceed what ``int``
    will format; its last three digits take the record number. From record
    ``gap_at`` on, the onsets skip one second: a gap EDFlib refuses.
    """
    _, block = _make_block(tmp_path, bdf)
    layout = _layout(block, bdf)
    raw = Path(block).read_bytes()
    sample_size = 3 if bdf else 2
    offset, width = layout.annotation_blocks[0]
    samples = max(width // sample_size, -(-(2 * len(digits) + 80) // sample_size))
    new_width = samples * sample_size
    n_signals = int(raw[252:256])
    index = n_signals - 1  # the writer puts the annotation signal last
    label_at = 256 + 16 * index
    assert raw[label_at : label_at + 16] == (b"BDF Annotations " if bdf else b"EDF Annotations ")
    spr_at = 256 + n_signals * 216 + 8 * index  # after label ... prefilter
    out = bytearray(raw[: layout.header_bytes])
    out[spr_at : spr_at + 8] = str(samples).ljust(8).encode("ascii")
    for record in range(layout.n_records):
        start = layout.header_bytes + record * layout.record_bytes
        late = record + (1 if gap_at is not None and record >= gap_at else 0)
        onset = digits[:-3] + f"{int(digits[-3:]) + late:03d}"
        tals = f"+{onset}\x14\x14\x00+{onset}.2500\x150.1000\x14event {record}\x14\x00"
        out += raw[start : start + offset]
        out += tals.encode("ascii").ljust(new_width, b"\x00")
        out += raw[start + offset + width : start + layout.record_bytes]
    stamped = tmp_path / f"stamped-{len(digits)}.{'bdf' if bdf else 'edf'}"
    stamped.write_bytes(bytes(out))
    return str(stamped)


@pytest.fixture(params=[False, True], ids=["edf", "bdf"])
def bdf(request) -> bool:
    return request.param


# --- The defect, as pyedflib reports it -----------------------------------------


def test_block_reproduces_defect(tmp_path, bdf):
    """pyedflib refuses the block; the first onset alone, or the type alone, is the cause."""
    session, block = _make_block(tmp_path, bdf)
    pyedflib.EdfReader(session).close()

    with pytest.raises(OSError, match=r"contains format errors"):
        pyedflib.EdfReader(block)

    corrected = str(tmp_path / "corrected")
    _shift_onsets(block, corrected, bdf, -_ONSET_AT_BLOCK_START, timekeeping_only=True)
    pyedflib.EdfReader(corrected).close()

    plain = str(tmp_path / "plain")
    data = bytearray(Path(block).read_bytes())
    assert data[192:197] == (b"BDF+C" if bdf else b"EDF+C")
    data[192:236] = b" " * 44
    Path(plain).write_bytes(bytes(data))
    pyedflib.EdfReader(plain).close()


# --- What must still be refused -------------------------------------------------


def _case_continuity_break(tmp_path: Path, bdf: bool) -> str:
    _, block = _make_block(tmp_path, bdf)
    _patch(block, bdf, 5, b"+45\x14\x14", b"+46\x14\x14")
    return block


def _case_malformed_later_duration(tmp_path: Path, bdf: bool) -> str:
    _, block = _make_block(tmp_path, bdf)
    _patch(block, bdf, 5, b"\x150.1000\x14", b"\x150.1x00\x14")
    return block


def _case_non_ascii_patient_byte(tmp_path: Path, bdf: bool) -> str:
    _, block = _make_block(tmp_path, bdf)
    data = bytearray(Path(block).read_bytes())
    data[8:88] = b"X" + b"\xe9" + b" " * 78
    Path(block).write_bytes(bytes(data))
    return block


def _case_non_ascii_data_signal_header(tmp_path: Path, bdf: bool) -> str:
    """A defect in a data signal's own header: a TAL-only copy would not carry it."""
    _, block = _make_block(tmp_path, bdf)
    data = bytearray(Path(block).read_bytes())
    n_signals = int(data[252:256])
    data[256 + n_signals * 224] = 0xE9  # first byte of the first signal's reserved field
    Path(block).write_bytes(bytes(data))
    return block


def _case_truncated_block(tmp_path: Path, bdf: bool) -> str:
    _, block = _make_block(tmp_path, bdf)
    data = Path(block).read_bytes()
    Path(block).write_bytes(data[:-100])
    return block


def _case_small_onset_with_later_defect(tmp_path: Path, bdf: bool) -> str:
    """A first onset of 0.5 s is not the late-onset defect: the later TAL is."""
    _, block = _make_block(tmp_path, bdf, first=1, count=10, record_duration=0.5)
    _patch(block, bdf, 5, b"\x150.1000\x14", b"\x150.1x00\x14")
    return block


def _case_unterminated_annotation_block(tmp_path: Path, bdf: bool) -> str:
    """A later block that does not end in a NUL: padding after the shift must not hide it."""
    _, block = _make_block(tmp_path, bdf)
    layout = _layout(block, bdf)
    offset, width = layout.annotation_blocks[0]
    at = layout.header_bytes + 5 * layout.record_bytes + offset
    head = b"+45\x14\x14\x00+45\x14"
    crafted = head + b"x" * (width - len(head) - 1) + b"\x14"  # a whole TAL, minus its NUL
    data = bytearray(Path(block).read_bytes())
    data[at : at + width] = crafted
    Path(block).write_bytes(bytes(data))
    return block


def _case_shifted_tal_no_longer_fits(tmp_path: Path, bdf: bool) -> str:
    """A full block whose TALs grow when brought back (+5 becomes -35): they overflow it."""
    _, block = _make_block(tmp_path, bdf, count=3)
    layout = _layout(block, bdf)
    offset, width = layout.annotation_blocks[0]
    at = layout.header_bytes + offset
    # "+40" gains nothing from the shift (it becomes "+0"), but "+5" and "+6" each
    # gain a character, so the shifted block is one byte longer than the block.
    head = b"+40\x14\x14\x00+5\x14a\x14\x00+6\x14"
    tail = b"\x14\x00"
    crafted = head + b"x" * (width - len(head) - len(tail)) + tail
    assert len(crafted) == width
    data = bytearray(Path(block).read_bytes())
    data[at : at + width] = crafted
    Path(block).write_bytes(bytes(data))
    return block


def _case_timekeeping_tal_without_onset(tmp_path: Path, bdf: bool) -> str:
    """Record 0 opens with a TAL whose onset has no sign: there is no first onset to bring back."""
    _, block = _make_block(tmp_path, bdf)
    _patch(block, bdf, 0, b"+40\x14\x14", b"x40\x14\x14")
    return block


def _case_timekeeping_tal_with_a_duration(tmp_path: Path, bdf: bool) -> str:
    """Record 0 opens with an onset and a duration: not the bare onset of a time-keeping TAL."""
    _, block = _make_block(tmp_path, bdf)
    _patch(block, bdf, 0, b"+40\x14\x14", b"+40\x151\x14\x14")
    return block


def _case_timekeeping_tal_with_text(tmp_path: Path, bdf: bool) -> str:
    """Record 0 opens with an annotation, not a time-keeping TAL."""
    _, block = _make_block(tmp_path, bdf)
    _patch(block, bdf, 0, b"+40\x14\x14", b"+40\x14x\x14")
    return block


def _case_later_tal_without_sign(tmp_path: Path, bdf: bool) -> str:
    """A later TAL whose onset has no sign cannot be brought back, and is not guessed at."""
    _, block = _make_block(tmp_path, bdf)
    _patch(block, bdf, 5, b"+45.25", b"x45.25")
    return block


def _case_timekeeping_onset_with_no_sign(tmp_path: Path, bdf: bool) -> str:
    """Record 0 opens with an UNSIGNED onset: bare digits, as an optional sign would take them.

    The "x" cases above swap the sign for a character no onset pattern accepts,
    so a pattern whose sign is optional still declines them. Dropping the sign
    leaves the digits themselves a valid onset, and only EDFlib's rule that an
    onset starts with ``+`` or ``-`` keeps the file refused.
    """
    _, block = _make_block(tmp_path, bdf)
    _patch(block, bdf, 0, b"+40\x14\x14", b"40\x14\x14")
    return block


def _case_later_onset_with_no_sign(tmp_path: Path, bdf: bool) -> str:
    """A later TAL opens with an UNSIGNED fractional onset: the digits are a number, the sign is missing."""
    _, block = _make_block(tmp_path, bdf)
    _patch(block, bdf, 5, b"+45.25", b"45.25")
    return block


_STILL_CORRUPT: list[Callable[[Path, bool], str]] = [
    _case_continuity_break,
    _case_malformed_later_duration,
    _case_non_ascii_patient_byte,
    _case_non_ascii_data_signal_header,
    _case_truncated_block,
    _case_unterminated_annotation_block,
    _case_small_onset_with_later_defect,
    _case_shifted_tal_no_longer_fits,
    _case_timekeeping_tal_without_onset,
    _case_timekeeping_tal_with_a_duration,
    _case_timekeeping_tal_with_text,
    _case_later_tal_without_sign,
    _case_timekeeping_onset_with_no_sign,
    _case_later_onset_with_no_sign,
]


# EDFlib's generic refusal, as the probe's first gate wants it. Some of the files
# above are refused for another reason (a bad header field, a short file), and a
# test that only reads the reader's real message would never get the probe past
# that gate: the generic message makes the probe judge the bytes itself.
_GENERIC_REFUSAL = OSError(
    "x: the file is not EDF(+) or BDF(+) compliant (it contains format errors)"
)


def _assert_probe_declines(path: str) -> None:
    assert probe_timekeeping_onset(path, _GENERIC_REFUSAL) is None


@pytest.mark.parametrize("make_case", _STILL_CORRUPT, ids=lambda f: f.__name__[len("_case_") :])
def test_still_corrupt_in_memory(tmp_path, bdf, make_case):
    path = make_case(tmp_path, bdf)
    _assert_probe_declines(path)
    with pytest.raises(CorruptFileError):
        EDFImporter().load(path)


@pytest.mark.parametrize("make_case", _STILL_CORRUPT, ids=lambda f: f.__name__[len("_case_") :])
def test_still_corrupt_streaming(tmp_path, bdf, make_case):
    path = make_case(tmp_path, bdf)
    _assert_probe_declines(path)
    with pytest.raises(CorruptFileError):
        _open_stream_source(path, None)


# --- The guard that recognizes the defect --------------------------------------


def test_probe_reports_first_onset(tmp_path, bdf):
    _, block = _make_block(tmp_path, bdf)
    found = probe_timekeeping_onset(block, _refusal(block))
    assert found is not None
    assert found.first_onset == 40.0
    # Onsets are relative to the first record, as for a compliant file.
    np.testing.assert_allclose(found.onsets, 0.25 + np.arange(10))
    np.testing.assert_allclose(found.durations, 0.1)
    assert list(found.descriptions) == [f"event {k}" for k in range(40, 50)]


def test_probe_reports_a_fractional_first_onset(tmp_path, bdf):
    _, block = _make_block(tmp_path, bdf, first=3, count=10, record_duration=0.5)
    found = probe_timekeeping_onset(block, _refusal(block))
    assert found is not None
    assert found.first_onset == 1.5
    np.testing.assert_allclose(found.onsets, (np.arange(10) + 0.25) * 0.5)


def test_probe_covers_every_annotation_signal(tmp_path, bdf):
    _, block = _make_block(tmp_path, bdf, annotation_signals=2)
    found = probe_timekeeping_onset(block, _refusal(block))
    assert found is not None
    assert found.first_onset == 40.0
    np.testing.assert_allclose(found.onsets, (np.arange(20) + 0.25) / 2)
    assert list(found.descriptions) == [f"event {j}" for j in range(80, 100)]


def test_probe_leaves_no_copy_behind(tmp_path, bdf, monkeypatch):
    _, block = _make_block(tmp_path, bdf)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    assert probe_timekeeping_onset(block, _refusal(block)) is not None
    assert list(scratch.iterdir()) == []


def test_probe_declines_other_refusals(tmp_path, bdf):
    _, block = _make_block(tmp_path, bdf)
    refusal = _refusal(block)
    # Only EDFlib's generic refusal can be a bad annotation list.
    assert (
        probe_timekeeping_onset(block, OSError("x: not EDF(+) or BDF(+) compliant (Filesize)"))
        is None
    )
    # A file that is not marked EDF+C/BDF+C has no time-keeping TAL to recover.
    plain = tmp_path / "plain"
    data = bytearray(Path(block).read_bytes())
    data[192:236] = b" " * 44
    plain.write_bytes(bytes(data))
    assert probe_timekeeping_onset(str(plain), refusal) is None


@pytest.mark.parametrize("make_case", _STILL_CORRUPT, ids=lambda f: f.__name__[len("_case_") :])
def test_probe_declines_a_file_that_is_still_corrupt(tmp_path, bdf, make_case):
    path = make_case(tmp_path, bdf)
    assert probe_timekeeping_onset(path, _refusal(path)) is None
    _assert_probe_declines(path)


# --- An onset no date can hold --------------------------------------------------
# The recording starts at its first record, header start plus first onset. An onset
# that float cannot hold, int cannot parse, or that carries the date past year 9999,
# is declined, and both read paths decline it the same way.


@pytest.mark.parametrize(
    ("digits", "gap_at"),
    [
        # EDFlib refuses this one only for its late first onset: the recoverable shape.
        pytest.param("3" + "0" * 11, None, id="beyond-year-9999"),
        # EDFlib reads these without complaint, so a gap is what makes it refuse the
        # file; the probe then meets a first onset that int or float cannot hold.
        pytest.param("9" + "0" * 308, 5, id="309-digits-beyond-float"),
        pytest.param("1" + "0" * 4999, 5, id="5000-digits"),
    ],
)
def test_an_onset_no_date_can_hold_is_refused_identically(tmp_path, bdf, digits, gap_at):
    path = _stamped_from(tmp_path, bdf, digits, gap_at=gap_at)
    refusal = _refusal(path)
    assert "contains format errors" in str(refusal)
    assert probe_timekeeping_onset(path, refusal) is None

    with pytest.raises(CorruptFileError) as memory:
        EDFImporter().load(path)
    with pytest.raises(CorruptFileError) as streaming:
        _open_stream_source(path, None)
    assert type(memory.value) is type(streaming.value)
    assert str(memory.value) == str(streaming.value)


def test_a_large_onset_the_calendar_can_hold_is_recovered(tmp_path, bdf):
    """The control: 10**9 s is 31 years, a date datetime holds, so the same file shape reads."""
    path = _stamped_from(tmp_path, bdf, "1" + "0" * 9)
    recovered = EDFImporter().load(path)
    assert recovered.metadata["edf_first_record_onset"] == 1e9
    assert recovered.metadata["startdate"] == _SESSION_START + timedelta(seconds=10**9)
    assert len(recovered.events) == 10

    source = _open_stream_source(path, None)
    try:
        assert source.extra_metadata["edf_first_record_onset"] == 1e9
    finally:
        source.close()


# --- Host conditions ------------------------------------------------------------
# The probe writes a small scratch copy. What the OS says about that copy is about
# this host, not about the recording, so it must leave as the OS's own OSError and
# never become a typed failure a caller would record against a valid file.

_NEEDS_PERMISSIONS = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs a POSIX user that directory permissions bind",
)


@pytest.mark.parametrize(
    "scratch",
    [
        pytest.param("missing", id="missing"),
        pytest.param("read-only", id="read-only", marks=_NEEDS_PERMISSIONS),
    ],
)
def test_unusable_scratch_directory_is_a_host_error(tmp_path, bdf, monkeypatch, scratch):
    """ENOENT (also tempfile's "no usable temporary directory") and EACCES alike."""
    _, block = _make_block(tmp_path, bdf)
    root = tmp_path / "scratch"
    if scratch == "read-only":
        root.mkdir()
        root.chmod(0o500)
        expected = errno.EACCES
    else:
        expected = errno.ENOENT
    # A real temporary root the OS refuses: nothing is replaced.
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    try:
        with pytest.raises(OSError) as memory:
            EDFImporter().load(block)
        with pytest.raises(OSError) as streaming:
            _open_stream_source(block, None)
    finally:
        if root.exists():
            root.chmod(0o700)
    for caught in (memory, streaming):
        assert not isinstance(caught.value, BiosigIOError)
        assert caught.value.errno == expected
    assert list(tmp_path.glob("**/biosigio-tal-*")) == []


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param("removed", id="removed"),
        pytest.param("unreadable", id="unreadable", marks=_NEEDS_PERMISSIONS),
    ],
)
def test_scratch_copy_that_cannot_be_read_back_is_a_host_error(tmp_path, bdf, monkeypatch, damage):
    """The copy was written, and the OS then refuses it: the host's doing, not EDFlib's verdict.

    ``pyedflib.EdfReader`` is wrapped only to damage the scratch copy the instant
    before the real reader opens it, as a failing disk or a temporary-file cleaner
    would; the real reader then runs on the real, damaged file.
    """
    _, block = _make_block(tmp_path, bdf)
    real = pyedflib.EdfReader

    def damaging(path, *args, **kwargs):
        if os.path.basename(str(path)).startswith("tal-only"):
            if damage == "removed":
                os.unlink(path)
            else:
                os.chmod(path, 0)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pyedflib, "EdfReader", damaging)
    expected = errno.ENOENT if damage == "removed" else errno.EACCES
    with pytest.raises(OSError) as memory:
        EDFImporter().load(block)
    with pytest.raises(OSError) as streaming:
        _open_stream_source(block, None)
    for caught in (memory, streaming):
        assert not isinstance(caught.value, BiosigIOError)
        assert caught.value.errno == expected


@pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(signal, "SIGXFSZ"),
    reason="needs POSIX file-size limits",
)
def test_a_failed_write_of_the_scratch_copy_is_a_host_error(tmp_path, bdf):
    """The directory is usable but the copy cannot be written (EFBIG here; ENOSPC and quota alike)."""
    import resource

    _, block = _make_block(tmp_path, bdf)
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    previous = signal.signal(
        signal.SIGXFSZ, signal.SIG_IGN
    )  # the write then fails, not the process
    # A real limit on file size, smaller than the copy: nothing is replaced.
    resource.setrlimit(resource.RLIMIT_FSIZE, (600, hard))
    try:
        with pytest.raises(OSError) as memory:
            EDFImporter().load(block)
        with pytest.raises(OSError) as streaming:
            _open_stream_source(block, None)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, previous)
    for caught in (memory, streaming):
        assert not isinstance(caught.value, BiosigIOError)
        assert caught.value.errno == errno.EFBIG


# Files the probe turns down before it needs any scratch space.
_DECLINED_BEFORE_SCRATCH: list[Callable[[Path, bool], str]] = [
    _case_small_onset_with_later_defect,
    _case_timekeeping_tal_without_onset,
    _case_timekeeping_tal_with_a_duration,
    _case_timekeeping_tal_with_text,
    _case_timekeeping_onset_with_no_sign,
]


@pytest.mark.parametrize(
    "make_case", _DECLINED_BEFORE_SCRATCH, ids=lambda f: f.__name__[len("_case_") :]
)
def test_a_declined_file_never_needs_scratch_space(tmp_path, bdf, monkeypatch, make_case):
    """With no usable temporary directory a file that is not recoverable is still corrupt, not a host error."""
    path = make_case(tmp_path, bdf)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "does-not-exist"))
    with pytest.raises(CorruptFileError):
        EDFImporter().load(path)
    with pytest.raises(CorruptFileError):
        _open_stream_source(path, None)


# --- The in-memory read ---------------------------------------------------------


def _corrected_copy(tmp_path: Path, block: str, bdf: bool, seconds: int) -> str:
    """The block with every TAL onset brought back by ``seconds``: a compliant file."""
    corrected = str(tmp_path / ("corrected.bdf" if bdf else "corrected.edf"))
    _shift_onsets(block, corrected, bdf, -seconds)
    return corrected


def _channel_samples(path: str, start: int | None = None, n: int | None = None) -> list[np.ndarray]:
    reader = pyedflib.EdfReader(path)
    try:
        if start is None:
            return [reader.readSignal(i) for i in range(reader.signals_in_file)]
        return [reader.readSignal(i, start, n) for i in range(reader.signals_in_file)]
    finally:
        reader.close()


def test_recovered_read_matches_onset_corrected_copy(tmp_path, bdf):
    """The oracle: pyedflib on the same block with its onsets brought back."""
    session, block = _make_block(tmp_path, bdf)
    corrected = _corrected_copy(tmp_path, block, bdf, _ONSET_AT_BLOCK_START)
    expected = EDFImporter().load(corrected)

    recovered = EDFImporter().load(block)

    labels = list(expected.channels)
    assert list(recovered.channels) == labels
    reference = _channel_samples(corrected)
    # The samples are the compliant file's exactly, and the session's own slice.
    sliced = _channel_samples(session, 40 * 32, 10 * 32)
    for label, ref, cut in zip(labels, reference, sliced, strict=True):
        np.testing.assert_array_equal(recovered.signals[label].to_numpy(), ref)
        np.testing.assert_array_equal(ref, cut)
    # Signal headers (units, ranges, rate, type) and every other file header field.
    assert recovered.channels == expected.channels
    skipped = {"startdate", "source_file"}
    assert {k: v for k, v in recovered.metadata.items() if k not in skipped} == {
        **{k: v for k, v in expected.metadata.items() if k not in skipped},
        "edf_tolerant_read": True,
        "edf_tolerant_read_reason": "late_timekeeping_onset",
        "edf_first_record_onset": 40.0,
    }
    # The recording starts where its first record does: the header's start plus 40 s.
    assert expected.metadata["startdate"] == _SESSION_START
    assert recovered.metadata["startdate"] == datetime(2026, 10, 16, 0, 0, 25)
    pd.testing.assert_frame_equal(recovered.events, expected.events)
    assert len(recovered.events) == 10


def test_startdate_follows_a_fractional_first_onset(tmp_path, bdf):
    start = datetime(2026, 12, 31, 23, 59, 59)
    _, block = _make_block(
        tmp_path, bdf, first=3, count=10, record_duration=0.5, rates=(64.0, 64.0), start=start
    )
    recovered = EDFImporter().load(block)
    assert recovered.metadata["edf_first_record_onset"] == 1.5
    assert recovered.metadata["startdate"] == datetime(2027, 1, 1, 0, 0, 0, 500000)


def test_recovered_events_are_relative_to_the_first_record(tmp_path, bdf):
    """A compliant, annotated file with every TAL moved by +40 reads as the original."""
    original = str(tmp_path / ("original.bdf" if bdf else "original.edf"))
    late = str(tmp_path / ("late.bdf" if bdf else "late.edf"))
    _write_session(original, bdf=bdf, n_records=12)
    _shift_onsets(original, late, bdf, 40)
    expected = EDFImporter().load(original)
    assert len(expected.events) == 12

    recovered = EDFImporter().load(late)

    pd.testing.assert_frame_equal(recovered.events, expected.events)
    for label in expected.channels:
        np.testing.assert_array_equal(
            recovered.signals[label].to_numpy(), expected.signals[label].to_numpy()
        )


def test_mixed_rate_policy_is_unchanged(tmp_path, bdf):
    _, block = _make_block(tmp_path, bdf, rates=(64.0, 16.0))
    with pytest.raises(MixedSamplingRateError):
        EDFImporter().load(block)

    recovered = EDFImporter().load(block, mixed_rate="resample")

    expected = EDFImporter().load(_corrected_copy(tmp_path, block, bdf, 40), mixed_rate="resample")
    assert recovered.metadata["mixed_rate_resampled"] is True
    assert recovered.metadata["edf_tolerant_read_reason"] == "late_timekeeping_onset"
    assert recovered.channels == expected.channels
    for label in expected.channels:
        np.testing.assert_array_equal(
            recovered.signals[label].to_numpy(), expected.signals[label].to_numpy()
        )


_NO_MNE = """
import sys
sys.modules["mne"] = None  # any import of mne now raises ImportError
from biosigio.exporters.zarr_stream import _open_stream_source
from biosigio.importers.edf import EDFImporter

path = sys.argv[1]
rec = EDFImporter().load(path)
assert rec.metadata["edf_tolerant_read_reason"] == "late_timekeeping_onset"
assert rec.metadata["edf_first_record_onset"] == 40.0

source = _open_stream_source(path, None)
assert source.extra_metadata["edf_first_record_onset"] == 40.0
assert source.read([0], 0, 8).shape == (1, 8)
source.close()
"""


def test_recovery_does_not_import_mne(tmp_path, bdf):
    """The other recoveries need the `meg` extra; this one must not, on either read path."""
    _, block = _make_block(tmp_path, bdf)
    done = subprocess.run(
        [sys.executable, "-c", _NO_MNE, block],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr


# --- The streaming read ---------------------------------------------------------

_FLAGS = {
    "edf_tolerant_read": True,
    "edf_tolerant_read_reason": "late_timekeeping_onset",
    "edf_first_record_onset": 40.0,
}


def test_edf_source_serves_bounded_windows(tmp_path, bdf):
    _, block = _make_block(tmp_path, bdf)
    expected = EDFImporter().load(block)

    source = _open_stream_source(block, None)
    try:
        # Not the preloaded MNE fallback: windows come from pyedflib's own reader.
        assert source._fallback_data is None
        assert source.extra_metadata == _FLAGS
        assert source.sfreq == 32.0
        assert source.n_samples == 320
        assert [c["label"] for c in source.channels] == list(expected.channels)
        full = np.stack([expected.signals[label].to_numpy() for label in expected.channels])
        for start, stop in [(0, 320), (0, 7), (31, 97), (250, 320)]:
            np.testing.assert_array_equal(source.read([0, 1], start, stop), full[:, start:stop])
        np.testing.assert_array_equal(source.read([1], 5, 50), full[1:, 5:50])
    finally:
        source.close()
    assert source.closed


def test_stream_mixed_rate_is_still_refused(tmp_path, bdf):
    _, block = _make_block(tmp_path, bdf, rates=(64.0, 16.0))
    with pytest.raises(MixedSamplingRateError):
        _open_stream_source(block, None)


def _stream_store(path: str, store: Path):
    from biosigio import stream_to_zarr

    stream_to_zarr(path, str(store), bids_channels="off", dtype="float32")
    return store


def _only_group(store: Path):
    import zarr

    root = zarr.open_group(str(store), mode="r")
    (name,) = root.attrs["channel_groups"]
    return root, root[name]


def test_stream_matches_in_memory_store(tmp_path, bdf):
    pytest.importorskip("zarr", reason="the Zarr store needs the 'zarr' extra")
    from biosigio import Recording

    _, block = _make_block(tmp_path, bdf)
    corrected = _corrected_copy(tmp_path, block, bdf, _ONSET_AT_BLOCK_START)

    streamed_root, streamed = _only_group(_stream_store(block, tmp_path / "streamed.zarr"))
    reference_root, reference = _only_group(_stream_store(corrected, tmp_path / "reference.zarr"))
    Recording.from_file(block).to_zarr(str(tmp_path / "in_memory.zarr"), dtype="float32")
    memory_root, in_memory = _only_group(tmp_path / "in_memory.zarr")

    # The same samples through the same pipeline: the corrected copy's store, exactly.
    np.testing.assert_array_equal(streamed["0"][:], reference["0"][:])
    # And the in-memory export (the streamed one goes through a float32 memmap).
    assert streamed["0"].shape == in_memory["0"].shape == (2, 320)
    np.testing.assert_allclose(streamed["0"][:], in_memory["0"][:], rtol=1e-6)
    streamed_meta = streamed_root.attrs["recording_metadata"]
    assert {k: streamed_meta[k] for k in _FLAGS} == _FLAGS
    assert {k: memory_root.attrs["recording_metadata"][k] for k in _FLAGS} == _FLAGS
    assert "edf_tolerant_read" not in reference_root.attrs["recording_metadata"]


# --- A real recording (opt-in) --------------------------------------------------

_LOCAL_FILE_ENV = "BIOSIGIO_EDF_TIMEKEEPING_FILE"


def test_real_recording_with_a_late_first_onset(tmp_path):
    """A real EDF+/BDF+ recording cut out of a longer session, named by the environment.

    It is not ours to redistribute, so the test runs only when the variable above
    names a local copy. It asserts behavior and nothing about which file it is:
    pyedflib refuses the file, the probe recovers it, and the recovered read equals
    pyedflib's read of the same file with its onsets brought back.
    """
    local = os.environ.get(_LOCAL_FILE_ENV)
    if not local:
        pytest.skip(f"set {_LOCAL_FILE_ENV} to a local copy of the recording to run this test")
    bdf = local.lower().endswith(".bdf")

    refusal = _refusal(local)
    assert "contains format errors" in str(refusal)
    found = probe_timekeeping_onset(local, refusal)
    assert found is not None
    assert found.first_onset >= 1
    onset = int(found.first_onset)
    if onset != found.first_onset:
        pytest.skip("a fractional first onset has no pyedflib reading to compare with")

    corrected = str(tmp_path / ("corrected.bdf" if bdf else "corrected.edf"))
    _shift_onsets(local, corrected, bdf, -onset)
    expected = EDFImporter().load(corrected)
    reference = _channel_samples(corrected)

    recovered = EDFImporter().load(local)

    assert list(recovered.channels) == list(expected.channels)
    for label, ref in zip(expected.channels, reference, strict=True):
        np.testing.assert_array_equal(recovered.signals[label].to_numpy(), ref)
    assert recovered.channels == expected.channels
    assert recovered.metadata["startdate"] == expected.metadata["startdate"] + timedelta(
        seconds=onset
    )
    assert recovered.metadata["edf_first_record_onset"] == onset
    pd.testing.assert_frame_equal(recovered.events, expected.events)

    source = _open_stream_source(local, None)
    try:
        assert source._fallback_data is None
        assert source.extra_metadata["edf_first_record_onset"] == onset
        window = 5000
        for start in range(0, source.n_samples, window):
            stop = min(start + window, source.n_samples)
            np.testing.assert_array_equal(
                source.read(list(range(len(reference))), start, stop),
                np.stack([ref[start:stop] for ref in reference]),
            )
    finally:
        source.close()
