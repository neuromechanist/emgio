"""EDF/BDF export keeps every sample within half an LSB of the stored scale (issue #170).

Two defects made the exported samples differ from the given ones by more than the
rounding limit of half a least significant bit (LSB):

1. The physical bounds rounded outward to fit the 8-character header field could be
   stored INSIDE the data window: edflib cuts a bound off from its binary expansion
   (95382.65 is stored as ``95382.64``, -53820.7 as ``-53820.6``), so the extreme
   samples saturated and a reader scaled by bounds the writer had not used.
2. pyedflib's ``writeSamples`` truncates a physical value toward zero when it turns
   it into a digital level, so errors reached a whole LSB.

The tests read the written files back with pyedflib and MNE, and take the LSB from the
header as stored. Synthetic arrays are only the input signal of a round trip; the
recordings in ``examples/`` are the real-data gate.
NO MOCKS: real writers, real files, real readers.
"""

import pathlib
import re
import tempfile
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pyedflib
import pytest
from pyedflib.edfwriter import WrongInputSize

from biosigio import EDFExporter, Recording
from biosigio.exporters.edf import (
    _fit_physical_bound,
    _fixed_text,
    _quantize_to_digital,
    _read_header_bounds,
    _settle_physical_bounds,
    _stored_physical_bounds,
    _unresolved_channels,
    _verify_header_bounds,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
EX = _REPO / "examples"
WRISTBAND = EX / "wristbandEMG_truncated.set"
SESSANTAQUATTRO = EX / "one_sessantaquattro_truncated.otb+"
MOUVI = EX / "two_mouvi_truncated.otb+"
MEG_FIF = EX / "bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"

FORMATS = ["edf", "bdf"]
READERS = ["pyedflib", "mne"]
EXAMPLES = ["wristband", "sessantaquattro", "mouvi", "meg"]

# The rounding limit is half an LSB. The read-back value is computed in float64 from the
# stored scale, so allow the evaluation noise (about 1e-9 LSB here) and nothing more.
HALF_LSB = 0.5
FLOAT_NOISE_LSB = 1e-6

# MNE returns SI units: it scales uV and mV channels to volts and leaves other units.
MNE_UNIT_SCALE = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}
FILETYPES = {"edf": pyedflib.FILETYPE_EDFPLUS, "bdf": pyedflib.FILETYPE_BDFPLUS}
DIGITAL_RANGE = {"edf": (-32768, 32767), "bdf": (-8388608, 8388607)}


def _export(
    rec: Recording, tmp_path: pathlib.Path, fmt: str, clip_outliers: bool | str = False
) -> pathlib.Path:
    """Write ``rec``; by default without clipping, so each header window is the data range."""
    path = EDFExporter.export(
        rec,
        str(tmp_path / "out"),
        format=fmt,
        bypass_analysis=True,
        create_channels_tsv=False,
        clip_outliers=clip_outliers,
    )
    return pathlib.Path(path)


def _stored_headers(path: pathlib.Path) -> list[dict]:
    """Signal headers as a reader parses them from the file (the stored bounds)."""
    with pyedflib.EdfReader(str(path)) as reader:
        return reader.getSignalHeaders()


def _single_channel(values, unit: str = "mV", fs: int = 1000) -> Recording:
    rec = Recording()
    rec.add_channel("c0", np.asarray(values, dtype=np.float64), fs, unit, "EMG")
    return rec


def _wristband_in_millivolts() -> Recording:
    """The wristband recording rescaled uV -> mV, as the exporter's own error advises.

    Its raw magnitudes (up to 3.3e8 uV) need more than the 8 characters of a header
    bound; in mV the same samples fit. The case behind the issue: ``emg0_left``
    reaches 95382.648 mV and ``emg1_left`` -53820.612 mV.
    """
    rec = Recording.from_file(str(WRISTBAND))
    for label in list(rec.channels):
        rec.signals[label] = rec.signals[label].astype(np.float64) / 1000.0
        rec.set_channel(label, physical_dimension="mV")
    return rec


@pytest.fixture(scope="module")
def wristband_mv() -> Recording:
    if not WRISTBAND.exists():
        pytest.skip("wristband example missing")
    return _wristband_in_millivolts()


@pytest.fixture(scope="module")
def sessantaquattro() -> Recording:
    if not SESSANTAQUATTRO.exists():
        pytest.skip("OTB sessantaquattro example missing")
    return Recording.from_file(str(SESSANTAQUATTRO), importer="otb")


@pytest.fixture(scope="module")
def mouvi() -> Recording:
    if not MOUVI.exists():
        pytest.skip("OTB mouvi example missing")
    return Recording.from_file(str(MOUVI), importer="otb")


@pytest.fixture(scope="module")
def meg() -> Recording:
    """Tesla-scale magnetometers: values near 1e-10, far below what a bound can hold."""
    if not MEG_FIF.exists():
        pytest.skip("MEG example missing")
    pytest.importorskip("mne", reason="reading a FIF file needs the optional 'meg' extra")
    return Recording.from_file(str(MEG_FIF))


# pyedflib parses a bound's header text with edflib's own parser, which can land one unit
# in the last place away from Python's float() on the same text (it reads 2.53135 as
# 2.5313499999999998), so a bound is compared with a few ulp of room.
BOUND_ULPS = 4


def _assert_brackets(header: dict, low: float, high: float, context: str = "") -> None:
    """The stored bounds of ``header`` bracket ``[low, high]``, within a few ulp."""
    min_room = BOUND_ULPS * np.spacing(abs(low))
    max_room = BOUND_ULPS * np.spacing(abs(high))
    assert header["physical_min"] <= low + min_room, (
        f"{context} stored physical_min {header['physical_min']} is inside the data window "
        f"(data minimum {low})"
    )
    assert header["physical_max"] >= high - max_room, (
        f"{context} stored physical_max {header['physical_max']} is inside the data window "
        f"(data maximum {high})"
    )
    assert header["physical_min"] < header["physical_max"], context


def _assert_bounds_bracket(rec: Recording, path: pathlib.Path) -> None:
    """Every stored (physical_min, physical_max) brackets that channel's data."""
    headers = _stored_headers(path)
    labels = list(rec.channels)
    for header, label in zip(headers, labels, strict=True):
        values = rec.signals[label].to_numpy(dtype=np.float64)
        _assert_brackets(header, float(values.min()), float(values.max()), f"{label}:")


# ----------------------- the header text of a bound (unit) -----------------------


def test_fixed_text_has_no_exponent():
    """str() gives '-1e-06'; the header text is fixed notation, nine characters long."""
    assert _fixed_text(-1e-6) == "-0.000001"
    assert _fixed_text(1e-5) == "0.00001"
    assert _fixed_text(95382.65) == "95382.65"
    assert _fixed_text(-53820.0) == "-53820"
    assert _fixed_text(0.0) == "0"


@pytest.mark.parametrize(
    "value,down,up",
    [
        (1e-6, 1e-6, 1e-6),  # '0.000001' fits as it is
        (-1e-6, -1e-5, 0),  # '-0.000001' needs nine characters, '-0.00001' fits
        (-3.5e-5, -4e-5, -3e-5),  # three digits ('-0.000035') do not fit, two do
        (-1.234e-7, -1e-5, 0),
        (-53820.612, -53820.7, -53820.6),
        (95382.648, 95382.64, 95382.65),
    ],
)
def test_fit_physical_bound_fits_fixed_notation(value, down, up):
    """A bound is rounded outward to the finest grid its header text can hold."""
    assert _fit_physical_bound(value, round_up=False) == down
    assert _fit_physical_bound(value, round_up=True) == up


def test_fit_physical_bound_fixed_text_sweep():
    """Fixed-notation fit and containment over magnitudes the exponent form hides."""
    rng = np.random.default_rng(170)
    values = 10.0 ** rng.uniform(-9, 6, size=3000) * rng.choice([-1.0, 1.0], size=3000)
    for v in values.tolist():
        up = _fit_physical_bound(v, round_up=True)
        down = _fit_physical_bound(v, round_up=False)
        assert up >= v and down <= v, f"bracket failed for {v}: [{down}, {up}]"
        assert len(_fixed_text(up)) <= 8, f"{up!r} needs {len(_fixed_text(up))} characters"
        assert len(_fixed_text(down)) <= 8, f"{down!r} needs {len(_fixed_text(down))} characters"


# ----------------------- stored bounds bracket the data window -----------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_stored_bounds_bracket_extreme_samples(tmp_path, fmt):
    """The issue's second case: str() rounding moved -53820.7 to -53820.6 in the header.

    The exporter chose -53820.7 (outside the data window, -53820.612), and edflib
    stored ``-53820.6``, inside it; 95382.65 was likewise stored as ``95382.64``.
    """
    rng = np.random.default_rng(0)
    y = rng.uniform(-50000, 90000, 20000)
    y[0], y[1] = -53820.612, 92309.104
    header = _stored_headers(_export(_single_channel(y), tmp_path, fmt))[0]
    assert (y.min(), y.max()) == (-53820.612, 92309.104)
    _assert_brackets(header, -53820.612, 92309.104)


@pytest.mark.parametrize("fmt", FORMATS)
def test_stored_bounds_bracket_when_a_decimal_is_stored_short(tmp_path, fmt):
    """A maximum whose binary expansion sits just below its decimal is stored short.

    95382.648 rounds up to 95382.65, which edflib stores as ``95382.64`` (inside the
    window); the exporter has to move the bound outward until the stored text holds.
    """
    y = np.array([-1.0, 0.5, 95382.648, 3.0])
    header = _stored_headers(_export(_single_channel(y, fs=4), tmp_path, fmt))[0]
    _assert_brackets(header, -1.0, 95382.648)


@pytest.mark.parametrize("fmt", FORMATS)
def test_stored_bounds_bracket_random_windows(tmp_path, fmt):
    """300 channels, each with its own random window, end to end through the writer."""
    rng = np.random.default_rng(1700)
    n = 300
    low = rng.choice([-1.0, 1.0], size=n) * 10.0 ** rng.uniform(-5, 6, size=n)
    width = 10.0 ** rng.uniform(-5, 6, size=n)
    rec = Recording()
    windows = []
    for i in range(n):
        lo, hi = float(low[i]), float(low[i] + width[i])
        windows.append((lo, hi))
        rec.add_channel(
            f"c{i}", np.array([lo, hi, (lo + hi) / 2, lo, hi, lo, hi, hi]), 8, "mV", "EMG"
        )
    headers = _stored_headers(_export(rec, tmp_path, fmt))
    for i, (lo, hi) in enumerate(windows):
        _assert_brackets(headers[i], lo, hi, f"channel {i}:")


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("values", [np.zeros(8), np.full(8, 1e-12), np.full(8, -3e-15)])
def test_zero_and_residue_channels_read_back_on_a_valid_window(tmp_path, fmt, values):
    """A constant zero, or float residue near zero, gets bounds the header holds.

    ``-1e-06`` is stored as ``-0.00000``, a bound of zero that is inside a window reaching
    below it, and a ``[-1e-6, 1e-6]`` window that stores as ``[0, 1e-6]`` read every such
    channel back with half the gain and an offset of 5e-7 (8 million LSB for a constant
    zero in BDF). The stored window brackets the data and the samples are within half an
    LSB, which fails for every case on the earlier exporter.
    """
    rec = _single_channel(values, unit="uV", fs=8)
    path = _export(rec, tmp_path, fmt)
    (header,) = _stored_headers(path)
    _assert_brackets(header, float(values.min()), float(values.max()))
    (back,), _ = _read_back(path, fmt, "pyedflib", 8)
    assert _error_in_lsb(values, back, header) <= HALF_LSB + FLOAT_NOISE_LSB


def test_settle_refuses_a_window_the_header_cannot_bracket(tmp_path):
    """A bound that cannot be moved to a storable text fails loudly, not silently.

    -1e-6 is stored as ``-0.00000`` however it is widened by a few units in the last
    place, so the stored minimum stays above the window.
    """
    channel = {
        "label": "c0",
        "dimension": "uV",
        "sample_frequency": 8,
        "physical_min": -1e-6,
        "physical_max": 1e-6,
        "digital_min": -32768,
        "digital_max": 32767,
        "prefilter": "",
        "transducer": "",
    }
    with pytest.raises(ValueError, match="cannot store physical bounds"):
        _settle_physical_bounds([channel], [(-1e-6, 1e-6)], FILETYPES["edf"], str(tmp_path))


def test_settle_widens_a_bound_that_is_stored_inside_its_window(tmp_path):
    """The unit-level view of the same fix: the stored bounds bracket after settling."""
    channel = {
        "label": "c0",
        "dimension": "mV",
        "sample_frequency": 8,
        "physical_min": -53820.7,
        "physical_max": 95382.65,
        "digital_min": -8388608,
        "digital_max": 8388607,
        "prefilter": "",
        "transducer": "",
    }
    window = (-53820.612, 95382.648)
    stored = _settle_physical_bounds([channel], [window], FILETYPES["bdf"], str(tmp_path))
    assert stored[0][0] <= window[0]
    assert stored[0][1] >= window[1]
    # The bounds handed to the writer moved outward by a few units in the last place only.
    assert abs(channel["physical_min"] - -53820.7) < 1e-8
    assert abs(channel["physical_max"] - 95382.65) < 1e-8


# ----------------------- real recordings (the gate) -----------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_wristband_stored_bounds_bracket_every_channel(tmp_path, wristband_mv, fmt):
    """The wristband recording, where the issue found the inside-the-window bounds."""
    path = _export(wristband_mv, tmp_path, fmt)
    _assert_bounds_bracket(wristband_mv, path)
    headers = {h["label"]: h for h in _stored_headers(path)}
    # The pair from the issue: emg1_left reaches -53820.612 mV, emg0_left 95382.648 mV.
    for label, extreme in (("emg1_left", -53820.612), ("emg0_left", 95382.648)):
        values = wristband_mv.signals[label].to_numpy(dtype=np.float64)
        assert extreme in (values.min(), values.max())
        _assert_brackets(headers[label], float(values.min()), float(values.max()), f"{label}:")


@pytest.mark.parametrize("fmt", FORMATS)
def test_sessantaquattro_stored_bounds_bracket_every_channel(tmp_path, sessantaquattro, fmt):
    """Guard: the earlier exporter brackets this recording too; its defect here is the
    rounding and the constant zero channels, which the half-LSB gate below fails on."""
    _assert_bounds_bracket(sessantaquattro, _export(sessantaquattro, tmp_path, fmt))


@pytest.mark.parametrize("fmt", FORMATS)
def test_mouvi_stored_bounds_bracket_every_channel(tmp_path, mouvi, fmt):
    """Guard: the earlier exporter brackets this recording too; see the half-LSB gate."""
    _assert_bounds_bracket(mouvi, _export(mouvi, tmp_path, fmt))


@pytest.mark.parametrize("fmt", FORMATS)
def test_meg_stored_bounds_bracket_every_channel(tmp_path, meg, fmt):
    """Values within 1e-8 of zero get the smallest window the header text can hold.

    ``-1e-06`` is stored as ``-0.00000``, so a fixed [-1e-6, 1e-6] window held no data
    below zero and every such channel read back with half the gain and an offset.
    """
    _assert_bounds_bracket(meg, _export(meg, tmp_path, fmt))


# ----------------------- every sample within half an LSB -----------------------


def _read_back(path: pathlib.Path, fmt: str, reader: str, n_samples: int):
    """Read ``path`` with ``reader``; return (signals in header units, headers).

    ``signals`` is one array per channel with the first ``n_samples`` samples (an EDF
    holds whole data records, so a file may be longer than the recording). ``headers``
    are the signal headers as stored, whatever ``reader`` is.
    """
    headers = _stored_headers(path)
    if reader == "pyedflib":
        with pyedflib.EdfReader(str(path)) as edf:
            signals = [edf.readSignal(i)[:n_samples] for i in range(len(headers))]
        return signals, headers

    mne = pytest.importorskip("mne", reason="the MNE read-back needs the optional 'meg' extra")
    read_raw = mne.io.read_raw_bdf if fmt == "bdf" else mne.io.read_raw_edf
    raw = read_raw(str(path), preload=True, verbose="error")
    assert raw.ch_names == [h["label"] for h in headers]
    scale = np.array([MNE_UNIT_SCALE.get(h["dimension"], 1.0) for h in headers])
    return list(raw.get_data()[:, :n_samples] / scale[:, None]), headers


def _error_in_lsb(source: np.ndarray, back: np.ndarray, header: dict) -> float:
    """Largest read-back error of one channel, in LSB of the scale the header stores."""
    lsb = (header["physical_max"] - header["physical_min"]) / (
        header["digital_max"] - header["digital_min"]
    )
    assert len(back) == len(source)
    return float(np.max(np.abs(back - source)) / lsb)


def _assert_within_half_lsb(rec: Recording, path: pathlib.Path, fmt: str, reader: str) -> None:
    labels = list(rec.channels)
    n = len(rec.signals)
    signals, headers = _read_back(path, fmt, reader, n)
    assert len(signals) == len(labels)
    worst = {
        label: _error_in_lsb(rec.signals[label].to_numpy(dtype=np.float64), back, header)
        for label, back, header in zip(labels, signals, headers, strict=True)
    }
    label = max(worst, key=lambda k: worst[k])
    assert worst[label] <= HALF_LSB + FLOAT_NOISE_LSB, (
        f"{reader} read of the {fmt} file: {label} is {worst[label]:.9f} LSB from the source "
        f"(limit {HALF_LSB} LSB)"
    )


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_uniform_noise_reads_back_within_half_lsb(tmp_path, fmt, reader):
    """The issue's first case: truncation toward zero left up to one whole LSB."""
    x = np.random.default_rng(0).uniform(-1, 1, 20000)
    rec = _single_channel(x, unit="uV")
    _assert_within_half_lsb(rec, _export(rec, tmp_path, fmt), fmt, reader)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_extreme_samples_read_back_within_half_lsb(tmp_path, fmt, reader):
    """The issue's second case: stored bounds inside the window saturated the extremes."""
    rng = np.random.default_rng(0)
    y = rng.uniform(-50000, 90000, 20000)
    y[0], y[1] = -53820.612, 92309.104
    rec = _single_channel(y)
    _assert_within_half_lsb(rec, _export(rec, tmp_path, fmt), fmt, reader)


@pytest.mark.parametrize("fmt", FORMATS)
def test_samples_outside_the_window_saturate_at_the_stored_bounds(tmp_path, fmt):
    """With clipping on, a singular spike reads back at the stored bound, the bulk exactly.

    The bulk keeps its half-LSB bound against the (narrow) window that ``clip_outliers``
    chose; only the spike is moved, to the stored maximum.
    """
    t = np.linspace(0, 1, 4000)
    x = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.01 * np.random.default_rng(3).standard_normal(4000)
    x[2000] = 500.0
    rec = _single_channel(x, unit="uV")
    with pytest.warns(UserWarning, match="will saturate"):
        path = pathlib.Path(
            EDFExporter.export(
                rec,
                str(tmp_path / "out"),
                format=fmt,
                bypass_analysis=True,
                create_channels_tsv=False,
                clip_outliers=True,
            )
        )
    (back,), (header,) = _read_back(path, fmt, "pyedflib", len(x))
    bulk = np.arange(len(x)) != 2000
    assert _error_in_lsb(x[bulk], back[bulk], header) <= HALF_LSB + FLOAT_NOISE_LSB
    lsb = (header["physical_max"] - header["physical_min"]) / (
        header["digital_max"] - header["digital_min"]
    )
    assert abs(back[2000] - header["physical_max"]) <= HALF_LSB * lsb + 1e-9


@pytest.mark.parametrize("fmt", FORMATS)
def test_nan_samples_are_written_as_zero(tmp_path, fmt):
    """NaN has no level; the exporter has always written zero for it, and still does."""
    x = np.random.default_rng(5).uniform(-2, 2, 2000)
    x[[10, 500]] = np.nan
    rec = _single_channel(x, unit="uV")
    path = _export(rec, tmp_path, fmt)
    (back,), (header,) = _read_back(path, fmt, "pyedflib", len(x))
    lsb = (header["physical_max"] - header["physical_min"]) / (
        header["digital_max"] - header["digital_min"]
    )
    assert np.all(np.abs(back[[10, 500]]) <= HALF_LSB * lsb + 1e-9)
    kept = np.isfinite(x)
    assert _error_in_lsb(x[kept], back[kept], header) <= HALF_LSB + FLOAT_NOISE_LSB


def test_quantize_rounds_to_nearest_level_and_saturates():
    """A pure check of the rounding rule, on the arithmetic a reader inverts."""
    # Levels 0..10 over [0, 10]: one unit per level.
    values = np.array([0.0, 0.49, 0.51, 4.5, 9.99, 10.0, -3.0, 12.0])
    levels = _quantize_to_digital(values, 0.0, 10.0, 0, 10)
    assert levels.dtype == np.int32
    assert levels.tolist() == [0, 0, 1, 4, 10, 10, 0, 10]  # 4.5 ties to the even level

    # The same rule against a bipolar digital range.
    levels = _quantize_to_digital(np.array([-5.0, -0.4, 0.4, 5.0]), -5.0, 5.0, -100, 100)
    assert levels.tolist() == [-100, -8, 8, 100]


@pytest.fixture(scope="module", params=[(e, f) for e in EXAMPLES for f in FORMATS], ids="-".join)
def exported_example(request, tmp_path_factory):
    """A real example written once per format; the readers below share the file."""
    name, fmt = request.param
    fixture = {
        "wristband": "wristband_mv",
        "sessantaquattro": "sessantaquattro",
        "mouvi": "mouvi",
        "meg": "meg",
    }
    rec = request.getfixturevalue(fixture[name])
    return rec, _export(rec, tmp_path_factory.mktemp(f"{name}_{fmt}"), fmt), fmt


@pytest.mark.parametrize("reader", READERS)
def test_real_examples_read_back_within_half_lsb(exported_example, reader):
    """The gate: wristband (in mV), OTB sessantaquattro, OTB mouvi and the MEG recording.

    Every amplitude of every channel, including the constant zero channels of the OTB
    recordings and the Tesla-scale MEG channels, is within half an LSB of the source,
    with the LSB taken from the header as stored, as pyedflib and as MNE read it.
    """
    rec, path, fmt = exported_example
    _assert_within_half_lsb(rec, path, fmt, reader)


# ----------------------- each channel keeps its own scale -----------------------

# EEG in uV (about +/-80 and +/-40), EMG in mV (+/-3) and EMG in uV (+/-30000): the widest
# channel must not set the scale of the narrow ones in the same file.
MIXED_CHANNELS = [
    ("eeg_a", 80.0, "uV", "EEG"),
    ("eeg_b", 40.0, "uV", "EEG"),
    ("emg_mv", 3.0, "mV", "EMG"),
    ("emg_uv", 30000.0, "uV", "EMG"),
]


def _mixed_eeg_emg() -> Recording:
    rng = np.random.default_rng(2026)
    rec = Recording()
    for label, amplitude, unit, kind in MIXED_CHANNELS:
        rec.add_channel(label, rng.uniform(-amplitude, amplitude, 5000), 1000, unit, kind)
    return rec


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_mixed_eeg_emg_channels_keep_their_own_scale(tmp_path, fmt, reader):
    """Each channel is within half of ITS OWN header LSB, and its window fits ITS OWN data.

    The default ``clip_outliers="auto"`` is used, as a caller who does not choose gets it.
    """
    rec = _mixed_eeg_emg()
    path = _export(rec, tmp_path, fmt, clip_outliers="auto")
    signals, headers = _read_back(path, fmt, reader, len(rec.signals))
    lsbs = []
    for (label, *_), back, header in zip(MIXED_CHANNELS, signals, headers, strict=True):
        source = rec.signals[label].to_numpy(dtype=np.float64)
        lsb = (header["physical_max"] - header["physical_min"]) / (
            header["digital_max"] - header["digital_min"]
        )
        lsbs.append(lsb)
        assert _error_in_lsb(source, back, header) <= HALF_LSB + FLOAT_NOISE_LSB, (
            f"{label}: more than half of its own header LSB ({lsb:g} {header['dimension']})"
        )
        # The window hugs this channel's data (outward rounding of the 8-character bounds
        # only), so no wider channel has set its scale.
        data_range = float(source.max() - source.min())
        window = header["physical_max"] - header["physical_min"]
        assert data_range <= window <= data_range * (1 + 1e-3), (
            f"{label}: window {window:g} for data range {data_range:g} {header['dimension']}"
        )
    # Four channels, four scales; an EEG LSB is far below the wide EMG channel's.
    assert len(set(lsbs)) == len(MIXED_CHANNELS)
    assert lsbs[0] < lsbs[3] / 100


# ----------------------- padding of a partial last record -----------------------


def _header_lsb(header: dict) -> float:
    return (header["physical_max"] - header["physical_min"]) / (
        header["digital_max"] - header["digital_min"]
    )


def _padding_error_in_lsb(back: np.ndarray, n_samples: int, header: dict) -> float:
    """Distance of the padded samples from physical 0 (or the nearest bound), in LSB."""
    nearest_zero = min(max(0.0, header["physical_min"]), header["physical_max"])
    return float(np.max(np.abs(back[n_samples:] - nearest_zero)) / _header_lsb(header))


@pytest.mark.parametrize("fmt", FORMATS)
def test_padding_of_a_partial_last_record_reads_back_as_zero(tmp_path, fmt):
    """1500 samples at 1000 Hz end inside the second record; the 500 padded samples are 0.

    pyedflib pads digital samples with digital 0, the middle of the window: 20000 for
    [-50000, 90000], 7.5 for [5, 10] and 3 for [-3, 9]. Physical 0 is the padding, and
    for [5, 10], which does not contain 0, the nearest value the window holds, its
    minimum.
    """
    windows = [(-50000.0, 90000.0), (5.0, 10.0), (-3.0, 9.0)]
    rng = np.random.default_rng(171)
    rec = Recording()
    for i, (lo, hi) in enumerate(windows):
        values = rng.uniform(lo, hi, 1500)
        values[0], values[1] = lo, hi
        rec.add_channel(f"c{i}", values, 1000, "mV", "EMG")
    path = _export(rec, tmp_path, fmt)
    signals, headers = _read_back(path, fmt, "pyedflib", 2000)
    for i, header in enumerate(headers):
        assert len(signals[i]) == 2000, "the last record is padded to whole records"
        assert _padding_error_in_lsb(signals[i], 1500, header) <= HALF_LSB + FLOAT_NOISE_LSB, (
            f"channel c{i}: padded samples are not physical 0"
        )
        source = rec.signals[f"c{i}"].to_numpy(dtype=np.float64)
        assert _error_in_lsb(source, signals[i][:1500], header) <= HALF_LSB + FLOAT_NOISE_LSB


@pytest.mark.parametrize("fmt", FORMATS)
def test_wristband_padding_reads_back_as_zero(tmp_path, wristband_mv, fmt):
    """The wristband recording has 20001 samples at 2000 Hz: 1999 are padded."""
    path = _export(wristband_mv, tmp_path, fmt)
    n = len(wristband_mv.signals)
    signals, headers = _read_back(path, fmt, "pyedflib", 22000)
    assert n == 20001 and len(signals[0]) == 22000
    for label, back, header in zip(wristband_mv.channels, signals, headers, strict=True):
        assert _padding_error_in_lsb(back, n, header) <= HALF_LSB + FLOAT_NOISE_LSB, label


# ----------------------- channels too narrow for their stored scale -----------------------

UNRESOLVED = re.compile(r"(\d+) channel\(s\) span fewer than 4 stored LSB")


def _unresolved_messages(caught) -> list[str]:
    return [str(w.message) for w in caught if UNRESOLVED.search(str(w.message))]


ENTRY = re.compile(
    r"'(?P<name>[^']+)' spans (?P<span>\S+) (?P<unit>\S+) = (?P<ratio>\S+) LSB "
    r"\(LSB (?P<lsb>\S+) (?P<unit_again>\S+)\)"
)


def _assert_entry_numbers(message: str, name: str, source: np.ndarray, header: dict) -> None:
    """The warning's span, LSB count and LSB for ``name`` are what the file and data say."""
    entries = {m["name"]: m for m in ENTRY.finditer(message)}
    entry = entries[name]
    span = float(source.max() - source.min())
    lsb = _header_lsb(header)
    assert entry["span"] == f"{span:.3g}", entry.group(0)
    assert entry["ratio"] == f"{span / lsb:.2g}", entry.group(0)
    assert entry["lsb"] == f"{lsb:.3g}", entry.group(0)
    assert entry["unit"] == entry["unit_again"] == header["dimension"], entry.group(0)


def _channels_under_four_lsb(rec: Recording, path: pathlib.Path) -> list[str]:
    """Channels whose data span is above 0 and below 4 LSB of the scale the file stores."""
    with pyedflib.EdfReader(str(path)) as edf:
        headers = edf.getSignalHeaders()
    out = []
    for label, header in zip(rec.channels, headers, strict=True):
        values = rec.signals[label].to_numpy(dtype=np.float64)
        span = float(values.max() - values.min())
        if 0 < span < 4 * _header_lsb(header):
            out.append(label)
    return out


def test_meg_in_edf_warns_about_channels_below_four_lsb(tmp_path, meg):
    """Tesla magnetometers span under one 16-bit LSB of the smallest range the header holds.

    The export still succeeds, and the warning names the channels with their span and
    their LSB; its count is the number of channels the file leaves under 4 LSB.
    """
    with pytest.warns(UserWarning, match=UNRESOLVED.pattern) as caught:
        path = _export(meg, tmp_path, "edf")
    assert path.exists()
    (message,) = _unresolved_messages(caught)
    assert " spans " in message and " LSB (LSB " in message and " T = " in message
    under = _channels_under_four_lsb(meg, path)
    # The counts the changelog and the format page state for the MEG example.
    assert Counter(i["channel_type"] for i in meg.channels.values()) == {
        "MEGMAG": 274,
        "MEGREFMAG": 29,
        "TRIG": 2,
    }
    assert Counter(meg.channels[name]["channel_type"] for name in under) == {
        "MEGMAG": 274,
        "MEGREFMAG": 20,
    }
    headers = {h["label"]: h for h in _stored_headers(path)}
    # The five channels the warning names, with their span, LSB count and LSB.
    for name in under[:5]:
        _assert_entry_numbers(message, name, meg.signals[name].to_numpy(np.float64), headers[name])
    assert len(ENTRY.findall(message)) == 5
    count = int(UNRESOLVED.search(message).group(1))
    assert count == len(under)
    assert count > 5 and f"and {count - 5} more" in message
    # Centred on zero these channels would be no better off: the unit is the problem.
    assert message.count("rescale it to a smaller unit than T") == 5
    assert "subtract the offset" not in message


def test_offset_channel_is_told_to_subtract_the_offset(tmp_path):
    """410986 mV, give or take 5e-6, in 16 bits: the offset, not the unit, is the problem.

    The bounds around 410986 hold one decimal ([410985.9, 410986.1]), so the 1e-5 mV
    variation covers 3.3 LSB. The same variation centred on zero is resolved, so the
    warning says to subtract the offset, and doing so removes the warning.
    """
    rng = np.random.default_rng(410986)
    values = 410986.0 + rng.uniform(-5e-6, 5e-6, 2000)
    values[0], values[1] = 410986.0 - 5e-6, 410986.0 + 5e-6
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = _export(_single_channel(values), tmp_path, "edf")
    (message,) = _unresolved_messages(caught)
    (header,) = _stored_headers(first)
    _assert_entry_numbers(message, "c0", values, header)
    assert "its variation is small against its offset (about 410986 mV)" in message
    assert "subtract the offset before export" in message
    assert "rescale" not in message

    subtracted = tmp_path / "subtracted"
    subtracted.mkdir()
    rec = _single_channel(values - 410986.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = _export(rec, subtracted, "edf")
    assert _unresolved_messages(caught) == []
    (back,), (header,) = _read_back(path, "edf", "pyedflib", 2000)
    assert _error_in_lsb(values - 410986.0, back, header) <= HALF_LSB + FLOAT_NOISE_LSB


def test_warning_names_the_channel_not_the_edf_label(tmp_path):
    """Two 24-character names are cut to 16 characters and numbered in the file.

    The EDF labels are ``Mini sensor 10-0`` and ``Mini sensor 10-1``; the warning about
    their narrow span names the recording's channels, and keeps the labels out.
    """
    rng = np.random.default_rng(10)
    rec = Recording()
    names = ["Mini sensor 10: ACC.X 10", "Mini sensor 10: ACC.Y 10"]
    for name in names:
        rec.add_channel(name, rng.uniform(-1e-11, 1e-11, 2000), 1000, "T", "OTHER")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = _export(rec, tmp_path, "edf")
    assert {h["label"] for h in _stored_headers(path)} == {"Mini sensor 10-0", "Mini sensor 10-1"}
    (message,) = _unresolved_messages(caught)
    assert UNRESOLVED.search(message).group(1) == "2"
    assert all(repr(name) in message for name in names)
    assert "Mini sensor 10-" not in message


def test_float_residue_channel_is_reported_and_loses_its_variation(tmp_path):
    """Values of 0 and 1e-12 in 16 bits: it varies, the file cannot hold it, so it is reported.

    The rule is the span, with no relative floor: a non-constant channel under 4 LSB
    warns. Both values land on the level of the stored bound at 0 and read back equal;
    in 24 bits the same channel spans 17 LSB, keeps two levels, and is not reported.
    """
    values = np.tile([0.0, 1e-12], 1000)
    rec = _single_channel(values, unit="uV")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = _export(rec, tmp_path, "edf")
    (message,) = _unresolved_messages(caught)
    (header,) = _stored_headers(path)
    _assert_entry_numbers(message, "c0", values, header)
    assert "rescale it to a smaller unit than uV" in message
    (back,), _ = _read_back(path, "edf", "pyedflib", 2000)
    assert len(np.unique(back)) == 1, "the variation is gone from the file"

    bdf = tmp_path / "bdf"
    bdf.mkdir()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bdf_path = _export(rec, bdf, "bdf")
    assert _unresolved_messages(caught) == []
    (back,), (bdf_header,) = _read_back(bdf_path, "bdf", "pyedflib", 2000)
    assert len(np.unique(back)) == 2
    assert _error_in_lsb(values, back, bdf_header) <= HALF_LSB + FLOAT_NOISE_LSB


@pytest.mark.parametrize("value", [1e-12, -3e-15])
def test_constant_below_half_an_lsb_is_not_reported_and_reads_back_as_zero(tmp_path, value):
    """A constant under half an LSB reads back as the stored bound at 0 (#172 tracks this).

    The window of ``[0, 1e-6]`` (``[-1e-5, 0]`` below zero) puts the constant on the
    level of the bound at 0, so it reads back as exactly that bound: within half an LSB
    of the source, but not the source. It has no variation, so nothing is reported.
    """
    values = np.full(2000, value)
    rec = _single_channel(values, unit="uV")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = _export(rec, tmp_path, "edf")
    assert _unresolved_messages(caught) == []
    (header,) = _stored_headers(path)
    assert 0.0 in (header["physical_min"], header["physical_max"])
    assert abs(value) < HALF_LSB * _header_lsb(header)
    (back,), _ = _read_back(path, "edf", "pyedflib", 2000)
    assert np.all(np.abs(back) <= FLOAT_NOISE_LSB * _header_lsb(header))
    assert _error_in_lsb(values, back, header) <= HALF_LSB + FLOAT_NOISE_LSB


def test_meg_in_bdf_resolves_its_channels_without_a_warning(tmp_path, meg):
    """Guard against a false alarm: 24 bits leave every MEG channel at least 4 LSB wide."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = _export(meg, tmp_path, "bdf")
    assert _unresolved_messages(caught) == []
    assert _channels_under_four_lsb(meg, path) == []


@pytest.mark.parametrize("fmt", FORMATS)
def test_constant_zero_channels_are_not_reported(tmp_path, sessantaquattro, mouvi, fmt):
    """Guard: a constant channel has no variation to lose; the OTB recordings have 6 and 38."""
    for rec in (sessantaquattro, mouvi):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = _export(rec, tmp_path, fmt)
        assert _unresolved_messages(caught) == []
        assert _channels_under_four_lsb(rec, path) == []


def test_unresolved_threshold_is_four_lsb():
    """Below 4 LSB is reported, 4 LSB and more is not, and a constant is never."""
    lsb = 2.0 / (32767 + 32768)
    channels = [
        {"label": name, "dimension": "uV", "digital_min": -32768, "digital_max": 32767}
        for name in ("below", "above", "constant")
    ]
    windows = [(0.0, 3.9 * lsb), (0.0, 4.1 * lsb), (1.0, 1.0)]
    stored = [(-1.0, 1.0)] * 3
    names = ["below", "above", "constant"]
    assert [n for n, *_ in _unresolved_channels(names, channels, windows, stored)] == ["below"]


def test_unresolved_threshold_is_strict_at_exactly_four_lsb():
    """A span of exactly 4.0 LSB is resolved; one ulp under it is not."""
    lsb = 2.0 / (32767 + 32768)
    channel = {"label": "c", "dimension": "uV", "digital_min": -32768, "digital_max": 32767}
    stored = [(-1.0, 1.0)]
    exactly = _unresolved_channels(["c"], [channel], [(0.0, 4 * lsb)], stored)
    just_under = _unresolved_channels(["c"], [channel], [(0.0, np.nextafter(4 * lsb, 0.0))], stored)
    assert exactly == []
    assert [n for n, *_ in just_under] == ["c"]


# ----------------------- the scratch header lives beside the output -----------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_export_does_not_need_the_system_temp_directory(tmp_path, monkeypatch, fmt):
    """Regression test: the scratch header is written beside the output, which the export can write to.

    It fails on 3d87ce0, which wrote the scratch header to the system temporary directory,
    and passes on main, which writes none. The system temporary directory is pointed at a
    directory that does not exist, as an unwritable TMPDIR would; the export succeeds and
    leaves only the file it was asked for.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "no_such_dir"))
    out = tmp_path / "out_dir"
    out.mkdir()
    rec = _single_channel(np.random.default_rng(4).uniform(-1, 1, 2000), unit="uV")
    path = _export(rec, out, fmt)
    assert [p.name for p in out.iterdir()] == [path.name]


def test_stored_bounds_that_raise_leave_no_scratch_file(tmp_path):
    """pyedflib refuses a digital minimum below the 16-bit range while the header is set.

    That happens after the scratch file is created, inside ``_stored_physical_bounds``
    itself; the file is gone when the error reaches the caller.
    """
    channel = {
        "label": "c0",
        "dimension": "uV",
        "sample_frequency": 8,
        "physical_min": -1.0,
        "physical_max": 1.0,
        "digital_min": -40000,
        "digital_max": 32767,
        "prefilter": "",
        "transducer": "",
    }
    with pytest.raises(ValueError, match="Digital minimum"):
        _stored_physical_bounds([channel], FILETYPES["edf"], str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_settle_leaves_no_scratch_file_on_success_or_failure(tmp_path):
    """Every scratch header is removed, whether the bounds settle or the call raises."""
    channel = {
        "label": "c0",
        "dimension": "mV",
        "sample_frequency": 8,
        "physical_min": -53820.7,
        "physical_max": 95382.65,
        "digital_min": -32768,
        "digital_max": 32767,
        "prefilter": "",
        "transducer": "",
    }
    _settle_physical_bounds([channel], [(-53820.612, 95382.648)], FILETYPES["edf"], str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    channel.update(physical_min=-1e-6, physical_max=1e-6)
    with pytest.raises(ValueError, match="cannot store physical bounds"):
        _settle_physical_bounds([channel], [(-1e-6, 1e-6)], FILETYPES["edf"], str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fmt", FORMATS)
def test_refused_export_leaves_no_file_behind(tmp_path, fmt):
    """Guard: a magnitude the header cannot hold is refused before any header is written.

    The refusal comes before the scratch header exists, so only the output file is at
    stake here; the scratch file's removal is covered by the settle and
    ``_stored_physical_bounds`` tests above and below.
    """
    rec = _single_channel(np.linspace(-2e7, 2e7, 2000), unit="uV")
    with pytest.raises(ValueError, match="cannot be stored in the EDF/BDF header"):
        _export(rec, tmp_path, fmt)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fmt", FORMATS)
def test_bounds_bracket_when_pyedflib_parses_a_bound_one_ulp_low(tmp_path, fmt):
    """Guard for the ulp tolerance: 2.5312 and 2.53135 are written as that text, and pyedflib
    reads the second low.

    Python's float() reads ``2.53135`` as the data maximum, pyedflib as 2.5313499999999998,
    one ulp below it. The stored bounds still bracket the data, within that ulp, and the
    samples are within half an LSB.
    """
    x = np.array([2.531246, 2.53135, 2.5312])
    rec = _single_channel(x, unit="uV", fs=3)
    path = _export(rec, tmp_path, fmt)
    (header,) = _stored_headers(path)
    _assert_brackets(header, float(x.min()), float(x.max()))
    (back,), _ = _read_back(path, fmt, "pyedflib", 3)
    assert _error_in_lsb(x, back, header) <= HALF_LSB + FLOAT_NOISE_LSB


# ----------------------- branches of the export that a normal file does not reach -----------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_header_verification_accepts_the_stored_bounds_and_refuses_others(tmp_path, fmt):
    """The check after the write passes on the file's own bounds and raises on any other."""
    rec = _single_channel(np.random.default_rng(8).uniform(-2, 2, 2000), unit="uV")
    path = str(_export(rec, tmp_path, fmt))
    stored = _read_header_bounds(path, 1)
    _verify_header_bounds(path, stored)
    low, high = stored[0]
    with pytest.raises(ValueError, match="other than the ones its samples were scaled by"):
        _verify_header_bounds(path, [(low - 1.0, high)])
    with pytest.raises(ValueError, match="other than the ones its samples were scaled by"):
        _verify_header_bounds(path, [(low, high + 1.0)])


def test_export_without_channels_raises_and_leaves_nothing(tmp_path):
    """No channel means no header to settle; pyedflib refuses the empty write, as before."""
    rec = Recording()
    rec.signals = pd.DataFrame()
    with pytest.raises(WrongInputSize, match="Data list is empty"):
        EDFExporter.export(
            rec,
            str(tmp_path / "out"),
            format="bdf",
            bypass_analysis=True,
            create_channels_tsv=False,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fmt", FORMATS)
def test_nan_in_a_positive_only_channel_reads_back_as_physical_min(tmp_path, fmt):
    """Documents the current behaviour, tracked in #173: NaN is written as 0.

    The window of a channel in [5, 10] is taken from its finite samples, so it does not
    contain 0, and the 0 saturates at the bottom of the digital range: a NaN reads back
    as the stored physical_min (5), not as a missing value and not as 0. When #173
    changes how NaN is written, this test changes with it.
    """
    rng = np.random.default_rng(173)
    x = rng.uniform(5.0, 10.0, 1000)
    x[0], x[1] = 5.0, 10.0
    nan_at = [100, 200]
    x[nan_at] = np.nan
    rec = _single_channel(x, unit="mV")
    path = _export(rec, tmp_path, fmt)
    (back,), (header,) = _read_back(path, fmt, "pyedflib", 1000)
    assert abs(header["physical_min"] - 5.0) <= 1e-6
    assert np.all(np.abs(back[nan_at] - header["physical_min"]) <= HALF_LSB * _header_lsb(header))
    kept = np.isfinite(x)
    assert _error_in_lsb(x[kept], back[kept], header) <= HALF_LSB + FLOAT_NOISE_LSB
