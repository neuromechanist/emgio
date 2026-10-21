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

import numpy as np
import pyedflib
import pytest

from biosigio import EDFExporter, Recording
from biosigio.exporters.edf import (
    _fit_physical_bound,
    _fixed_text,
    _settle_physical_bounds,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
EX = _REPO / "examples"
WRISTBAND = EX / "wristbandEMG_truncated.set"
SESSANTAQUATTRO = EX / "one_sessantaquattro_truncated.otb+"
MOUVI = EX / "two_mouvi_truncated.otb+"
MEG_FIF = EX / "bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"

FORMATS = ["edf", "bdf"]
FILETYPES = {"edf": pyedflib.FILETYPE_EDFPLUS, "bdf": pyedflib.FILETYPE_BDFPLUS}
DIGITAL_RANGE = {"edf": (-32768, 32767), "bdf": (-8388608, 8388607)}


def _export(rec: Recording, tmp_path: pathlib.Path, fmt: str) -> pathlib.Path:
    """Write ``rec`` without clipping, so the header window is the full data range."""
    path = EDFExporter.export(
        rec,
        str(tmp_path / "out"),
        format=fmt,
        bypass_analysis=True,
        create_channels_tsv=False,
        clip_outliers=False,
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


def _assert_bounds_bracket(rec: Recording, path: pathlib.Path) -> None:
    """Every stored (physical_min, physical_max) brackets that channel's data."""
    headers = _stored_headers(path)
    labels = list(rec.channels)
    for header, label in zip(headers, labels, strict=True):
        values = rec.signals[label].to_numpy(dtype=np.float64)
        assert header["physical_min"] <= values.min(), (
            f"{label}: stored physical_min {header['physical_min']} is inside the data "
            f"window (data minimum {values.min()})"
        )
        assert header["physical_max"] >= values.max(), (
            f"{label}: stored physical_max {header['physical_max']} is inside the data "
            f"window (data maximum {values.max()})"
        )
        assert header["physical_min"] < header["physical_max"]


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
    assert header["physical_min"] <= y.min() == -53820.612
    assert header["physical_max"] >= y.max() == 92309.104


@pytest.mark.parametrize("fmt", FORMATS)
def test_stored_bounds_bracket_when_a_decimal_is_stored_short(tmp_path, fmt):
    """A maximum whose binary expansion sits just below its decimal is stored short.

    95382.648 rounds up to 95382.65, which edflib stores as ``95382.64`` (inside the
    window); the exporter has to move the bound outward until the stored text holds.
    """
    y = np.array([-1.0, 0.5, 95382.648, 3.0])
    header = _stored_headers(_export(_single_channel(y, fs=4), tmp_path, fmt))[0]
    assert header["physical_max"] >= 95382.648
    assert header["physical_min"] <= -1.0


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
        assert headers[i]["physical_min"] <= lo, f"channel {i}: min {headers[i]['physical_min']}"
        assert headers[i]["physical_max"] >= hi, f"channel {i}: max {headers[i]['physical_max']}"
        assert headers[i]["physical_min"] < headers[i]["physical_max"]


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("values", [np.zeros(8), np.full(8, 1e-12), np.full(8, -3e-15)])
def test_zero_and_residue_channels_keep_a_valid_window(tmp_path, fmt, values):
    """A constant zero, or float residue near zero, still gets bounds the header holds.

    ``-1e-06`` is stored as ``-0.00000``, a bound of zero that is inside a window
    reaching below it.
    """
    rec = _single_channel(values, unit="uV", fs=8)
    header = _stored_headers(_export(rec, tmp_path, fmt))[0]
    assert header["physical_min"] <= values.min()
    assert header["physical_max"] >= values.max()
    assert header["physical_min"] < header["physical_max"]


def test_settle_refuses_a_window_the_header_cannot_bracket():
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
        _settle_physical_bounds([channel], [(-1e-6, 1e-6)], FILETYPES["edf"])


def test_settle_widens_a_bound_that_is_stored_inside_its_window():
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
    stored = _settle_physical_bounds([channel], [window], FILETYPES["bdf"])
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
    assert headers["emg1_left"]["physical_min"] <= -53820.612
    assert headers["emg0_left"]["physical_max"] >= 95382.648


@pytest.mark.parametrize("fmt", FORMATS)
def test_sessantaquattro_stored_bounds_bracket_every_channel(tmp_path, sessantaquattro, fmt):
    _assert_bounds_bracket(sessantaquattro, _export(sessantaquattro, tmp_path, fmt))


@pytest.mark.parametrize("fmt", FORMATS)
def test_mouvi_stored_bounds_bracket_every_channel(tmp_path, mouvi, fmt):
    _assert_bounds_bracket(mouvi, _export(mouvi, tmp_path, fmt))


@pytest.mark.parametrize("fmt", FORMATS)
def test_meg_stored_bounds_bracket_every_channel(tmp_path, meg, fmt):
    """Values within 1e-8 of zero get the smallest window the header text can hold.

    ``-1e-06`` is stored as ``-0.00000``, so a fixed [-1e-6, 1e-6] window held no data
    below zero and every such channel read back with half the gain and an offset.
    """
    _assert_bounds_bracket(meg, _export(meg, tmp_path, fmt))
