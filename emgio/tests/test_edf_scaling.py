"""Soundness + edge-case tests for EDF/BDF physical-window scaling (issue #61).

The exporter must never silently clip the bulk signal: physical_min/physical_max
have to bracket whatever is written, fit the 8-char EDF/BDF header field, and a
genuine singularity must be handled deliberately (auto-protect) rather than
corrupting the recording. These tests pin those invariants directly on the
scaling helpers and end-to-end on real fixtures with controlled perturbations.
NO MOCKS: real signal arrays and real EDF/BDF round-trips.
"""

import pathlib

import numpy as np
import pyedflib
import pytest

from emgio import EMG
from emgio.exporters.edf import (
    _PHYS_FIELD_CHARS,
    _determine_scaling_factors,
    _fit_physical_bound,
    _resolve_physical_window,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
EX = _REPO / "examples"
EMG_FIXTURE = EX / "bids/emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf"

# A diverse set of bound values: integers, decimals, scientific-scale, both signs,
# tiny and large magnitudes; the invariants below must hold for every one.
_BOUND_VALUES = [
    0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    12.57,
    -357.512,
    2325.4321,
    -2325.4321,
    616.7234,
    0.0001234,
    -0.0002587,
    1e-6,
    -1e-6,
    1e5,
    -1e5,
    999999.9,
    -999999.9,
    0.01698,
    1.4730512e4,
    3.0,
    -360.0,
    22254.7,
    -7481.3,
    130.0,
]


@pytest.mark.parametrize("value", _BOUND_VALUES)
def test_fit_physical_bound_brackets_and_fits(value):
    """A fitted bound always rounds outward and fits the 8-char header field."""
    up = _fit_physical_bound(value, round_up=True)
    down = _fit_physical_bound(value, round_up=False)
    assert np.isfinite(up) and np.isfinite(down)
    assert len(str(up)) <= _PHYS_FIELD_CHARS, f"{up!r} exceeds {_PHYS_FIELD_CHARS} chars"
    assert len(str(down)) <= _PHYS_FIELD_CHARS, f"{down!r} exceeds {_PHYS_FIELD_CHARS} chars"
    if value != 0.0:
        assert up >= value, f"round_up({value}) -> {up} did not bracket above"
        assert down <= value, f"round_down({value}) -> {down} did not bracket below"


def test_fit_physical_bound_random_sweep():
    """Containment + field-fit over a randomized magnitude sweep (deterministic)."""
    rng = np.random.default_rng(1234)
    mags = 10.0 ** rng.uniform(-6, 6, size=2000)
    signs = rng.choice([-1.0, 1.0], size=2000)
    for v in (mags * signs).tolist():
        up = _fit_physical_bound(v, round_up=True)
        down = _fit_physical_bound(v, round_up=False)
        assert up >= v and down <= v, f"bracket failed for {v}: [{down}, {up}]"
        assert len(str(up)) <= _PHYS_FIELD_CHARS
        assert len(str(down)) <= _PHYS_FIELD_CHARS


def test_fit_physical_bound_zero():
    assert _fit_physical_bound(0.0, round_up=True) == 0.0
    assert _fit_physical_bound(0.0, round_up=False) == 0.0


@pytest.mark.parametrize("use_bdf", [False, True])
def test_determine_scaling_factors_brackets(use_bdf):
    """physical_min <= signal_min < signal_max <= physical_max for every window."""
    rng = np.random.default_rng(7)
    cases = [
        (-1.0, 1.0),
        (0.0, 12.57),
        (-357.5, 616.7),
        (-2325.0, -1243.0),
        (584.3, 12250.0),
        (-1e4, 3026.0),
        (1e-5, 2e-5),
        (-1e6, 1e6),
    ]
    cases += [tuple(sorted(rng.uniform(-5e4, 5e4, size=2))) for _ in range(200)]
    for smin, smax in cases:
        pmin, pmax, dmin, dmax, _ = _determine_scaling_factors(smin, smax, use_bdf=use_bdf)
        assert pmin <= smin, f"pmin {pmin} > smin {smin}"
        assert pmax >= smax, f"pmax {pmax} < smax {smax}"
        assert pmin < pmax, f"degenerate window [{pmin}, {pmax}] for [{smin}, {smax}]"
        assert len(str(pmin)) <= _PHYS_FIELD_CHARS
        assert len(str(pmax)) <= _PHYS_FIELD_CHARS
        if use_bdf:
            assert (dmin, dmax) == (-8388608, 8388607)
        else:
            assert (dmin, dmax) == (-32768, 32767)


def test_determine_scaling_factors_special_cases():
    """Zero, constant, NaN and tiny-range windows stay valid (pmin < pmax)."""
    pmin, pmax, *_ = _determine_scaling_factors(0.0, 0.0)
    assert (pmin, pmax) == (-1e-6, 1e-6)

    pmin, pmax, *_ = _determine_scaling_factors(1.0, 1.0)
    assert pmin <= 1.0 <= pmax and pmin < pmax

    pmin, pmax, *_ = _determine_scaling_factors(np.nan, np.nan)
    assert pmin < pmax

    # Extremely narrow range must not collapse to a zero-width window.
    pmin, pmax, *_ = _determine_scaling_factors(1.0000001, 1.0000002)
    assert pmin < pmax and pmin <= 1.0000001 and pmax >= 1.0000002


def _sine(n=4000, amp=1.0, off=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    return amp * np.sin(2 * np.pi * 10 * t) + 0.05 * amp * rng.standard_normal(n) + off


def test_resolve_window_clean_signal_is_full_range():
    x = _sine()
    lo, hi, n_clip, _ = _resolve_physical_window(x, False, "auto", 8.0, 10.0)
    assert n_clip == 0
    assert lo <= x.min() and hi >= x.max()


def test_resolve_window_auto_clips_single_spike_in_edf():
    """One huge spike on a small bulk: auto+EDF clips ONLY the spike."""
    x = _sine(amp=0.5)
    x[2000] = 500.0  # singular outlier ~1000x the bulk
    lo, hi, n_clip, exc = _resolve_physical_window(x, False, "auto", 8.0, 10.0)
    assert n_clip == 1, f"expected exactly one clipped sample, got {n_clip}"
    assert hi < 500.0 and exc > 1.0
    assert lo <= -0.4 and hi >= 0.4  # bulk fully inside the window


def test_resolve_window_bdf_absorbs_outlier_that_edf_cannot():
    """An outlier ~500x the bulk: 24-bit BDF keeps it; 16-bit EDF must clip it.

    24-bit leaves the bulk ~16 effective bits (above the 10-bit floor), so BDF
    stays lossless; 16-bit would drop the bulk to ~8 bits, so EDF clips the spike.
    """
    x = _sine(amp=1.0)
    x[2000] = 500.0
    _, _, n_clip_bdf, _ = _resolve_physical_window(x, True, "auto", 8.0, 10.0)
    _, _, n_clip_edf, _ = _resolve_physical_window(x, False, "auto", 8.0, 10.0)
    assert n_clip_bdf == 0, "BDF should absorb the outlier losslessly"
    assert n_clip_edf == 1, "16-bit EDF cannot, so the spike is clipped"


def test_resolve_window_false_never_clips():
    x = _sine(amp=0.5)
    x[2000] = 1e4
    lo, hi, n_clip, _ = _resolve_physical_window(x, False, False, 8.0, 10.0)
    assert n_clip == 0 and lo <= x.min() and hi >= x.max()


def test_resolve_window_constant_bulk_with_spikes():
    """Marker/status channel: constant level with sparse glitch spikes (CH75-like)."""
    x = np.full(6000, 199.0)
    x[1000:1006] = -7481.0  # six singular glitches
    lo, hi, n_clip, _ = _resolve_physical_window(x, False, "auto", 8.0, 10.0)
    assert n_clip == 6
    assert lo >= -10.0 and hi <= 210.0  # window snaps to the constant level, not the glitches


def test_resolve_window_true_is_noop_without_outliers():
    x = _sine()
    lo, hi, n_clip, _ = _resolve_physical_window(x, False, True, 8.0, 10.0)
    assert n_clip == 0 and lo <= x.min() and hi >= x.max()


# ----------------------------- end-to-end on files -----------------------------


def _export_reload(emg, tmp_path, **kw):
    out = tmp_path / "case.edf"
    emg.to_edf(str(out), **kw)
    written = out if out.exists() else out.with_suffix(".bdf")
    return EMG.from_file(str(written), bids_channels="off"), written


@pytest.mark.parametrize(
    "amp,off,fmt",
    [(0.5, 0.0, "edf"), (1e6, 0.0, "bdf"), (1e-5, 0.0, "bdf"), (1.0, 5000.0, "edf")],
)
def test_export_never_clips_within_header_window(tmp_path, amp, off, fmt):
    """Every reloaded sample lies inside the stored physical window (no overflow)."""
    emg = EMG()
    emg.add_channel("S", _sine(amp=amp, off=off), 1000, "uV", "EMG")
    reloaded, written = _export_reload(emg, tmp_path, format=fmt, bypass_analysis=True)
    with pyedflib.EdfReader(str(written)) as r:
        h = r.getSignalHeaders()[0]
    vals = reloaded.signals["S"].values
    eps = (h["physical_max"] - h["physical_min"]) * 1e-6
    assert vals.min() >= h["physical_min"] - eps
    assert vals.max() <= h["physical_max"] + eps


@pytest.mark.parametrize("constant", [0.0, 1.0, -3.5])
def test_export_constant_and_zero_channels(tmp_path, constant):
    emg = EMG()
    emg.add_channel("C", np.full(2000, constant), 1000, "uV", "EMG")
    reloaded, _ = _export_reload(emg, tmp_path, format="bdf", bypass_analysis=True)
    assert np.allclose(reloaded.signals["C"].values, constant, atol=1e-3)


@pytest.mark.skipif(not EMG_FIXTURE.exists(), reason="EMG fixture missing")
def test_singularity_protection_recovers_bulk(tmp_path):
    """Auto-protect on a real channel + injected spike: bulk correlation survives.

    With clip_outliers='auto' the lone spike is clipped so the bulk keeps full
    16-bit resolution (r ~ 1.0); with clip_outliers=False the spike inflates the
    range and the bulk loses resolution. Auto must do at least as well, and meet
    the >0.99 bar that clipping is meant to protect.
    """
    emg = EMG.from_file(str(EMG_FIXTURE))
    ch = list(emg.channels)[0]
    base = emg.signals[ch].values.astype(float).copy()
    spike_idx = len(base) // 2
    emg.signals.iloc[spike_idx, emg.signals.columns.get_loc(ch)] = base.max() * 500.0
    bulk = np.ones(len(base), bool)
    bulk[spike_idx] = False

    def bulk_r(clip):
        reloaded, _ = _export_reload(
            emg, tmp_path, format="edf", bypass_analysis=True, clip_outliers=clip
        )
        rel = reloaded.signals[ch].values.astype(float)[: len(base)]
        return float(np.corrcoef(base[bulk], rel[bulk])[0, 1])

    r_auto = bulk_r("auto")
    r_none = bulk_r(False)
    assert r_auto > 0.99, f"auto-protect bulk correlation too low: {r_auto}"
    assert r_auto >= r_none - 1e-6, f"auto ({r_auto}) worse than no-clip ({r_none})"


# ---- magnitude limit: EDF/BDF physical fields are 8 chars (audit finding) ----


@pytest.mark.parametrize("value", [1e7, -1e7, 1e8, -1e8, 1.2e8, -1.2e8, 1e9, -1e9, 1e10])
def test_fit_physical_bound_brackets_extreme_magnitudes(value):
    """Containment must hold even when a magnitude is too wide for the 8-char field.

    The bracket invariant is unconditional; only the 8-char fit is conditional
    (the exporter enforces representability separately).
    """
    assert _fit_physical_bound(value, round_up=True) >= value
    assert _fit_physical_bound(value, round_up=False) <= value


@pytest.mark.parametrize("fmt", ["edf", "bdf"])
def test_export_rejects_unrepresentable_magnitude(tmp_path, fmt):
    """A bulk spanning +/-2e7 needs a 9-char physical_min (sign + 8 digits).

    pyedflib would silently truncate it (e.g. -20000000 -> -2000000, a 10x error
    that re-creates the #61 corruption), so the exporter must reject it loudly
    before writing, for BOTH formats (the physical field is 8 chars in each).
    """
    emg = EMG()
    emg.add_channel("BIG", _sine(amp=2e7), 1000, "uV", "EMG")
    with pytest.raises(ValueError, match="cannot be stored in the EDF/BDF header"):
        emg.to_edf(str(tmp_path / "big.edf"), format=fmt, bypass_analysis=True, clip_outliers=False)


def test_export_rejects_huge_positive_offset(tmp_path):
    """A large positive DC offset (~1.2e8) also overflows the 8-char field."""
    emg = EMG()
    emg.add_channel("OFF", _sine(amp=1000.0) + 1.2e8, 1000, "uV", "EMG")
    with pytest.raises(ValueError, match="cannot be stored"):
        emg.to_edf(
            str(tmp_path / "off.bdf"), format="bdf", bypass_analysis=True, clip_outliers=False
        )
