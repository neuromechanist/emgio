"""Real-data round-trip harness (issue #48).

Imports every real fixture, exports it to EDF/BDF, reimports, and checks signal
integrity. This is the grounding backbone the mock-based ``to_edf`` tests lacked
(a mock exporter never wrote real bytes, which is how the 1-second truncation bug
shipped). NO MOCKS.

Fidelity is judged on a random 10-second window with per-channel Pearson
correlation > 0.99, the exemplar metric a conversion must meet; near-constant
channels (markers/flat references) have undefined correlation and are checked by
NRMSE instead. This is stricter than amplitude-relative NRMSE alone, which can
stay small even when quantization has destroyed a channel's waveform (the #61
class of bug).

Known limitations are asserted explicitly rather than skipped:
- mixed per-channel sample rates (XDF, Trigno) must fail loudly on export;
- MEG ``.fif`` imports via the MNE-backed importer (#53; needs the 'meg' extra).
"""

import atexit
import os
import pathlib
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest

from biosigio import Recording

# Round-trips are expensive; compute each fixture's once and share it across the
# structural and value-integrity tests.
_RT_DIR = tempfile.mkdtemp(prefix="emgio_roundtrip_")
_RT_CACHE: dict = {}
atexit.register(shutil.rmtree, _RT_DIR, ignore_errors=True)

_REPO = pathlib.Path(__file__).resolve().parents[2]
EX = _REPO / "examples"
BIDS = EX / "bids"

_RNG = np.random.default_rng(0)
_WINDOW_S = 10.0  # exemplar fidelity window
_CONST_PTP = 1e-9  # below this peak-to-peak a channel has no defined correlation


@dataclass(frozen=True)
class Case:
    name: str
    path: pathlib.Path
    importer: str | None
    has_emg: bool  # whether EMG channels are legitimately present at import
    tolerance: float = 0.05  # NRMSE tolerance for near-constant channels


# Single-rate fixtures spanning EMG / iEEG / EEG / OTB. Structural integrity and
# per-channel waveform fidelity (r > 0.99) must hold for all after the #61 fix.
ROUNDTRIP_CASES = [
    Case(
        "emg",
        BIDS / "emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf",
        None,
        has_emg=True,
    ),
    Case(
        "ieeg",
        BIDS / "ieeg/sub-01/ses-postimp/ieeg/sub-01_ses-postimp_task-stim_run-08_ieeg.edf",
        None,
        has_emg=False,
    ),
    Case(
        "eeg",
        BIDS / "eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set",
        "eeglab",
        has_emg=False,
    ),
    Case("otb", EX / "one_sessantaquattro_truncated.otb+", "otb", has_emg=True),
]


# Mixed per-channel sample rate -> EDF export must raise (one rate per file).
MIXED_RATE_CASES = [
    Case("xdf", EX / "multi_stream_test.xdf", None, has_emg=True),
    Case("trigno", EX / "truncated_trigno_sample.csv", "trigno", has_emg=True),
]


def _roundtrip(case):
    """Import -> export (auto) -> reimport once per fixture (cached)."""
    if case.name not in _RT_CACHE:
        emg = Recording.from_file(str(case.path), importer=case.importer)
        out = os.path.join(_RT_DIR, f"{case.name}.edf")
        # format="auto" exercises the real EDF/BDF selection (and ignores
        # bypass_analysis by design), which is what a round-trip harness wants.
        emg.to_edf(out, format="auto")
        written = out if os.path.exists(out) else os.path.splitext(out)[0] + ".bdf"
        _RT_CACHE[case.name] = (emg, Recording.from_file(written))
    return _RT_CACHE[case.name]


@pytest.mark.parametrize("case", ROUNDTRIP_CASES, ids=lambda c: c.name)
def test_roundtrip_preserves_structure(case):
    """No sample loss, channel count preserved, no modality creep (all fixtures)."""
    if not case.path.exists():
        pytest.skip(f"fixture missing: {case.path}")
    emg, reloaded = _roundtrip(case)

    # No samples lost (modulo one EDF data-record of zero padding).
    rate = max(int(c["sample_frequency"]) for c in emg.channels.values())
    assert emg.signals.shape[0] <= reloaded.signals.shape[0] <= emg.signals.shape[0] + rate
    assert len(reloaded.channels) == len(emg.channels)

    # No modality creep: a fixture with no EMG must not gain EMG channels.
    if not case.has_emg:
        reloaded_emg = [c for c, i in reloaded.channels.items() if i["channel_type"] == "EMG"]
        assert not reloaded_emg, f"{case.name}: EMG channels appeared after round-trip"


@pytest.mark.parametrize("case", ROUNDTRIP_CASES, ids=lambda c: c.name)
def test_roundtrip_preserves_signal_values(case):
    """Per-channel waveform fidelity on a random 10 s window (r > 0.99).

    Channels with meaningful variance must reach Pearson r > 0.99 after the real
    EDF/BDF round-trip; near-constant channels have undefined correlation, so
    their value is checked by NRMSE instead. Proves the #61 corruption is gone
    (corrupted channels collapse to r ~ 0 even when amplitude-relative NRMSE
    stays small).
    """
    if not case.path.exists():
        pytest.skip(f"fixture missing: {case.path}")
    emg, reloaded = _roundtrip(case)

    rate = max(int(c["sample_frequency"]) for c in emg.channels.values())
    n = emg.signals.shape[0]
    width = min(int(_WINDOW_S * rate), n)
    start = int(_RNG.integers(0, n - width + 1)) if n > width else 0
    window = slice(start, start + width)

    compared = 0
    low_corr = []
    for ch in emg.channels:
        assert ch in reloaded.signals.columns, f"{case.name}: channel '{ch}' lost on reload"
        original = emg.signals[ch].values[window].astype(float)
        roundtripped = reloaded.signals[ch].values[window].astype(float)
        compared += 1
        ptp = float(np.ptp(original))
        if ptp <= _CONST_PTP or np.std(original) == 0.0:
            # Near-constant channel: normalize by 1.0 (raw RMSE), never by the
            # tiny ptp, which would blow a single-LSB error up past tolerance.
            divisor = ptp if ptp > _CONST_PTP else 1.0
            nrmse = float(np.sqrt(np.mean((original - roundtripped) ** 2)) / divisor)
            assert nrmse < case.tolerance, f"{case.name}:{ch} near-constant nrmse {nrmse:.3g}"
            continue
        corr = float(np.corrcoef(original, roundtripped)[0, 1])
        if not corr > 0.99:
            low_corr.append((ch, round(corr, 4)))
    assert compared, "no channels were compared"
    assert not low_corr, f"{case.name}: channels below r>0.99: {low_corr}"


@pytest.mark.parametrize("case", MIXED_RATE_CASES, ids=lambda c: c.name)
def test_mixed_rate_export_raises(case, tmp_path):
    if not case.path.exists():
        pytest.skip(f"fixture missing: {case.path}")
    emg = Recording.from_file(str(case.path), importer=case.importer)
    rates = {int(c["sample_frequency"]) for c in emg.channels.values()}
    if len(rates) <= 1:
        pytest.skip(f"{case.name}: fixture is single-rate, guard not exercised")
    with pytest.raises(ValueError, match="single sampling rate"):
        emg.to_edf(str(tmp_path / f"{case.name}.edf"), format="edf", bypass_analysis=True)


def test_meg_fif_imports_via_mne():
    """MEG .fif now imports through the MNE-backed importer (#53)."""
    meg = BIDS / "meg/sub-01/meg/sub-01_task-mouse_meg.fif"
    if not meg.exists():
        pytest.skip("MEG fixture missing")
    pytest.importorskip("mne", reason="MEG import requires the optional 'meg' extra (mne)")
    emg = Recording.from_file(str(meg))
    assert len(emg.channels) > 0
    # No modality creep: a MEG file must not invent EMG channels.
    assert not [c for c, i in emg.channels.items() if i["channel_type"] == "EMG"]


def test_event_roundtrip_through_edf(tmp_path):
    """Events exported as EDF+ annotations survive a real reimport (issue #47).

    Events are added explicitly (the source fixture has none) so this exercises
    export-write -> annotation read-back. Onset/duration are checked within EDF+'s
    representable precision; descriptions must match exactly.
    """
    fixture = BIDS / "emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf"
    if not fixture.exists():
        pytest.skip("EMG fixture missing")
    emg = Recording.from_file(str(fixture))
    emg.add_event(onset=0.5, duration=0.0, description="m1")
    emg.add_event(onset=1.0, duration=0.2, description="m2")
    out = tmp_path / "ev.edf"
    emg.to_edf(str(out), format="edf", bypass_analysis=True)

    reloaded = Recording.from_file(str(out), bids_channels="off")
    assert len(reloaded.events) == 2
    assert list(reloaded.events["description"]) == ["m1", "m2"]
    assert np.allclose(reloaded.events["onset"].to_numpy(), [0.5, 1.0], atol=1e-3)
    assert np.allclose(reloaded.events["duration"].to_numpy(), [0.0, 0.2], atol=1e-3)
