"""Low-resolution export pipeline tests (biosigIO phase 2).

Covers ``EMG.resample`` (anti-aliased polyphase down-sampling) and the CLI
``lowres`` subcommand. The headline test is the anti-aliasing proof: a tone
above the new Nyquist must be removed by the resampler's filter, NOT folded
back into the band. NO MOCKS: real fixtures and a real synthetic-signal FFT.
"""

import pathlib
from math import ceil, gcd

import numpy as np
import pytest

from emgio import EMG
from emgio.cli import EXIT_OK, main

_REPO = pathlib.Path(__file__).resolve().parents[2]
EEG = _REPO / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"
EMG_EDF = _REPO / "examples/bids/emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf"

requires_eeg = pytest.mark.skipif(not EEG.exists(), reason="EEG fixture missing")
requires_emg = pytest.mark.skipif(not EMG_EDF.exists(), reason="EMG fixture missing")

_CONST_PTP = 1e-9  # below this peak-to-peak a channel has no defined correlation


def _rates(emg: EMG) -> set:
    return {info["sample_frequency"] for info in emg.channels.values()}


@requires_eeg
def test_resample_eeg_250_to_100_structure():
    """250 -> 100 Hz: new rate on all channels, expected sample count, channels/events kept."""
    emg = EMG.from_file(str(EEG), importer="eeglab")
    n_old = emg.signals.shape[0]
    n_channels = len(emg.channels)
    n_events = len(emg.events)

    rs = emg.resample(100)

    assert _rates(rs) == {100}, "every channel must report the new rate"
    # resample_poly output length is ceil(n * up / down), not round(n * ratio).
    g = gcd(250, 100)
    assert rs.signals.shape[0] == ceil(n_old * (100 // g) / (250 // g))
    assert len(rs.channels) == n_channels, "channel count preserved"
    assert set(rs.signals.columns) == set(emg.signals.columns), "channel names preserved"
    assert len(rs.events) == n_events, "events preserved"

    # Non-destructive: the source is untouched.
    assert emg.signals.shape[0] == n_old
    assert _rates(emg) == {250}


@requires_eeg
def test_resample_preserves_channel_and_recording_metadata():
    """Channel type/modality/units/prefilter and recording metadata survive."""
    emg = EMG.from_file(str(EEG), importer="eeglab")
    emg.set_metadata("subject", "sub-01")
    rs = emg.resample(100)

    for ch, info in emg.channels.items():
        new = rs.channels[ch]
        assert new["channel_type"] == info["channel_type"]
        assert new["modality"] == info["modality"]
        assert new["physical_dimension"] == info["physical_dimension"]
        assert new["prefilter"] == info["prefilter"]
    assert rs.get_metadata("subject") == "sub-01"


def test_resample_anti_aliasing_removes_above_nyquist():
    """A tone above the new Nyquist is attenuated, not folded back (no aliasing).

    Source 500 Hz, signal = 5 Hz + 80 Hz. Resampling to 100 Hz (Nyquist 50 Hz)
    must remove the 80 Hz tone. A naive stride-decimation would alias 80 Hz to
    |80 - 100| = 20 Hz; we assert the 20 Hz bin stays tiny relative to the 5 Hz
    tone, proving resample_poly's anti-alias FIR did its job.
    """
    fs = 500.0
    t = np.arange(int(fs * 10)) / fs
    signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 80 * t)

    emg = EMG()
    emg.add_channel("S", signal, fs, "uV", "EEG")
    rs = emg.resample(100)

    y = rs.signals["S"].to_numpy()
    assert _rates(rs) == {100}
    freqs = np.fft.rfftfreq(len(y), d=1 / 100.0)
    mag = np.abs(np.fft.rfft(y))

    def power_at(f: float) -> float:
        return float(mag[int(np.argmin(np.abs(freqs - f)))])

    p_5 = power_at(5.0)
    p_alias = power_at(20.0)  # where 80 Hz WOULD fold to under naive decimation
    ratio = p_alias / p_5
    # The aliased-fold bin carries < 1% of the retained tone's power.
    assert ratio < 0.01, f"aliasing detected: 20 Hz/5 Hz power ratio {ratio:.4g}"


@requires_eeg
def test_resample_roundtrip_through_edf():
    """resample(100) -> EDF export -> reload: 100 Hz, per-channel r > 0.99 on 10 s."""
    emg = EMG.from_file(str(EEG), importer="eeglab")
    rs = emg.resample(100)

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "lowres.edf"
        rs.to_edf(str(out), format="edf", bypass_analysis=True)
        reloaded = EMG.from_file(str(out), bids_channels="off")

        assert _rates(reloaded) == {100}, "reloaded recording must be 100 Hz"

        rate = 100
        n = rs.signals.shape[0]
        width = min(10 * rate, n)
        window = slice(0, width)

        low_corr = []
        for ch in rs.channels:
            assert ch in reloaded.signals.columns, f"channel {ch} lost on reload"
            original = rs.signals[ch].to_numpy()[window].astype(float)
            roundtripped = reloaded.signals[ch].to_numpy()[window].astype(float)
            ptp = float(np.ptp(original))
            if ptp <= _CONST_PTP or np.std(original) == 0.0:
                divisor = ptp if ptp > _CONST_PTP else 1.0
                nrmse = float(np.sqrt(np.mean((original - roundtripped) ** 2)) / divisor)
                assert nrmse < 0.05, f"{ch} near-constant nrmse {nrmse:.3g}"
                continue
            corr = float(np.corrcoef(original, roundtripped)[0, 1])
            if not corr > 0.99:
                low_corr.append((ch, round(corr, 4)))
        assert not low_corr, f"channels below r>0.99 after lowres round-trip: {low_corr}"


@requires_eeg
def test_cli_lowres_creates_100hz_16bit(tmp_path):
    """CLI lowres exit 0; output reloads at 100 Hz with 16-bit EDF (digital_max 32767)."""
    out = tmp_path / "o.edf"
    assert main(["lowres", str(EEG), str(out), "--rate", "100"]) == EXIT_OK
    assert out.exists()

    reloaded = EMG.from_file(str(out), bids_channels="off")
    assert _rates(reloaded) == {100}

    # 16-bit EDF brackets digital values to int16 (max 32767), confirming the
    # EDF (not BDF/24-bit) path was taken by default.
    import pyedflib

    with pyedflib.EdfReader(str(out)) as r:
        assert r.getDigitalMaximum(0) == 32767


@requires_eeg
def test_cli_lowres_default_is_double_lowres(tmp_path):
    """No --rate/--bits flags => 100 Hz + 16-bit EDF (double low-res default)."""
    out = tmp_path / "d.edf"
    assert main(["lowres", str(EEG), str(out)]) == EXIT_OK
    reloaded = EMG.from_file(str(out), bids_channels="off")
    assert _rates(reloaded) == {100}


@requires_emg
def test_cli_lowres_skips_when_already_low(tmp_path, capsys):
    """Source already <= target rate: export without resampling, with a stderr note."""
    emg = EMG.from_file(str(EMG_EDF))
    source_rate = max(_rates(emg))
    target = source_rate + 100.0  # guarantee source <= target

    out = tmp_path / "skip.edf"
    assert main(["lowres", str(EMG_EDF), str(out), "--rate", str(target)]) == EXIT_OK
    assert out.exists()
    assert "without resampling" in capsys.readouterr().err

    reloaded = EMG.from_file(str(out), bids_channels="off")
    assert _rates(reloaded) == _rates(emg), "rate unchanged when no resampling occurred"


@requires_eeg
def test_resample_equal_rate_returns_unchanged_copy():
    """target == source: a copy with the same rate and sample count, but a new object."""
    emg = EMG.from_file(str(EEG), importer="eeglab")
    rs = emg.resample(250)
    assert rs is not emg
    assert _rates(rs) == {250}
    assert rs.signals.shape == emg.signals.shape
    assert np.allclose(rs.signals.to_numpy(), emg.signals.to_numpy())


@requires_eeg
def test_resample_above_source_raises():
    """Up-sampling is refused (low-res only)."""
    emg = EMG.from_file(str(EEG), importer="eeglab")
    with pytest.raises(ValueError, match="exceeds source rate"):
        emg.resample(500)


def test_resample_non_integer_target_stores_actual_rate():
    """A non-integer target snaps to the achievable rational rate and stores THAT.

    Guards against silent metadata corruption: 256 Hz -> requested 100.4 Hz uses
    integer factors up=25/down=64 -> achieves exactly 100.0 Hz, which must be what
    is written to the channels (not the requested 100.4).
    """
    emg = EMG()
    rng = np.random.default_rng(0)
    emg.add_channel("X", rng.standard_normal(2560), 256, "uV", "EEG")
    rs = emg.resample(100.4)
    assert _rates(rs) == {100.0}  # the ACHIEVED rate, never the requested 100.4
    assert 100.4 not in _rates(rs)


def test_resample_mixed_rate_raises():
    """Channels with distinct sample_frequency cannot be resampled together.

    emgio stores one uniform-length grid, so the two channels share a length;
    only their declared sample_frequency differs, which the guard must reject.
    """
    emg = EMG()
    emg.add_channel("A", np.zeros(1000), 500.0, "uV", "EEG")
    emg.add_channel("B", np.zeros(1000), 250.0, "uV", "EEG")
    with pytest.raises(ValueError, match="single sampling rate"):
        emg.resample(100)


def test_resample_no_signals_raises():
    """An empty EMG cannot be resampled."""
    with pytest.raises(ValueError, match="No signals loaded"):
        EMG().resample(100)
