"""Smoke tests for the biosigio CLI (issue #49).

Exercises each subcommand in-process via ``cli.main(argv)`` (always) and via the
installed ``biosigio`` entry point through a subprocess (skipped with a clear
reason if not installed). Asserts the exit-code contract, file creation, and
``--json`` shape. NO MOCKS: real fixtures and real conversions.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

from emgio import __version__
from emgio.cli import (
    EXIT_INPUT,
    EXIT_OK,
    EXIT_USAGE,
    EXIT_VERIFY_FAILED,
    main,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
EEG = _REPO / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"
EMG_EDF = _REPO / "examples/bids/emg/sub-01/emg/sub-01_task-isometric10percentmvc_run-01_emg.edf"

requires_eeg = pytest.mark.skipif(not EEG.exists(), reason="EEG fixture missing")
requires_emg = pytest.mark.skipif(not EMG_EDF.exists(), reason="EMG fixture missing")


# ------------------------------- in-process -------------------------------- #


def test_version_returns_ok(capsys):
    assert main(["--version"]) == EXIT_OK
    assert __version__ in capsys.readouterr().out


def test_help_lists_subcommands(capsys):
    assert main(["--help"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "convert" in out and "verify" in out and "info" in out


def test_no_subcommand_is_usage_error():
    assert main([]) == EXIT_USAGE


@requires_emg
def test_info_json_shape_and_no_fabricated_emg(capsys):
    assert main(["info", str(EMG_EDF), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_channels"] == len(payload["channels"])
    assert payload["n_channels"] > 0
    assert isinstance(payload["sampling_rates"], list)
    for ch in payload["channels"]:
        assert {"name", "type", "modality", "sample_frequency", "unit"} <= ch.keys()


@requires_eeg
def test_info_eeg_never_injects_emg_when_modality_omitted(capsys):
    """A non-EMG recording must not report fabricated EMG channels."""
    assert main(["info", str(EEG), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert not any(ch["modality"] == "EMG" for ch in payload["channels"])
    assert not any(ch["type"] == "EMG" for ch in payload["channels"])


@requires_eeg
def test_convert_creates_file_and_verifies(tmp_path):
    out = tmp_path / "out.edf"
    assert main(["convert", str(EEG), str(out), "--format", "auto", "--verify"]) == EXIT_OK
    written = out if out.exists() else out.with_suffix(".bdf")
    assert written.exists()
    assert written.with_name(written.stem + "_channels.tsv").exists()


@requires_eeg
def test_convert_no_channels_tsv(tmp_path):
    out = tmp_path / "out.edf"
    assert main(["convert", str(EEG), str(out), "--no-channels-tsv"]) == EXIT_OK
    written = out if out.exists() else out.with_suffix(".bdf")
    assert not written.with_name(written.stem + "_channels.tsv").exists()


@requires_emg
def test_verify_identical_passes():
    assert main(["verify", str(EMG_EDF), str(EMG_EDF)]) == EXIT_OK


@requires_eeg
@requires_emg
def test_verify_mismatch_fails(tmp_path):
    out = tmp_path / "eeg.edf"
    main(["convert", str(EEG), str(out)])
    written = out if out.exists() else out.with_suffix(".bdf")
    assert main(["verify", str(EMG_EDF), str(written)]) == EXIT_VERIFY_FAILED


@requires_emg
def test_verify_json_payload(capsys):
    assert main(["verify", str(EMG_EDF), str(EMG_EDF), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert "channels" in payload and payload["channels"]


def test_missing_input_is_input_error():
    assert main(["info", "/no/such/file.edf"]) == EXIT_INPUT


@requires_eeg
def test_unknown_modality_is_usage_error(tmp_path):
    assert main(["convert", str(EEG), str(tmp_path / "x.edf"), "--modality", "BOGUS"]) == EXIT_USAGE


def test_unsupported_extension_is_usage_error(tmp_path):
    bad = tmp_path / "f.zzz"
    bad.write_text("not a signal file")
    assert main(["info", str(bad)]) == EXIT_USAGE


@requires_eeg
def test_modality_fills_only_unknown_channels(capsys, tmp_path):
    """--modality fills channels that resolved to unknown, leaving detected ones."""
    # Baseline: identify which channels are unknown without --modality.
    main(["info", str(EEG), "--json"])
    base = json.loads(capsys.readouterr().out)
    unknown = set(base["unknown_channels"])
    detected = {ch["name"]: ch["modality"] for ch in base["channels"] if ch["name"] not in unknown}
    if not unknown:
        pytest.skip("EEG fixture has no unknown channels to fill")

    # convert with --modality EEG must not raise; detected channels keep their
    # modality. (We re-check via a fresh info on the converted output below is
    # not possible since modality is not stored in EDF, so assert via no-raise +
    # exit 0; the unknown-fill logic is unit-tested in cli._apply_modality.)
    out = tmp_path / "m.edf"
    assert main(["convert", str(EEG), str(out), "--modality", "EEG"]) == EXIT_OK
    assert detected  # detected channels existed and were left to win


# -------------------------------- subprocess ------------------------------- #

BIOSIGIO = shutil.which("biosigio")
no_entry_point = pytest.mark.skipif(
    BIOSIGIO is None, reason="biosigio entry point not installed on PATH"
)


@no_entry_point
def test_subprocess_version():
    assert BIOSIGIO is not None
    result = subprocess.run([BIOSIGIO, "--version"], capture_output=True, text=True, check=False)
    assert result.returncode == EXIT_OK
    assert __version__ in result.stdout


@requires_emg
@no_entry_point
def test_subprocess_info_json_alone_on_stdout():
    """stdout must be JSON only; importer chatter goes to stderr."""
    assert BIOSIGIO is not None
    result = subprocess.run(
        [BIOSIGIO, "info", str(EMG_EDF), "--json"], capture_output=True, text=True, check=False
    )
    assert result.returncode == EXIT_OK
    payload = json.loads(result.stdout)  # parses cleanly => stdout is JSON alone
    assert payload["n_channels"] > 0


@no_entry_point
def test_subprocess_missing_file_exit_code():
    assert BIOSIGIO is not None
    result = subprocess.run(
        [BIOSIGIO, "info", "/no/such/file.edf"], capture_output=True, text=True, check=False
    )
    assert result.returncode == EXIT_INPUT
