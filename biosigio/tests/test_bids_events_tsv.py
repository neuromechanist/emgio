"""Tests for BIDS events.tsv loading (issue #94).

A BIDS ``_events.tsv`` is the authoritative event table for a recording, richer
than the data file's own markers. ``bids.apply_events_tsv`` loads it into
``rec.events`` (the symmetric counterpart of ``apply_channels_tsv``). NO MOCKS:
uses the real CC0 EEG BIDS fixture and real on-disk TSV files written per test.
"""

import pathlib

import pytest

from biosigio import Recording
from biosigio.bids import apply_events_tsv, find_events_tsv

_REPO = pathlib.Path(__file__).resolve().parents[2]
EEG_SET = _REPO / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set"
EEG_EVENTS = _REPO / "examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_events.tsv"


@pytest.mark.skipif(not EEG_SET.exists(), reason="EEG BIDS fixture missing")
def test_apply_events_tsv_real_fixture():
    """The real EEG fixture's events.tsv loads with exact onsets/descriptions."""
    rec = Recording.from_file(str(EEG_SET), importer="eeglab")
    n = apply_events_tsv(rec, str(EEG_EVENTS))
    assert n == 3
    assert rec.events["onset"].tolist() == pytest.approx([0.5, 20.0, 40.0])
    assert rec.events["duration"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert list(rec.events["description"]) == [
        "eyes_open_start",
        "fixation_check",
        "rest_marker",
    ]


@pytest.mark.skipif(not EEG_SET.exists(), reason="EEG BIDS fixture missing")
def test_apply_events_tsv_replaces_native_events():
    """events.tsv is authoritative: it overwrites importer-loaded events."""
    rec = Recording.from_file(str(EEG_SET), importer="eeglab")
    rec.add_event(onset=999.0, duration=0.0, description="STALE")
    apply_events_tsv(rec, str(EEG_EVENTS))
    assert "STALE" not in list(rec.events["description"])
    assert len(rec.events) == 3


@pytest.mark.skipif(not EEG_SET.exists(), reason="EEG BIDS fixture missing")
def test_find_events_tsv_resolves_sibling():
    assert find_events_tsv(str(EEG_SET)) is not None
    assert find_events_tsv("/tmp/nonexistent_eeg.edf") is None


def _write_tsv(path: pathlib.Path, text: str) -> None:
    path.write_text(text)


def test_description_prefers_trial_type_then_value(tmp_path):
    """Per row: trial_type if present and not n/a, else value, else 'n/a'."""
    rec = Recording()
    p = tmp_path / "sub-x_events.tsv"
    _write_tsv(
        p,
        "onset\tduration\ttrial_type\tvalue\n"
        "0.0\t1.0\tgo\tS1\n"  # trial_type wins
        "1.0\t0\tn/a\tS2\n"  # trial_type n/a -> value
        "2.0\tn/a\t\tS3\n"  # trial_type empty -> value; duration n/a -> 0.0
        "3.0\t0\tn/a\tn/a\n",  # both n/a -> 'n/a'
    )
    n = apply_events_tsv(rec, str(p))
    assert n == 4
    assert list(rec.events["description"]) == ["go", "S2", "S3", "n/a"]
    assert rec.events["duration"].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_description_column_override(tmp_path):
    rec = Recording()
    p = tmp_path / "sub-x_events.tsv"
    _write_tsv(p, "onset\tduration\ttrial_type\tvalue\n0.0\t0\tgo\tS1\n")
    apply_events_tsv(rec, str(p), description_column="value")
    assert list(rec.events["description"]) == ["S1"]


def test_skips_rows_with_missing_onset(tmp_path):
    rec = Recording()
    p = tmp_path / "sub-x_events.tsv"
    _write_tsv(
        p,
        "onset\tduration\tvalue\n"
        "0.0\t0\tA\n"
        "n/a\t0\tB\n"  # non-numeric onset -> skipped
        "2.0\t0\tC\n",
    )
    n = apply_events_tsv(rec, str(p))
    assert n == 2
    assert list(rec.events["description"]) == ["A", "C"]


def test_events_sorted_by_onset(tmp_path):
    rec = Recording()
    p = tmp_path / "sub-x_events.tsv"
    _write_tsv(p, "onset\tduration\tvalue\n5.0\t0\tlate\n1.0\t0\tearly\n")
    apply_events_tsv(rec, str(p))
    assert rec.events["onset"].tolist() == pytest.approx([1.0, 5.0])
    assert list(rec.events["description"]) == ["early", "late"]
