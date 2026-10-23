"""Header start date/time is derived from the recording, not the wall clock.

EDF/BDF export used to leave pyedflib's default in place, so the header's
start field held ``datetime.now()`` at export time: two exports of the same
recording differed, and the conversion date sat in a field readers treat as
the acquisition time (#174). These tests pin the resolution order: explicit
argument first, then ``rec.metadata["startdate"]`` when the importer filled
it, then the anonymized ``01.01.85 00.00.00`` from the EDF specification.
"""
import datetime

import numpy as np
import pytest

from biosigio import Recording
from biosigio.exporters.edf import EDFExporter, _resolve_start_datetime


class _FakeEdfWriter:
    """Records what the exporter sets, so the tests assert the real call path."""

    last = None

    def __init__(self, path, n_signals, file_type=None):
        self.path = path
        self.file_type = file_type
        self.start_datetime = "unset"
        self.signal_headers = None
        self.samples = None
        self.closed = False
        # The exporter verifies the output file exists and is non-empty.
        with open(path, "wb") as handle:
            handle.write(b"fake-edf")
        _FakeEdfWriter.last = self

    def setStartdatetime(self, value):
        self.start_datetime = value

    def setSignalHeaders(self, headers):
        self.signal_headers = headers

    def writeSamples(self, signals):
        self.samples = signals

    def writeAnnotation(self, onset, duration, description):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def fake_edflib(monkeypatch):
    import biosigio.exporters.edf as edf_module

    monkeypatch.setattr(edf_module.pyedflib, "EdfWriter", _FakeEdfWriter, raising=True)
    return _FakeEdfWriter


def _recording(startdate=None):
    rec = Recording()
    rec.add_channel("c0", np.arange(8, dtype=float), 8, "uV", "EEG")
    if startdate is not None:
        rec.set_metadata("startdate", startdate)
    return rec


def test_export_sets_the_header_start_from_metadata(fake_edflib, tmp_path):
    start = datetime.datetime(2024, 3, 1, 12, 30, 0)
    EDFExporter.export(
        _recording(start), str(tmp_path / "out"), format="bdf",
        bypass_analysis=True, create_channels_tsv=False,
    )
    assert fake_edflib.last.start_datetime == start


def test_explicit_start_datetime_wins_over_metadata(fake_edflib, tmp_path):
    explicit = datetime.datetime(2001, 2, 3, 4, 5, 6)
    start = datetime.datetime(2024, 3, 1, 12, 30, 0)
    EDFExporter.export(
        _recording(start), str(tmp_path / "out"), format="bdf",
        bypass_analysis=True, create_channels_tsv=False,
        start_datetime=explicit,
    )
    assert fake_edflib.last.start_datetime == explicit


def test_unknown_start_is_anonymized_not_the_wall_clock(fake_edflib, tmp_path):
    EDFExporter.export(
        _recording(), str(tmp_path / "out"), format="bdf",
        bypass_analysis=True, create_channels_tsv=False,
    )
    # The EDF specification's value for an unknown start date, so two exports
    # of the same recording produce identical headers.
    assert fake_edflib.last.start_datetime == datetime.datetime(1985, 1, 1)


def test_iso_string_startdate_is_accepted(fake_edflib, tmp_path):
    EDFExporter.export(
        _recording("2020-01-02T03:04:05"), str(tmp_path / "out"), format="bdf",
        bypass_analysis=True, create_channels_tsv=False,
    )
    assert fake_edflib.last.start_datetime == datetime.datetime(2020, 1, 2, 3, 4, 5)


def test_rejects_a_non_datetime_argument():
    with pytest.raises(ValueError, match="start_datetime must be a datetime"):
        EDFExporter.export(
            _recording(), "unused", format="bdf",
            bypass_analysis=True, create_channels_tsv=False,
            start_datetime="2020-01-02",
        )


def test_unparseable_startdate_falls_back_to_the_anonymized_value():
    rec = _recording()
    rec.set_metadata("startdate", "not a date")
    assert _resolve_start_datetime(rec) == datetime.datetime(1985, 1, 1)
