"""EDF/BDF export refuses input it cannot represent without changing it (#173).

NaN samples used to be replaced with 0.0 by ``nan_to_num`` without a word, so
a missing sample read back as a small non-zero value once quantized, and a
sampling rate that is not a whole number of hertz was cut to an integer
silently (1999.5 Hz was written, and read back, as 1999 Hz). Both now raise
``ValueError`` before anything is written; an empty signal keeps its
pre-existing warning.
"""
import numpy as np
import pytest

from biosigio import Recording
from biosigio.exporters.edf import EDFExporter


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


def _export(rec, fake_edflib, tmp_path, **kwargs):
    return EDFExporter.export(
        rec, str(tmp_path / "out"), format="bdf",
        bypass_analysis=True, create_channels_tsv=False, **kwargs
    )


def test_nan_samples_are_refused(fake_edflib, tmp_path):
    rec = Recording()
    values = [float(v) for v in range(8)]
    values[3] = float("nan")
    rec.add_channel("c0", np.array(values), 8, "uV", "EEG")
    with pytest.raises(ValueError, match="Channel 'c0'.*non-finite"):
        _export(rec, fake_edflib, tmp_path)


def test_inf_samples_are_refused(fake_edflib, tmp_path):
    rec = Recording()
    values = [float(v) for v in range(8)]
    values[3] = float("inf")
    rec.add_channel("c0", np.array(values), 8, "uV", "EEG")
    with pytest.raises(ValueError, match="non-finite"):
        _export(rec, fake_edflib, tmp_path)


def test_an_all_nan_channel_is_refused_not_silently_zeroed(fake_edflib, tmp_path):
    rec = Recording()
    rec.add_channel("c0", np.full(8, np.nan), 8, "uV", "EEG")
    with pytest.raises(ValueError, match="non-finite"):
        _export(rec, fake_edflib, tmp_path)


def test_a_fractional_sampling_rate_is_refused(fake_edflib, tmp_path):
    rec = Recording()
    rec.add_channel("c0", np.arange(8, dtype=float), 1999.5, "uV", "EEG")
    with pytest.raises(ValueError, match="1999.5.*not a whole number"):
        _export(rec, fake_edflib, tmp_path)


def test_finite_signal_with_a_whole_number_rate_still_exports(fake_edflib, tmp_path):
    rec = Recording()
    values = np.arange(8, dtype=float)
    rec.add_channel("c0", values, 8, "uV", "EEG")
    path = _export(rec, fake_edflib, tmp_path)
    assert path.endswith(".bdf")
    writer = fake_edflib.last
    assert writer.signal_headers[0]["sample_frequency"] == 8


def test_an_empty_signal_keeps_its_warning(fake_edflib, tmp_path):
    rec = Recording()
    rec.add_channel("c0", np.array([]), 8, "uV", "EEG")
    with pytest.warns(UserWarning, match="empty or all-NaN"):
        _export(rec, fake_edflib, tmp_path)
