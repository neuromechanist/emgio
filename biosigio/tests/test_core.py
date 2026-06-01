import builtins
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyedflib
import pytest

from ..core.emg import Recording


@pytest.fixture
def empty_rec():
    """Create an empty Recording object."""
    return Recording()


@pytest.fixture
def sample_rec():
    """Create a Recording object with sample data."""
    rec = Recording()

    # Add sample channels
    time = np.linspace(0, 1, 1000)  # 1 second at 1000Hz
    emg_data = np.sin(2 * np.pi * 10 * time)  # 10Hz sine wave
    acc_data = np.cos(2 * np.pi * 5 * time)  # 5Hz cosine wave

    rec.add_channel("EMG1", emg_data, 1000, "mV", channel_type="EMG")
    rec.add_channel("ACC1", acc_data, 1000, "g", channel_type="ACC")

    return rec


def test_emg_initialization(empty_rec):
    """Test Recording object initialization."""
    assert empty_rec.signals is None
    assert empty_rec.metadata == {}
    assert empty_rec.channels == {}


def test_add_channel(empty_rec):
    """Test adding a channel to Recording object."""
    data = np.array([1, 2, 3, 4, 5])
    empty_rec.add_channel("EMG1", data, 1000, "mV", "EMG")

    assert "EMG1" in empty_rec.signals.columns
    assert "EMG1" in empty_rec.channels
    assert empty_rec.channels["EMG1"]["sample_frequency"] == 1000
    assert empty_rec.channels["EMG1"]["physical_dimension"] == "mV"
    assert empty_rec.channels["EMG1"]["channel_type"] == "EMG"


def test_select_channels(sample_rec):
    """Test channel selection."""
    # Store original state
    original_channels = list(sample_rec.signals.columns)

    # Select multiple channels
    rec_multi = sample_rec.select_channels(["EMG1", "ACC1"])
    assert list(rec_multi.signals.columns) == ["EMG1", "ACC1"]
    assert list(rec_multi.channels.keys()) == ["EMG1", "ACC1"]
    # Verify original object is unchanged
    assert list(sample_rec.signals.columns) == original_channels

    # Select single channel
    rec_single = sample_rec.select_channels("EMG1")
    assert list(rec_single.signals.columns) == ["EMG1"]
    assert list(rec_single.channels.keys()) == ["EMG1"]
    # Verify original object is unchanged
    assert list(sample_rec.signals.columns) == original_channels


def test_metadata(empty_rec):
    """Test metadata handling."""
    empty_rec.set_metadata("subject", "S001")
    assert empty_rec.get_metadata("subject") == "S001"

    # Test non-existent key
    assert empty_rec.get_metadata("nonexistent") is None


def test_invalid_channel_selection(sample_rec):
    """Test error handling for invalid channel selection."""
    with pytest.raises(ValueError):
        sample_rec.select_channels("NonexistentChannel")


def test_plot_signals_validation(empty_rec):
    """Test plot_signals input validation."""
    with pytest.raises(ValueError):
        empty_rec.plot_signals()  # Should raise error when no signals are loaded


def test_get_channel_types(sample_rec):
    """Test getting unique channel types."""
    types = sample_rec.get_channel_types()
    assert set(types) == {"EMG", "ACC"}


def test_get_channels_by_type(sample_rec):
    """Test getting channels of specific type."""
    emg_channels = sample_rec.get_channels_by_type("EMG")
    acc_channels = sample_rec.get_channels_by_type("ACC")

    assert emg_channels == ["EMG1"]
    assert acc_channels == ["ACC1"]
    assert sample_rec.get_channels_by_type("NONEXISTENT") == []


# This will be implemented after #3 is resolved
# def test_select_channels_by_type(sample_rec):
#     """Test channel selection by type."""
#     # Select all EMG channels
#     emg_only = sample_rec.select_channels(channel_type='EMG')
#     assert list(emg_only.signals.columns) == ['EMG1']
#     assert all(info['channel_type'] == 'EMG' for info in emg_only.channels.values())

#     # Select all ACC channels
#     acc_only = sample_rec.select_channels(channel_type='ACC')
#     assert list(acc_only.signals.columns) == ['ACC1']
#     assert all(info['channel_type'] == 'ACC' for info in acc_only.channels.values())

#     # Test with non-existent type
#     with pytest.raises(ValueError):
#         sample_rec.select_channels(channel_type='NONEXISTENT')


def test_select_channels_with_type_filter(sample_rec):
    """Test channel selection with type filtering."""
    # Store original state
    original_channels = list(sample_rec.signals.columns)

    # Select specific channels with type filter
    result = sample_rec.select_channels(["EMG1", "ACC1"], channel_type="EMG")
    assert list(result.signals.columns) == ["EMG1"]
    assert all(info["channel_type"] == "EMG" for info in result.channels.values())
    # Verify original object is unchanged
    assert list(sample_rec.signals.columns) == original_channels

    # Test when no channels match type
    with pytest.raises(ValueError):
        sample_rec.select_channels(["EMG1", "ACC1"], channel_type="GYRO")
    # Verify original object is unchanged after error
    assert list(sample_rec.signals.columns) == original_channels


def test_add_channel_validation(empty_rec):
    """Test add_channel with various data types and validation."""
    # Test with different numpy data types
    data_int = np.array([1, 2, 3], dtype=np.int32)
    empty_rec.add_channel("INT", data_int, 1000, "count", channel_type="OTHER")
    assert np.array_equal(empty_rec.signals["INT"].values, data_int)

    # Test with float data
    data_float = np.array([1.1, 2.2, 3.3])
    empty_rec.add_channel("FLOAT", data_float, 1000, "mV", "EMG")
    assert np.array_equal(empty_rec.signals["FLOAT"].values, data_float)

    # Test channel info storage
    assert empty_rec.channels["INT"]["channel_type"] == "OTHER"
    assert empty_rec.channels["FLOAT"]["channel_type"] == "EMG"  # default type
    assert empty_rec.channels["INT"]["sample_frequency"] == 1000
    assert empty_rec.channels["INT"]["physical_dimension"] == "count"


@pytest.fixture
def mock_importers(monkeypatch):
    """Mock importers for testing from_file method."""

    class MockBaseImporter:
        """Base class for mock importers to ensure consistent interface."""

        def load(self, filepath, **kwargs):
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            return self._load(filepath, **kwargs)

    class MockTrignoImporter(MockBaseImporter):
        def _load(self, filepath, **kwargs):
            rec = Recording()
            rec.add_channel("TEST", np.array([1, 2, 3]), 1000, "mV", channel_type="EMG")
            rec.set_metadata("device", "Delsys Trigno")
            rec.set_metadata("source_file", filepath)
            return rec

    class MockOTBImporter(MockBaseImporter):
        def _load(self, filepath, **kwargs):
            rec = Recording()
            rec.add_channel("OTB", np.array([4, 5, 6]), 2000, "mV", channel_type="EMG")
            rec.set_metadata("device", "OT Bioelettronica")
            rec.set_metadata("source_file", filepath)
            return rec

    class MockCSVImporter(MockBaseImporter):
        def _detect_specialized_format(self, filepath):
            # Mock format detection - anything with 'trigno' in the name is detected as Trigno
            if "trigno" in filepath.lower():
                return "trigno"
            return None

        def _load(self, filepath, force_generic=False, **kwargs):
            if not force_generic:
                detected_format = self._detect_specialized_format(filepath)
                if detected_format == "trigno":
                    raise ValueError(
                        "This file appears to be a Delsys Trigno CSV export. "
                        "For better metadata extraction and channel detection, use:\n\n"
                        "recording = Recording.from_file(filepath, importer='trigno')\n\n"
                        "If you still want to use the generic CSV importer, set force_generic=True"
                    )

            rec = Recording()
            rec.add_channel("CSV_CH1", np.array([7, 8, 9]), 1000, "mV", channel_type="EMG")
            rec.set_metadata("file_format", "CSV")
            rec.set_metadata("source_file", filepath)
            return rec

    def mock_import(name, *args):
        # Only intercept our specific importer paths
        if any(
            x in name
            for x in [
                "biosigio.importers.trigno",
                "biosigio.importers.otb",
                "biosigio.importers.csv",
            ]
        ):
            if "trigno" in name:
                return type("TrignoModule", (), {"TrignoImporter": MockTrignoImporter})
            elif "otb" in name:
                return type("OTBModule", (), {"OTBImporter": MockOTBImporter})
            elif "csv" in name:
                return type("CSVModule", (), {"CSVImporter": MockCSVImporter})
        # Let all other imports pass through to the original __import__
        return original_import(name, *args)

    original_import = builtins.__import__

    monkeypatch.setattr("builtins.__import__", mock_import)


def test_from_file(mock_importers, tmp_path):
    """Test factory method with different importers."""
    # Create temporary test files
    trigno_file = tmp_path / "test.csv"
    trigno_file.write_text("")  # Empty file is sufficient for testing

    otb_file = tmp_path / "test.otb"
    otb_file.write_text("")

    csv_file = tmp_path / "test.txt"
    csv_file.write_text("")

    trigno_named_file = tmp_path / "trigno_data.csv"
    trigno_named_file.write_text("")

    # Test Trigno importer
    rec_trigno = Recording.from_file(str(trigno_file), importer="trigno")
    assert "TEST" in rec_trigno.signals.columns
    assert rec_trigno.channels["TEST"]["sample_frequency"] == 1000

    # Test OTB importer (including auto-detection)
    for importer in ["otb", None]:
        rec_otb = Recording.from_file(str(otb_file), importer=importer)
        assert "OTB" in rec_otb.signals.columns
        assert rec_otb.channels["OTB"]["sample_frequency"] == 2000

    # Test CSV importer
    rec_csv = Recording.from_file(str(csv_file), importer="csv")
    assert "CSV_CH1" in rec_csv.signals.columns
    assert rec_csv.get_metadata("file_format") == "CSV"

    # Test CSV importer with auto-detection for .txt files
    rec_txt = Recording.from_file(str(csv_file), importer=None)
    assert "CSV_CH1" in rec_txt.signals.columns
    assert rec_txt.get_metadata("file_format") == "CSV"

    # Test format detection and force_csv
    # First, test that format detection raises an error for trigno file
    with pytest.raises(ValueError, match="Delsys Trigno CSV export"):
        Recording.from_file(str(trigno_named_file), importer="csv")

    # Now test with force_csv=True to bypass detection
    rec_forced = Recording.from_file(str(trigno_named_file), importer="csv", force_csv=True)
    assert "CSV_CH1" in rec_forced.signals.columns

    # Test passing parameters to CSV importer
    custom_kwargs = {"channel_types": {"CSV_CH1": "ACC"}}
    rec_with_params = Recording.from_file(str(csv_file), importer="csv", **custom_kwargs)
    assert "CSV_CH1" in rec_with_params.signals.columns

    # Test invalid importer
    with pytest.raises(ValueError, match="Unsupported importer"):
        Recording.from_file(str(trigno_file), importer="invalid")


class MockPlt:
    """Custom mock for matplotlib.pyplot."""

    def __init__(self):
        self.fig = MagicMock()
        self.reset()

    def reset(self):
        """Reset the mock state."""
        self.show_called = False
        self.subplots_called = False
        self.axes = []

    def subplots(self, nrows=1, ncols=1, **kwargs):
        """Mock subplots creation."""
        self.subplots_called = True
        if nrows == 1:
            # For single subplot, return a single MagicMock with list-like behavior
            self.axes = MagicMock()
            self.axes.__iter__ = lambda x: iter([self.axes])
            self.axes.__len__ = lambda x: 1
            self.axes.__getitem__ = lambda x, i: self.axes
        else:
            # For multiple subplots, return a list of MagicMocks
            self.axes = [MagicMock() for _ in range(nrows)]
        return self.fig, self.axes

    def show(self):
        """Mock show function."""
        self.show_called = True

    def tight_layout(self):
        """Mock tight_layout function."""
        pass


@pytest.fixture
def mock_plt(monkeypatch):
    """Mock matplotlib.pyplot for testing plot functions."""
    mock = MockPlt()
    monkeypatch.setattr("biosigio.visualization.static.plt", mock)
    return mock


@pytest.mark.skip(reason="visualization not critical for core functionality")
def test_plot_signals_basic(sample_rec, mock_plt):
    """Test basic plotting functionality."""
    sample_rec.plot_signals(show=False, plt_module=mock_plt)

    # Verify figure creation
    assert mock_plt.subplots_called

    # Verify plot calls on each axis
    if isinstance(mock_plt.axes, list):
        for ax in mock_plt.axes:
            ax.plot.assert_called_once()
    else:
        mock_plt.axes.plot.assert_called_once()

    # Verify show was called
    assert mock_plt.show_called


@pytest.mark.skip(reason="visualization not critical for core functionality")
def test_plot_signals_style_options(sample_rec, mock_plt):
    """Test different plot styles."""
    # Test dots style
    sample_rec.plot_signals(show=False, plt_module=mock_plt)
    if isinstance(mock_plt.axes, list):
        for ax in mock_plt.axes:
            ax.scatter.assert_called_once()
            ax.plot.assert_not_called()
    else:
        mock_plt.axes.scatter.assert_called_once()
        mock_plt.axes.plot.assert_not_called()
    assert mock_plt.show_called

    mock_plt.reset()

    # Test line style
    sample_rec.plot_signals(show=False, plt_module=mock_plt)
    if isinstance(mock_plt.axes, list):
        for ax in mock_plt.axes:
            ax.plot.assert_called_once()
            ax.scatter.assert_not_called()
    else:
        mock_plt.axes.plot.assert_called_once()
        mock_plt.axes.scatter.assert_not_called()
    assert mock_plt.show_called


@pytest.mark.skip(reason="visualization not critical for core functionality")
def test_plot_signals_customization(sample_rec, mock_plt):
    """Test plot customization options."""
    title = "Test Plot"
    sample_rec.plot_signals(
        channels=["EMG1"], grid=True, title=title, show=False, plt_module=mock_plt
    )

    # Verify title
    mock_plt.fig.suptitle.assert_called_with(title, fontsize=14, y=1.02)

    # Verify grid
    mock_plt.axes.grid.assert_called_with(True, linestyle="--", alpha=0.7)

    # Verify show was called
    assert mock_plt.show_called


def test_plot_signals_channel_selection(sample_rec, mock_plt):
    """Test plotting with channel selection."""
    # Test single channel
    sample_rec.plot_signals(channels=["EMG1"], show=False, plt_module=mock_plt)
    assert not isinstance(mock_plt.axes, list)  # Should be single axis

    mock_plt.reset()

    # Test invalid channel
    with pytest.raises(ValueError) as exc_info:
        sample_rec.plot_signals(channels=["NonexistentChannel"])
    assert "Channels not found" in str(exc_info.value)


def test_plot_signals_time_range(sample_rec, mock_plt):
    """Test plotting with time range selection."""
    time_range = (0.2, 0.8)
    sample_rec.plot_signals(time_range=time_range, show=False, plt_module=mock_plt)

    # Verify data selection
    if isinstance(mock_plt.axes, list):
        for ax in mock_plt.axes:
            plot_calls = ax.plot.call_args_list
            assert len(plot_calls) > 0, "No plot calls were made"
            data = plot_calls[-1][0][1]  # Get y-values from last plot call
            assert len(data) < len(sample_rec.signals)  # Should be subset of data
    else:
        plot_calls = mock_plt.axes.plot.call_args_list
        assert len(plot_calls) > 0, "No plot calls were made"
        data = plot_calls[-1][0][1]  # Get y-values from last plot call
        assert len(data) < len(sample_rec.signals)  # Should be subset of data


def test_emg_add_event(empty_rec):
    """Test adding events to the Recording object."""
    assert empty_rec.events.empty

    # Add first event
    empty_rec.add_event(onset=1.0, duration=0.5, description="Event A")
    assert len(empty_rec.events) == 1
    pd.testing.assert_frame_equal(
        empty_rec.events, pd.DataFrame([{"onset": 1.0, "duration": 0.5, "description": "Event A"}])
    )
    # onset/duration must be float64 (not object) even from the empty frame
    assert empty_rec.events["onset"].dtype == np.float64
    assert empty_rec.events["duration"].dtype == np.float64

    # Add second event (should be sorted)
    empty_rec.add_event(onset=0.5, duration=0.1, description="Event B")
    assert len(empty_rec.events) == 2
    expected_df = pd.DataFrame(
        [
            {"onset": 0.5, "duration": 0.1, "description": "Event B"},
            {"onset": 1.0, "duration": 0.5, "description": "Event A"},
        ]
    )
    pd.testing.assert_frame_equal(empty_rec.events, expected_df)

    # Add third event
    empty_rec.add_event(onset=1.5, duration=0.0, description="Event C")
    assert len(empty_rec.events) == 3
    expected_df = pd.DataFrame(
        [
            {"onset": 0.5, "duration": 0.1, "description": "Event B"},
            {"onset": 1.0, "duration": 0.5, "description": "Event A"},
            {"onset": 1.5, "duration": 0.0, "description": "Event C"},
        ]
    )
    pd.testing.assert_frame_equal(empty_rec.events, expected_df)
    # dtype must survive the concat path too
    assert empty_rec.events["onset"].dtype == np.float64
    assert empty_rec.events["duration"].dtype == np.float64


def test_add_event_integer_inputs_coerced_to_float(empty_rec):
    """Integer onset/duration must be stored as float64, not int64 or object."""
    empty_rec.add_event(onset=2, duration=0, description="Integer onset")
    assert empty_rec.events["onset"].dtype == np.float64
    assert empty_rec.events["duration"].dtype == np.float64
    assert empty_rec.events.loc[0, "onset"] == 2.0
    assert empty_rec.events.loc[0, "duration"] == 0.0


def test_to_edf_writes_real_file_and_channels_tsv(sample_rec, tmp_path):
    """to_edf writes a real EDF/BDF file plus a BIDS channels.tsv by default."""
    out = tmp_path / "out.edf"
    sample_rec.to_edf(str(out), format="bdf", bypass_analysis=True)
    written = out if out.exists() else out.with_suffix(".bdf")
    assert written.exists()
    assert written.with_name(written.stem + "_channels.tsv").exists()
    reloaded = Recording.from_file(str(written), bids_channels="off")
    assert set(reloaded.signals.columns) == {"EMG1", "ACC1"}


def test_to_edf_bypass_analysis_defaulting(sample_rec, tmp_path, capsys):
    """Forced format skips analysis by default; 'auto' always analyzes.

    The decision is observable in the exporter's output ('Summary skipped...' when
    analysis is bypassed, 'Recommended Format:' when it runs), so this verifies
    the real to_edf -> EDFExporter behaviour without mocking the exporter.
    """
    # Forced format, bypass=None (default) -> analysis skipped.
    sample_rec.to_edf(str(tmp_path / "a.edf"), format="edf", bypass_analysis=None)
    assert "Summary skipped as signal analysis was bypassed." in capsys.readouterr().out

    # Forced format, bypass=False -> analysis runs.
    sample_rec.to_edf(str(tmp_path / "b.edf"), format="edf", bypass_analysis=False)
    out = capsys.readouterr().out
    assert "Recommended Format:" in out and "Summary skipped" not in out

    # 'auto' -> analysis runs even when bypass=True is requested (and ignored).
    sample_rec.to_edf(str(tmp_path / "c.edf"), format="auto", bypass_analysis=True)
    assert "Summary skipped" not in capsys.readouterr().out

    # Forced BDF mirrors forced EDF: bypass by default, analyze when bypass=False.
    sample_rec.to_edf(str(tmp_path / "d.bdf"), format="bdf", bypass_analysis=None)
    assert "Summary skipped as signal analysis was bypassed." in capsys.readouterr().out
    sample_rec.to_edf(str(tmp_path / "e.bdf"), format="bdf", bypass_analysis=False)
    assert "Recommended Format:" in capsys.readouterr().out


def test_to_edf_external_events_are_written_and_object_untouched(sample_rec, tmp_path):
    """The external events_df (not self.events) is what reaches the file, and the
    Recording object's own events are left intact.

    Forwarding is verified by reading the EDF+ annotations back with pyedflib
    directly (biosigio's own annotation read-back is pending #47).
    """
    sample_rec.add_event(onset=0.1, duration=0.0, description="Marker 1")
    sample_rec.add_event(onset=0.5, duration=0.2, description="Activity")
    external = pd.DataFrame([{"onset": 0.3, "duration": 0.1, "description": "External"}])
    out = tmp_path / "ev.edf"
    sample_rec.to_edf(str(out), format="edf", bypass_analysis=True, events_df=external)

    assert len(sample_rec.events) == 2  # self.events untouched

    with pyedflib.EdfReader(str(out)) as reader:
        descriptions = list(reader.readAnnotations()[2])
    assert "External" in descriptions
    assert "Marker 1" not in descriptions and "Activity" not in descriptions


def test_to_edf_empty_raises(empty_rec, tmp_path):
    """Exporting a Recording object with no signals raises ValueError."""
    with pytest.raises(ValueError):
        empty_rec.to_edf(str(tmp_path / "test.edf"))


def test_add_channel_with_prefilter(empty_rec):
    """Test adding channel with prefilter specification."""
    data = np.array([1, 2, 3])
    prefilter = "HP 20Hz"
    empty_rec.add_channel("EMG1", data, 1000, "mV", "EMG", prefilter=prefilter)

    assert empty_rec.channels["EMG1"]["prefilter"] == prefilter


def test_select_channels_none_with_type(sample_rec):
    """Test selecting all channels of a type when channels=None."""
    # Add another EMG channel for testing
    data = np.linspace(0, 1, 1000)
    sample_rec.add_channel("EMG2", data, 1000, "mV", channel_type="EMG")
    original_channels = list(sample_rec.signals.columns)

    # Select all EMG channels
    result = sample_rec.select_channels(channels=None, channel_type="EMG")
    assert set(result.signals.columns) == {"EMG1", "EMG2"}
    assert all(info["channel_type"] == "EMG" for info in result.channels.values())
    # Verify original object is unchanged
    assert list(sample_rec.signals.columns) == original_channels


def test_plot_single_axis(sample_rec, mock_plt):
    """Test plotting only makes a single axis."""
    # add the same channel data to test multiple channels
    sample_rec.add_channel("EMG2", sample_rec.signals["EMG1"], 1000, "mV", channel_type="EMG")
    sample_rec.plot_signals(channels=["EMG1", "EMG2"], show=False, plt_module=mock_plt)

    # Verify axis is one
    assert len(mock_plt.axes) == 1


def test_select_channels_empty_result(sample_rec):
    """Test selecting channels with type filter resulting in empty selection."""
    with pytest.raises(ValueError) as exc_info:
        sample_rec.select_channels(["EMG1"], channel_type="GYRO")
    assert "None of the selected channels are of type" in str(exc_info.value)


def test_add_multiple_channels(empty_rec):
    """Test adding multiple channels with different properties."""
    # Add first channel
    data1 = np.array([1, 2, 3])
    empty_rec.add_channel("CH1", data1, 1000, "mV", channel_type="EMG")

    # Add second channel with different properties
    data2 = np.array([4, 5, 6])
    empty_rec.add_channel("CH2", data2, 2000, "g", channel_type="ACC")

    # Verify both channels exist with correct properties
    assert set(empty_rec.signals.columns) == {"CH1", "CH2"}
    assert empty_rec.channels["CH1"]["sample_frequency"] == 1000
    assert empty_rec.channels["CH2"]["sample_frequency"] == 2000
    assert empty_rec.channels["CH1"]["channel_type"] == "EMG"
    assert empty_rec.channels["CH2"]["channel_type"] == "ACC"
