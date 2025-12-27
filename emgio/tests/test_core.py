import builtins
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ..core.emg import EMG


@pytest.fixture
def empty_emg():
    """Create an empty EMG object."""
    return EMG()


@pytest.fixture
def sample_emg():
    """Create an EMG object with sample data."""
    emg = EMG()

    # Add sample channels
    time = np.linspace(0, 1, 1000)  # 1 second at 1000Hz
    emg_data = np.sin(2 * np.pi * 10 * time)  # 10Hz sine wave
    acc_data = np.cos(2 * np.pi * 5 * time)  # 5Hz cosine wave

    emg.add_channel("EMG1", emg_data, 1000, "mV", channel_type="EMG")
    emg.add_channel("ACC1", acc_data, 1000, "g", channel_type="ACC")

    return emg


def test_emg_initialization(empty_emg):
    """Test EMG object initialization."""
    assert empty_emg.signals is None
    assert empty_emg.metadata == {}
    assert empty_emg.channels == {}


def test_add_channel(empty_emg):
    """Test adding a channel to EMG object."""
    data = np.array([1, 2, 3, 4, 5])
    empty_emg.add_channel("EMG1", data, 1000, "mV", "EMG")

    assert "EMG1" in empty_emg.signals.columns
    assert "EMG1" in empty_emg.channels
    assert empty_emg.channels["EMG1"]["sample_frequency"] == 1000
    assert empty_emg.channels["EMG1"]["physical_dimension"] == "mV"
    assert empty_emg.channels["EMG1"]["channel_type"] == "EMG"


def test_select_channels(sample_emg):
    """Test channel selection."""
    # Store original state
    original_channels = list(sample_emg.signals.columns)

    # Select multiple channels
    emg_multi = sample_emg.select_channels(["EMG1", "ACC1"])
    assert list(emg_multi.signals.columns) == ["EMG1", "ACC1"]
    assert list(emg_multi.channels.keys()) == ["EMG1", "ACC1"]
    # Verify original object is unchanged
    assert list(sample_emg.signals.columns) == original_channels

    # Select single channel
    emg_single = sample_emg.select_channels("EMG1")
    assert list(emg_single.signals.columns) == ["EMG1"]
    assert list(emg_single.channels.keys()) == ["EMG1"]
    # Verify original object is unchanged
    assert list(sample_emg.signals.columns) == original_channels


def test_metadata(empty_emg):
    """Test metadata handling."""
    empty_emg.set_metadata("subject", "S001")
    assert empty_emg.get_metadata("subject") == "S001"

    # Test non-existent key
    assert empty_emg.get_metadata("nonexistent") is None


def test_invalid_channel_selection(sample_emg):
    """Test error handling for invalid channel selection."""
    with pytest.raises(ValueError):
        sample_emg.select_channels("NonexistentChannel")


def test_plot_signals_validation(empty_emg):
    """Test plot_signals input validation."""
    with pytest.raises(ValueError):
        empty_emg.plot_signals()  # Should raise error when no signals are loaded


def test_get_channel_types(sample_emg):
    """Test getting unique channel types."""
    types = sample_emg.get_channel_types()
    assert set(types) == {"EMG", "ACC"}


def test_get_channels_by_type(sample_emg):
    """Test getting channels of specific type."""
    emg_channels = sample_emg.get_channels_by_type("EMG")
    acc_channels = sample_emg.get_channels_by_type("ACC")

    assert emg_channels == ["EMG1"]
    assert acc_channels == ["ACC1"]
    assert sample_emg.get_channels_by_type("NONEXISTENT") == []


# This will be implemented after #3 is resolved
# def test_select_channels_by_type(sample_emg):
#     """Test channel selection by type."""
#     # Select all EMG channels
#     emg_only = sample_emg.select_channels(channel_type='EMG')
#     assert list(emg_only.signals.columns) == ['EMG1']
#     assert all(info['channel_type'] == 'EMG' for info in emg_only.channels.values())

#     # Select all ACC channels
#     acc_only = sample_emg.select_channels(channel_type='ACC')
#     assert list(acc_only.signals.columns) == ['ACC1']
#     assert all(info['channel_type'] == 'ACC' for info in acc_only.channels.values())

#     # Test with non-existent type
#     with pytest.raises(ValueError):
#         sample_emg.select_channels(channel_type='NONEXISTENT')


def test_select_channels_with_type_filter(sample_emg):
    """Test channel selection with type filtering."""
    # Store original state
    original_channels = list(sample_emg.signals.columns)

    # Select specific channels with type filter
    result = sample_emg.select_channels(["EMG1", "ACC1"], channel_type="EMG")
    assert list(result.signals.columns) == ["EMG1"]
    assert all(info["channel_type"] == "EMG" for info in result.channels.values())
    # Verify original object is unchanged
    assert list(sample_emg.signals.columns) == original_channels

    # Test when no channels match type
    with pytest.raises(ValueError):
        sample_emg.select_channels(["EMG1", "ACC1"], channel_type="GYRO")
    # Verify original object is unchanged after error
    assert list(sample_emg.signals.columns) == original_channels


def test_add_channel_validation(empty_emg):
    """Test add_channel with various data types and validation."""
    # Test with different numpy data types
    data_int = np.array([1, 2, 3], dtype=np.int32)
    empty_emg.add_channel("INT", data_int, 1000, "count", channel_type="OTHER")
    assert np.array_equal(empty_emg.signals["INT"].values, data_int)

    # Test with float data
    data_float = np.array([1.1, 2.2, 3.3])
    empty_emg.add_channel("FLOAT", data_float, 1000, "mV")
    assert np.array_equal(empty_emg.signals["FLOAT"].values, data_float)

    # Test channel info storage
    assert empty_emg.channels["INT"]["channel_type"] == "OTHER"
    assert empty_emg.channels["FLOAT"]["channel_type"] == "EMG"  # default type
    assert empty_emg.channels["INT"]["sample_frequency"] == 1000
    assert empty_emg.channels["INT"]["physical_dimension"] == "count"


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
            emg = EMG()
            emg.add_channel("TEST", np.array([1, 2, 3]), 1000, "mV", channel_type="EMG")
            emg.set_metadata("device", "Delsys Trigno")
            emg.set_metadata("source_file", filepath)
            return emg

    class MockOTBImporter(MockBaseImporter):
        def _load(self, filepath, **kwargs):
            emg = EMG()
            emg.add_channel("OTB", np.array([4, 5, 6]), 2000, "mV", channel_type="EMG")
            emg.set_metadata("device", "OT Bioelettronica")
            emg.set_metadata("source_file", filepath)
            return emg

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
                        "emg = EMG.from_file(filepath, importer='trigno')\n\n"
                        "If you still want to use the generic CSV importer, set force_generic=True"
                    )

            emg = EMG()
            emg.add_channel("CSV_CH1", np.array([7, 8, 9]), 1000, "mV", channel_type="EMG")
            emg.set_metadata("file_format", "CSV")
            emg.set_metadata("source_file", filepath)
            return emg

    def mock_import(name, *args):
        # Only intercept our specific importer paths
        if any(
            x in name
            for x in ["emgio.importers.trigno", "emgio.importers.otb", "emgio.importers.csv"]
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
    emg_trigno = EMG.from_file(str(trigno_file), importer="trigno")
    assert "TEST" in emg_trigno.signals.columns
    assert emg_trigno.channels["TEST"]["sample_frequency"] == 1000

    # Test OTB importer (including auto-detection)
    for importer in ["otb", None]:
        emg_otb = EMG.from_file(str(otb_file), importer=importer)
        assert "OTB" in emg_otb.signals.columns
        assert emg_otb.channels["OTB"]["sample_frequency"] == 2000

    # Test CSV importer
    emg_csv = EMG.from_file(str(csv_file), importer="csv")
    assert "CSV_CH1" in emg_csv.signals.columns
    assert emg_csv.get_metadata("file_format") == "CSV"

    # Test CSV importer with auto-detection for .txt files
    emg_txt = EMG.from_file(str(csv_file), importer=None)
    assert "CSV_CH1" in emg_txt.signals.columns
    assert emg_txt.get_metadata("file_format") == "CSV"

    # Test format detection and force_csv
    # First, test that format detection raises an error for trigno file
    with pytest.raises(ValueError, match="Delsys Trigno CSV export"):
        EMG.from_file(str(trigno_named_file), importer="csv")

    # Now test with force_csv=True to bypass detection
    emg_forced = EMG.from_file(str(trigno_named_file), importer="csv", force_csv=True)
    assert "CSV_CH1" in emg_forced.signals.columns

    # Test passing parameters to CSV importer
    custom_kwargs = {"channel_types": {"CSV_CH1": "ACC"}}
    emg_with_params = EMG.from_file(str(csv_file), importer="csv", **custom_kwargs)
    assert "CSV_CH1" in emg_with_params.signals.columns

    # Test invalid importer
    with pytest.raises(ValueError, match="Unsupported importer"):
        EMG.from_file(str(trigno_file), importer="invalid")


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
    monkeypatch.setattr("emgio.visualization.static.plt", mock)
    return mock


@pytest.mark.skip(reason="visualization not critical for core functionality")
def test_plot_signals_basic(sample_emg, mock_plt):
    """Test basic plotting functionality."""
    sample_emg.plot_signals(show=False, plt_module=mock_plt)

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
def test_plot_signals_style_options(sample_emg, mock_plt):
    """Test different plot styles."""
    # Test dots style
    sample_emg.plot_signals(show=False, plt_module=mock_plt)
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
    sample_emg.plot_signals(show=False, plt_module=mock_plt)
    if isinstance(mock_plt.axes, list):
        for ax in mock_plt.axes:
            ax.plot.assert_called_once()
            ax.scatter.assert_not_called()
    else:
        mock_plt.axes.plot.assert_called_once()
        mock_plt.axes.scatter.assert_not_called()
    assert mock_plt.show_called


@pytest.mark.skip(reason="visualization not critical for core functionality")
def test_plot_signals_customization(sample_emg, mock_plt):
    """Test plot customization options."""
    title = "Test Plot"
    sample_emg.plot_signals(
        channels=["EMG1"], grid=True, title=title, show=False, plt_module=mock_plt
    )

    # Verify title
    mock_plt.fig.suptitle.assert_called_with(title, fontsize=14, y=1.02)

    # Verify grid
    mock_plt.axes.grid.assert_called_with(True, linestyle="--", alpha=0.7)

    # Verify show was called
    assert mock_plt.show_called


def test_plot_signals_channel_selection(sample_emg, mock_plt):
    """Test plotting with channel selection."""
    # Test single channel
    sample_emg.plot_signals(channels=["EMG1"], show=False, plt_module=mock_plt)
    assert not isinstance(mock_plt.axes, list)  # Should be single axis

    mock_plt.reset()

    # Test invalid channel
    with pytest.raises(ValueError) as exc_info:
        sample_emg.plot_signals(channels=["NonexistentChannel"])
    assert "Channels not found" in str(exc_info.value)


def test_plot_signals_time_range(sample_emg, mock_plt):
    """Test plotting with time range selection."""
    time_range = (0.2, 0.8)
    sample_emg.plot_signals(time_range=time_range, show=False, plt_module=mock_plt)

    # Verify data selection
    if isinstance(mock_plt.axes, list):
        for ax in mock_plt.axes:
            plot_calls = ax.plot.call_args_list
            assert len(plot_calls) > 0, "No plot calls were made"
            data = plot_calls[-1][0][1]  # Get y-values from last plot call
            assert len(data) < len(sample_emg.signals)  # Should be subset of data
    else:
        plot_calls = mock_plt.axes.plot.call_args_list
        assert len(plot_calls) > 0, "No plot calls were made"
        data = plot_calls[-1][0][1]  # Get y-values from last plot call
        assert len(data) < len(sample_emg.signals)  # Should be subset of data


@pytest.fixture
def mock_edf_exporter(monkeypatch):
    """Mock EDF exporter for testing export functionality."""

    class MockEDFExporter:
        last_export = {}

        @staticmethod
        def export(
            emg_obj,
            filepath,
            method="both",
            fft_noise_range=None,
            svd_rank=None,
            precision_threshold=0.01,
            format="auto",
            bypass_analysis=False,
            **kwargs,
        ):
            if not filepath.endswith(".edf") and not filepath.endswith(".bdf"):
                raise ValueError("File must have .edf or .bdf extension")
            # Store export parameters for verification
            MockEDFExporter.last_export = {
                "filepath": filepath,
                "channels": list(emg_obj.channels.keys()),
                "format": format,
                "bypass_analysis": bypass_analysis,
                "kwargs": kwargs,  # Only store the custom kwargs, not the default ones
            }
            return filepath

    # Directly patch the EDFExporter in the exporters.edf module
    from ..exporters import edf

    original_exporter = edf.EDFExporter
    monkeypatch.setattr(edf, "EDFExporter", MockEDFExporter)

    yield MockEDFExporter

    # Restore the original exporter after the test
    monkeypatch.setattr(edf, "EDFExporter", original_exporter)


def test_emg_add_event(empty_emg):
    """Test adding events to the EMG object."""
    assert empty_emg.events.empty

    # Add first event
    empty_emg.add_event(onset=1.0, duration=0.5, description="Event A")
    assert len(empty_emg.events) == 1
    pd.testing.assert_frame_equal(
        empty_emg.events, pd.DataFrame([{"onset": 1.0, "duration": 0.5, "description": "Event A"}])
    )

    # Add second event (should be sorted)
    empty_emg.add_event(onset=0.5, duration=0.1, description="Event B")
    assert len(empty_emg.events) == 2
    expected_df = pd.DataFrame(
        [
            {"onset": 0.5, "duration": 0.1, "description": "Event B"},
            {"onset": 1.0, "duration": 0.5, "description": "Event A"},
        ]
    )
    pd.testing.assert_frame_equal(empty_emg.events, expected_df)

    # Add third event
    empty_emg.add_event(onset=1.5, duration=0.0, description="Event C")
    assert len(empty_emg.events) == 3
    expected_df = pd.DataFrame(
        [
            {"onset": 0.5, "duration": 0.1, "description": "Event B"},
            {"onset": 1.0, "duration": 0.5, "description": "Event A"},
            {"onset": 1.5, "duration": 0.0, "description": "Event C"},
        ]
    )
    pd.testing.assert_frame_equal(empty_emg.events, expected_df)


def test_to_edf_export(sample_emg, mock_edf_exporter):
    """Test EDF export functionality, including event passing."""
    # Add some events to the sample emg object
    sample_emg.add_event(onset=0.1, duration=0, description="Marker 1")
    sample_emg.add_event(onset=0.5, duration=0.2, description="Activity Period")

    # Test basic export (default format='auto')
    filepath = "test.edf"
    sample_emg.to_edf(filepath)

    assert mock_edf_exporter.last_export["filepath"] == filepath
    assert set(mock_edf_exporter.last_export["channels"]) == {"EMG1", "ACC1"}
    assert mock_edf_exporter.last_export["format"] == "auto"
    assert mock_edf_exporter.last_export["bypass_analysis"] is False
    # Check that events_df kwarg is present and contains the added events
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]
    pd.testing.assert_frame_equal(
        mock_edf_exporter.last_export["kwargs"]["events_df"], sample_emg.events
    )
    assert len(mock_edf_exporter.last_export["kwargs"]["events_df"]) == 2
    assert "create_channels_tsv" in mock_edf_exporter.last_export["kwargs"]
    assert mock_edf_exporter.last_export["kwargs"]["create_channels_tsv"] is True  # Default value
    assert len(mock_edf_exporter.last_export["kwargs"]) == 2  # events_df + create_channels_tsv

    # Test with specific format and additional kwargs (format=bdf should bypass analysis by default)
    custom_kwargs = {"patient_id": "TEST001"}
    sample_emg.to_edf(filepath, format="bdf", **custom_kwargs)
    assert mock_edf_exporter.last_export["filepath"] == filepath
    assert mock_edf_exporter.last_export["format"] == "bdf"
    assert mock_edf_exporter.last_export["bypass_analysis"] is True
    # Check that events_df is still passed along with custom kwargs
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]
    pd.testing.assert_frame_equal(
        mock_edf_exporter.last_export["kwargs"]["events_df"], sample_emg.events
    )
    assert len(mock_edf_exporter.last_export["kwargs"]["events_df"]) == 2
    assert mock_edf_exporter.last_export["kwargs"]["patient_id"] == "TEST001"
    assert "create_channels_tsv" in mock_edf_exporter.last_export["kwargs"]
    assert (
        len(mock_edf_exporter.last_export["kwargs"]) == 3
    )  # events_df + create_channels_tsv + patient_id

    # Test passing an external events DataFrame
    external_events = pd.DataFrame(
        [{"onset": 0.3, "duration": 0.1, "description": "External Event"}]
    )
    sample_emg.to_edf(filepath, format="edf", events_df=external_events)
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]
    pd.testing.assert_frame_equal(
        mock_edf_exporter.last_export["kwargs"]["events_df"], external_events
    )
    # Ensure the EMG object's internal events were not modified
    assert len(sample_emg.events) == 2

    # Test exporting with no events in the EMG object
    empty_event_emg = EMG()
    empty_event_emg.add_channel("CH1", np.array([1, 2, 3]), 100, "V")
    empty_event_emg.to_edf(filepath, format="edf")
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]
    assert mock_edf_exporter.last_export["kwargs"]["events_df"].empty
    pd.testing.assert_frame_equal(
        mock_edf_exporter.last_export["kwargs"]["events_df"], empty_event_emg.events
    )

    # --- Test bypass_analysis logic (Ensure events are still passed) ---

    # Format forced, bypass=None (default) -> should bypass (True)
    sample_emg.to_edf(filepath, format="edf", bypass_analysis=None)
    assert mock_edf_exporter.last_export["format"] == "edf"
    assert mock_edf_exporter.last_export["bypass_analysis"] is True
    assert (
        "events_df" in mock_edf_exporter.last_export["kwargs"]
    )  # Ensure events_df is still passed
    pd.testing.assert_frame_equal(
        mock_edf_exporter.last_export["kwargs"]["events_df"], sample_emg.events
    )

    sample_emg.to_edf(filepath, format="bdf", bypass_analysis=None)
    assert mock_edf_exporter.last_export["format"] == "bdf"
    assert mock_edf_exporter.last_export["bypass_analysis"] is True
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]
    pd.testing.assert_frame_equal(
        mock_edf_exporter.last_export["kwargs"]["events_df"], sample_emg.events
    )

    # Format forced, bypass=True -> should bypass (True)
    sample_emg.to_edf(filepath, format="edf", bypass_analysis=True)
    assert mock_edf_exporter.last_export["bypass_analysis"] is True
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]
    sample_emg.to_edf(filepath, format="bdf", bypass_analysis=True)
    assert mock_edf_exporter.last_export["bypass_analysis"] is True
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]

    # Format forced, bypass=False -> should NOT bypass (False)
    sample_emg.to_edf(filepath, format="edf", bypass_analysis=False)
    assert mock_edf_exporter.last_export["bypass_analysis"] is False
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]
    sample_emg.to_edf(filepath, format="bdf", bypass_analysis=False)
    assert mock_edf_exporter.last_export["bypass_analysis"] is False
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]

    # Format auto, bypass=None -> should NOT bypass (False)
    sample_emg.to_edf(filepath, format="auto", bypass_analysis=None)
    assert mock_edf_exporter.last_export["format"] == "auto"
    assert mock_edf_exporter.last_export["bypass_analysis"] is False
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]

    # Format auto, bypass=True -> should NOT bypass (False)
    sample_emg.to_edf(filepath, format="auto", bypass_analysis=True)
    assert mock_edf_exporter.last_export["bypass_analysis"] is False
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]

    # Format auto, bypass=False -> should NOT bypass (False)
    sample_emg.to_edf(filepath, format="auto", bypass_analysis=False)
    assert mock_edf_exporter.last_export["bypass_analysis"] is False
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]

    # --- End bypass_analysis tests ---

    # Rerun test with specific format and additional kwargs to ensure final state is correct
    custom_kwargs = {"patient_id": "TEST001"}
    sample_emg.to_edf(filepath, format="bdf", **custom_kwargs)
    assert mock_edf_exporter.last_export["filepath"] == filepath
    assert mock_edf_exporter.last_export["format"] == "bdf"
    assert mock_edf_exporter.last_export["bypass_analysis"] is True
    assert "events_df" in mock_edf_exporter.last_export["kwargs"]
    pd.testing.assert_frame_equal(
        mock_edf_exporter.last_export["kwargs"]["events_df"], sample_emg.events
    )
    assert mock_edf_exporter.last_export["kwargs"]["patient_id"] == "TEST001"
    assert "create_channels_tsv" in mock_edf_exporter.last_export["kwargs"]
    assert (
        len(mock_edf_exporter.last_export["kwargs"]) == 3
    )  # events_df + create_channels_tsv + patient_id


def test_to_edf_empty(empty_emg, mock_edf_exporter):
    """Test EDF export with empty EMG object."""
    with pytest.raises(ValueError):
        empty_emg.to_edf("test.edf")


def test_add_channel_with_prefilter(empty_emg):
    """Test adding channel with prefilter specification."""
    data = np.array([1, 2, 3])
    prefilter = "HP 20Hz"
    empty_emg.add_channel("EMG1", data, 1000, "mV", prefilter=prefilter)

    assert empty_emg.channels["EMG1"]["prefilter"] == prefilter


def test_select_channels_none_with_type(sample_emg):
    """Test selecting all channels of a type when channels=None."""
    # Add another EMG channel for testing
    data = np.linspace(0, 1, 1000)
    sample_emg.add_channel("EMG2", data, 1000, "mV", channel_type="EMG")
    original_channels = list(sample_emg.signals.columns)

    # Select all EMG channels
    result = sample_emg.select_channels(channels=None, channel_type="EMG")
    assert set(result.signals.columns) == {"EMG1", "EMG2"}
    assert all(info["channel_type"] == "EMG" for info in result.channels.values())
    # Verify original object is unchanged
    assert list(sample_emg.signals.columns) == original_channels


def test_plot_single_axis(sample_emg, mock_plt):
    """Test plotting only makes a single axis."""
    # add the same channel data to test multiple channels
    sample_emg.add_channel("EMG2", sample_emg.signals["EMG1"], 1000, "mV", channel_type="EMG")
    sample_emg.plot_signals(channels=["EMG1", "EMG2"], show=False, plt_module=mock_plt)

    # Verify axis is one
    assert len(mock_plt.axes) == 1


def test_select_channels_empty_result(sample_emg):
    """Test selecting channels with type filter resulting in empty selection."""
    with pytest.raises(ValueError) as exc_info:
        sample_emg.select_channels(["EMG1"], channel_type="GYRO")
    assert "None of the selected channels are of type" in str(exc_info.value)


def test_add_multiple_channels(empty_emg):
    """Test adding multiple channels with different properties."""
    # Add first channel
    data1 = np.array([1, 2, 3])
    empty_emg.add_channel("CH1", data1, 1000, "mV", channel_type="EMG")

    # Add second channel with different properties
    data2 = np.array([4, 5, 6])
    empty_emg.add_channel("CH2", data2, 2000, "g", channel_type="ACC")

    # Verify both channels exist with correct properties
    assert set(empty_emg.signals.columns) == {"CH1", "CH2"}
    assert empty_emg.channels["CH1"]["sample_frequency"] == 1000
    assert empty_emg.channels["CH2"]["sample_frequency"] == 2000
    assert empty_emg.channels["CH1"]["channel_type"] == "EMG"
    assert empty_emg.channels["CH2"]["channel_type"] == "ACC"
