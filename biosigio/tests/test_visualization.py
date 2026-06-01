from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from biosigio.core.emg import Recording
from biosigio.visualization.static import plot_comparison, plot_signals


@pytest.fixture
def sample_rec():
    """Create a sample Recording object for visualization testing."""
    rec = Recording()

    # Create sample data
    time = np.linspace(0, 1, 1000)  # 1 second at 1000Hz
    signal1 = np.sin(2 * np.pi * 10 * time)  # 10Hz sine wave
    signal2 = np.cos(2 * np.pi * 5 * time)  # 5Hz cosine wave
    signal3 = 0.5 * np.sin(2 * np.pi * 15 * time)  # 15Hz sine wave

    # Add channels
    rec.add_channel("EMG1", signal1, 1000, "mV", channel_type="EMG")
    rec.add_channel("EMG2", signal2, 1000, "mV", channel_type="EMG")
    rec.add_channel("ACC1", signal3, 1000, "g", channel_type="ACC")

    return rec


@pytest.fixture
def rec_pair():
    """Create a pair of Recording objects for comparison testing."""
    rec_original = Recording()
    rec_reloaded = Recording()

    # Create sample data
    time = np.linspace(0, 1, 1000)
    signal1 = np.sin(2 * np.pi * 10 * time)
    signal2 = np.cos(2 * np.pi * 5 * time)

    # Add channels to original
    rec_original.add_channel("EMG1", signal1, 1000, "mV", channel_type="EMG")
    rec_original.add_channel("EMG2", signal2, 1000, "mV", channel_type="EMG")

    # Add slightly modified channels to reloaded
    signal1_modified = signal1 + 0.01 * np.random.randn(len(signal1))
    rec_reloaded.add_channel("EMG1", signal1_modified, 1000, "mV", channel_type="EMG")
    rec_reloaded.add_channel("EMG2", signal2.copy(), 1000, "mV", channel_type="EMG")

    return rec_original, rec_reloaded


class MockPlt:
    """Mock for plt to test plotting without showing plots."""

    def __init__(self):
        self.show_called = False
        self.tight_layout_called = False

    def subplots(self, *args, **kwargs):
        """Mock subplots method that accepts any parameters"""
        # Create mocked figure and axis objects
        mock_fig = MagicMock()

        # For plot_signals, we need a single axis
        if len(args) == 0 or args[0] == 1:
            mock_ax = MagicMock()
            mock_ax.plot = MagicMock()
            mock_ax.set_xlabel = MagicMock()
            mock_ax.set_yticks = MagicMock()
            mock_ax.set_yticklabels = MagicMock()
            mock_ax.set_ylim = MagicMock()
            mock_ax.grid = MagicMock()
            mock_ax.set_title = MagicMock()
            return mock_fig, mock_ax

        # For plot_comparison, we need a 1D array of axes
        else:
            nrows = args[0] if args else kwargs.get("nrows", 1)
            mock_axes = []
            for _ in range(nrows):
                mock_ax = MagicMock()
                mock_ax.plot = MagicMock()
                mock_ax.set_ylabel = MagicMock()
                mock_ax.ticklabel_format = MagicMock()
                mock_ax.grid = MagicMock()
                mock_ax.legend = MagicMock()
                mock_axes.append(mock_ax)

            # Create a custom class that pretends to be a numpy array but can have a flatten method
            class MockAxesArray(list):
                def __init__(self, axes):
                    self.axes = axes

                def __getitem__(self, idx):
                    return self.axes[idx]

                def flatten(self):
                    return self.axes

            return mock_fig, MockAxesArray(mock_axes)

    def show(self):
        self.show_called = True

    def tight_layout(self, rect=None):
        self.tight_layout_called = True


def test_plot_signals_basic(sample_rec):
    """Test basic functionality of plot_signals."""
    mock_plt = MockPlt()
    plot_signals(sample_rec, show=True, plt_module=mock_plt)
    assert mock_plt.show_called


@patch("matplotlib.pyplot.subplots")
@patch("matplotlib.pyplot.show")
def test_plot_signals_parameters(mock_show, mock_subplots, sample_rec):
    """Test plot_signals with various parameters."""
    fig_mock = MagicMock()
    ax_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, ax_mock)

    # Test with specific channels
    plot_signals(sample_rec, channels=["EMG1", "EMG2"], show=True)

    # Verify the plot was created with the correct channels
    assert ax_mock.plot.call_count == 2  # Two channels should be plotted

    # Test with time range
    plot_signals(sample_rec, time_range=(0.1, 0.5), show=True)
    ax_mock.set_xlabel.assert_called_with("Time (s)")


def test_plot_signals_errors(sample_rec):
    """Test error handling in plot_signals."""
    # Test with invalid channel
    with pytest.raises(ValueError):
        plot_signals(sample_rec, channels=["NonExistentChannel"])

    # Test with empty Recording object
    empty_rec = Recording()
    with pytest.raises(ValueError):
        plot_signals(empty_rec)


def test_plot_signals_options(sample_rec):
    """Test various options of plot_signals."""
    mock_plt = MockPlt()

    # Test with grid=False
    plot_signals(sample_rec, grid=False, show=False, plt_module=mock_plt)

    # Test with detrend=True
    plot_signals(sample_rec, detrend=True, show=False, plt_module=mock_plt)

    # Test with uniform_scale=False
    plot_signals(sample_rec, uniform_scale=False, show=False, plt_module=mock_plt)

    # Test with title
    plot_signals(sample_rec, title="Test Plot", show=False, plt_module=mock_plt)

    assert not mock_plt.show_called, "show() should not be called when show=False"
    assert mock_plt.tight_layout_called, "tight_layout() should be called"


def test_plot_signals_with_constant(sample_rec):
    """Test plot_signals with constant signals (zero range)."""
    # Add a constant signal
    sample_rec.add_channel("CONST", np.ones(1000), 1000, "mV", "EMG")

    mock_plt = MockPlt()
    # This should not raise any errors despite the constant signal
    plot_signals(sample_rec, channels=["CONST"], show=False, plt_module=mock_plt)


def test_plot_comparison_basic(rec_pair):
    """Test basic functionality of plot_comparison."""
    rec_original, rec_reloaded = rec_pair
    mock_plt = MockPlt()
    plot_comparison(rec_original, rec_reloaded, show=True, plt_module=mock_plt)
    assert mock_plt.show_called


def test_plot_comparison_parameters(rec_pair):
    """Test plot_comparison with various parameters."""
    rec_original, rec_reloaded = rec_pair
    mock_plt = MockPlt()

    # Test with specific channels
    plot_comparison(rec_original, rec_reloaded, channels=["EMG1"], show=False, plt_module=mock_plt)

    # Test with time range
    plot_comparison(
        rec_original, rec_reloaded, time_range=(0.1, 0.5), show=False, plt_module=mock_plt
    )

    # Test with detrend=True
    plot_comparison(rec_original, rec_reloaded, detrend=True, show=False, plt_module=mock_plt)

    # Test with grid=False
    plot_comparison(rec_original, rec_reloaded, grid=False, show=False, plt_module=mock_plt)

    # Test with custom suptitle
    plot_comparison(
        rec_original, rec_reloaded, suptitle="Custom Title", show=False, plt_module=mock_plt
    )

    # Ensure show was not called
    assert not mock_plt.show_called


def test_plot_comparison_with_channel_map(rec_pair):
    """Test plot_comparison with channel mapping."""
    rec_original, rec_reloaded = rec_pair

    # Create a reloaded Recording with different channel names
    rec_renamed = Recording()
    time = np.linspace(0, 1, 1000)
    signal1 = np.sin(2 * np.pi * 10 * time)
    signal2 = np.cos(2 * np.pi * 5 * time)

    rec_renamed.add_channel("Channel1", signal1, 1000, "mV", "EMG")
    rec_renamed.add_channel("Channel2", signal2, 1000, "mV", "EMG")

    # Define channel mapping
    channel_map = {"EMG1": "Channel1", "EMG2": "Channel2"}

    mock_plt = MockPlt()
    # This should not raise any errors
    plot_comparison(
        rec_original, rec_renamed, channel_map=channel_map, show=False, plt_module=mock_plt
    )


def test_plot_comparison_no_common_channels():
    """Test plot_comparison with no common channels."""
    # Create two Recording objects with completely different channels
    rec1 = Recording()
    rec2 = Recording()

    time = np.linspace(0, 1, 1000)
    signal1 = np.sin(2 * np.pi * 10 * time)
    signal2 = np.cos(2 * np.pi * 5 * time)

    rec1.add_channel("CH1", signal1, 1000, "mV", "EMG")
    rec2.add_channel("CH2", signal2, 1000, "mV", "EMG")

    # This should not raise errors, but will print a warning
    with patch("builtins.print") as mock_print:
        mock_plt = MockPlt()
        plot_comparison(rec1, rec2, show=False, plt_module=mock_plt)
        # Verify warning was printed
        mock_print.assert_called()


def test_plot_comparison_different_lengths():
    """Test plot_comparison with signals of different lengths."""
    rec1 = Recording()
    rec2 = Recording()

    # Create signals with different lengths
    time1 = np.linspace(0, 1, 1000)
    time2 = np.linspace(0, 1, 950)

    signal1 = np.sin(2 * np.pi * 10 * time1)
    signal2 = np.sin(2 * np.pi * 10 * time2)

    rec1.add_channel("CH1", signal1, 1000, "mV", "EMG")
    rec2.add_channel("CH1", signal2, 1000, "mV", "EMG")

    mock_plt = MockPlt()
    # This should handle different length signals
    plot_comparison(rec1, rec2, show=False, plt_module=mock_plt)


def test_plot_comparison_order_based():
    """Test plot_comparison with order-based matching."""
    # Create two Recording objects with different channel names
    rec1 = Recording()
    rec2 = Recording()

    time = np.linspace(0, 1, 1000)
    signal1 = np.sin(2 * np.pi * 10 * time)
    signal2 = np.cos(2 * np.pi * 5 * time)

    rec1.add_channel("CH1", signal1, 1000, "mV", "EMG")
    rec1.add_channel("CH2", signal2, 1000, "mV", "EMG")

    rec2.add_channel("Channel1", signal1, 1000, "mV", "EMG")
    rec2.add_channel("Channel2", signal2, 1000, "mV", "EMG")

    # No common names, should fall back to order-based
    with patch("builtins.print") as mock_print:
        mock_plt = MockPlt()
        plot_comparison(rec1, rec2, show=False, plt_module=mock_plt)

        # Verify that print was called and check if any call contains 'order_based'
        assert mock_print.called
        order_based_message_found = False
        for call_args in mock_print.call_args_list:
            args, _ = call_args
            if args and "order_based" in str(args[0]):
                order_based_message_found = True
                break
        assert order_based_message_found, "Expected to find message containing 'order_based'"
