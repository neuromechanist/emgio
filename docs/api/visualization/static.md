# Static Visualization API

The `static` module in `biosigio.visualization` provides functions for static plotting of biosignal data. These functions are primarily used internally by the `Recording` class methods but can also be called directly for advanced customization.

## Module Documentation

::: biosigio.visualization.static
    options:
      show_root_heading: true
      show_source: true
      members: true

## Key Functions Summary

- `plot_signals()`: Plot EMG signals in a single figure with vertical offsets.
- `plot_comparison()`: Plot original and reloaded signals overlaid for visual comparison.

## Direct Usage Examples

While these functions are typically accessed through the `Recording` class methods, they can be called directly for advanced use cases:

### Plotting Signals Directly

```python
from biosigio import Recording
from biosigio.visualization.static import plot_signals

# Load EMG data
rec = Recording.from_file("data.csv", importer="trigno")

# Plot signals directly with custom parameters
plot_signals(
    rec_object=rec,
    channels=['EMG1', 'EMG2'],
    time_range=(0, 5),
    offset_scale=0.7,
    uniform_scale=False,
    detrend=True,
    grid=True,
    title="Custom EMG Plot",
    show=True
)
```

### Plotting Comparison Directly

```python
from biosigio import Recording
from biosigio.visualization.static import plot_comparison

# Load original and reloaded EMG data
rec_original = Recording.from_file("original.csv", importer="trigno")
rec_reloaded = Recording.from_file("reloaded.edf")

# Create channel mapping
channel_map = {
    'EMG1': 'Channel_1',
    'EMG2': 'Channel_2'
}

# Plot comparison directly
plot_comparison(
    rec_original=rec_original,
    rec_reloaded=rec_reloaded,
    channels=['EMG1', 'EMG2'],
    time_range=(1, 3),
    detrend=True,
    grid=True,
    suptitle="Signal Comparison",
    channel_map=channel_map,
    show=True
)
```

## Customizing Plots

The static plotting functions provide several parameters for customization:

- **Channel selection**: Display only specific channels
- **Time range**: Plot a specific time window
- **Detrending**: Remove mean value for better comparison
- **Uniform scaling**: Control whether all signals use the same scale
- **Offset scale**: Control spacing between channels
- **Grid lines**: Toggle grid visibility
- **Titles**: Add custom titles to plots

For most use cases, the `Recording.plot_signals()` method is recommended as it provides a simplified interface that delegates to `plot_signals()` here. Note that `plot_comparison()` is a module-level function only; there is no `Recording.plot_comparison()` method, so call it directly from `biosigio.visualization.static` (as shown above, or automatically via `Recording.to_edf(..., verify=True, verify_plot=True)`).