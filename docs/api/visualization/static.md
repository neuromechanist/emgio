# Static Visualization API

The `static` module in `emgio.visualization` provides functions for static plotting of EMG data. These functions are primarily used internally by the `EMG` class methods but can also be called directly for advanced customization.

## Module Documentation

::: emgio.visualization.static
    options:
      show_root_heading: true
      show_source: true
      members: true

## Key Functions Summary

- `plot_signals()`: Plot EMG signals in a single figure with vertical offsets.
- `plot_comparison()`: Plot original and reloaded signals overlaid for visual comparison.

## Direct Usage Examples

While these functions are typically accessed through the `EMG` class methods, they can be called directly for advanced use cases:

### Plotting Signals Directly

```python
from emgio import EMG
from emgio.visualization.static import plot_signals

# Load EMG data
emg = EMG.from_file("data.csv", importer="trigno")

# Plot signals directly with custom parameters
plot_signals(
    emg_object=emg,
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
from emgio import EMG
from emgio.visualization.static import plot_comparison

# Load original and reloaded EMG data
emg_original = EMG.from_file("original.csv", importer="trigno")
emg_reloaded = EMG.from_file("reloaded.edf")

# Create channel mapping
channel_map = {
    'EMG1': 'Channel_1',
    'EMG2': 'Channel_2'
}

# Plot comparison directly
plot_comparison(
    emg_original=emg_original,
    emg_reloaded=emg_reloaded,
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

For most use cases, the corresponding methods on the `EMG` class (`plot_signals()` and `plot_comparison()`) are recommended as they provide simplified interfaces to these functions. 