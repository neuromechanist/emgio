# Signal Verification Examples

This page demonstrates how to use biosigIO's verification capabilities to ensure signal integrity when transferring biosignal data between formats.

## Basic Verification Workflow

```python
import os
from biosigio import Recording
import matplotlib.pyplot as plt
from biosigio.analysis.verification import compare_signals, report_verification_results

# Load original data from a device-specific format
emg_original = Recording.from_file('sample_data.csv', importer='trigno')
# just keep the EMG channels
emg_channels = [ch for ch, info in emg_original.channels.items() if info['channel_type'] == 'EMG']
emg_original = emg_original.select_channels(emg_channels)  # Creates a new Recording object with only EMG channels

# Export to EDF (automatically selects EDF or BDF based on signal characteristics).
# Pass a base path without an extension; the writer appends .edf or .bdf.
emg_original.to_edf('exported_data')

# Reload the exported data. Resolve whichever extension to_edf selected.
exported_path = 'exported_data.edf' if os.path.exists('exported_data.edf') else 'exported_data.bdf'
emg_reloaded = Recording.from_file(exported_path)

# Verify signals using the verification module
results = compare_signals(emg_original, emg_reloaded, tolerance=0.01)
is_identical = report_verification_results(results, verify_tolerance=0.01)

print(f"Verification result: {'Passed' if is_identical else 'Failed'}")
```

## Visual Comparison

The `plot_comparison()` function provides a visual way to compare signals:

```python
from biosigio.visualization.static import plot_comparison

# Visual comparison of original and reloaded signals
plot_comparison(emg_original, emg_reloaded, channels=['EMG1', 'EMG2'])
plt.show()

# Customize the comparison
plot_comparison(
    emg_original,
    emg_reloaded,
    channels=['EMG1', 'EMG2'],
    time_range=(0, 5),  # Only show first 5 seconds
    detrend=True,       # Remove mean for easier comparison
    grid=True           # Add grid lines
)
plt.show()
```

## Detailed Verification Analysis

For more control and detailed analysis, you can use the verification module directly:

```python
from biosigio.analysis.verification import compare_signals, report_verification_results

# Compare with detailed metrics
results = compare_signals(
    emg_original,
    emg_reloaded,
    tolerance=0.01,  # Set custom tolerance (1%)
    channel_map=None  # Use automatic channel matching
)

# Generate detailed report
is_identical = report_verification_results(results, verify_tolerance=0.01)

# Inspect results for specific channels
print("\nDetailed metrics per channel:")
for channel, metrics in results.items():
    if channel != 'channel_summary':
        print(f"Channel: {channel}")
        print(f"  - NRMSE: {metrics['nrmse']:.6f}")
        print(f"  - Max Normalized Difference: {metrics['max_norm_abs_diff']:.6f}")
        print(f"  - Identical within tolerance: {metrics['is_identical']}")
```

## Handling Channel Mapping

When channels are renamed during export/import:

```python
from biosigio.visualization.static import plot_comparison

# Define explicit channel mapping
channel_map = {
    'EMG_biceps': 'CH1',
    'EMG_triceps': 'CH2',
    'EMG_forearm': 'CH3'
}

# Compare with channel mapping
results = compare_signals(
    emg_original,
    emg_reloaded,
    tolerance=0.01,
    channel_map=channel_map
)

# Generate report with the mapping applied
is_identical = report_verification_results(results, verify_tolerance=0.01)

# Visual comparison with channel mapping
plot_comparison(
    emg_original,
    emg_reloaded,
    channel_map=channel_map,
    time_range=(0, 5)
)
plt.show()
```

## Real-World Example: CSV to EDF Conversion

```python
import os
from biosigio import Recording
import matplotlib.pyplot as plt
from biosigio.analysis.verification import compare_signals, report_verification_results
from biosigio.visualization.static import plot_comparison

# 1. Load original Trigno CSV data
emg_trigno = Recording.from_file('trigno_data.csv', importer='trigno')

# 2. Export to EDF (the writer appends .edf or .bdf based on signal analysis)
emg_trigno.to_edf('converted_data')
print(f"Data exported. Duration: {emg_trigno.get_duration():.2f}s, "
      f"Channels: {emg_trigno.get_n_channels()}")

# 3. Reload from the exported file (resolve the auto-selected extension)
converted_path = 'converted_data.edf' if os.path.exists('converted_data.edf') else 'converted_data.bdf'
emg_edf = Recording.from_file(converted_path)
print(f"Data reloaded. Duration: {emg_edf.get_duration():.2f}s, "
      f"Channels: {emg_edf.get_n_channels()}")

# 4. Verification using the analysis module
results = compare_signals(emg_trigno, emg_edf, tolerance=0.01)
is_identical = report_verification_results(results, verify_tolerance=0.01)
print(f"Verification result: {'Passed' if is_identical else 'Failed'}")

# 5. Visual verification of a few channels
plt.figure(figsize=(14, 8))
plot_comparison(
    emg_trigno,
    emg_edf,
    channels=list(emg_trigno.signals.columns)[:3],  # First 3 channels
    time_range=(0, 2),  # First 2 seconds
    detrend=True
)
plt.suptitle("Trigno CSV to EDF Conversion Verification", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
``` 