# Signal Verification Examples

This page demonstrates how to use EMGIO's verification capabilities to ensure signal integrity when transferring EMG data between formats.

## Basic Verification Workflow

```python
from emgio import EMG
import matplotlib.pyplot as plt

# Load original data from a device-specific format
emg_original = EMG.from_file('sample_data.csv', importer='trigno')
# just keep the EMG channels
emg_channels = [ch for ch, info in emg_original.channels.items() if info['channel_type'] == 'EMG']
emg_original = emg_original.select_channels(emg_channels)  # Creates a new EMG object with only EMG channels

# Export to EDF (automatically selects EDF or BDF based on signal characteristics)
emg_original.to_edf('exported_data')

# Reload the exported data
emg_reloaded = EMG.from_file('exported_data.edf')

# Verify signals match within tolerance
result = emg_original.verify_against(emg_reloaded)
print(f"Verification result: {'Passed' if result else 'Failed'}")
```

## Visual Comparison

The `plot_comparison()` method provides a visual way to compare signals:

```python
# Visual comparison of original and reloaded signals
emg_original.plot_comparison(emg_reloaded, channels=['EMG1', 'EMG2'])
plt.show()

# Customize the comparison
emg_original.plot_comparison(
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
from emgio.analysis.verification import compare_signals, report_verification_results

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
emg_original.plot_comparison(
    emg_reloaded,
    channel_map=channel_map,
    time_range=(0, 5)
)
plt.show()
```

## Real-World Example: CSV to EDF Conversion

```python
from emgio import EMG
import matplotlib.pyplot as plt
from emgio.analysis.verification import compare_signals, report_verification_results

# 1. Load original Trigno CSV data
emg_trigno = EMG.from_file('trigno_data.csv', importer='trigno')

# 2. Export to EDF
emg_trigno.to_edf('converted_data')
print(f"Data exported. Duration: {emg_trigno.get_duration():.2f}s, "
      f"Channels: {emg_trigno.get_n_channels()}")

# 3. Reload from EDF
emg_edf = EMG.from_file('converted_data.edf')
print(f"Data reloaded. Duration: {emg_edf.get_duration():.2f}s, "
      f"Channels: {emg_edf.get_n_channels()}")

# 4. Basic verification
result = emg_trigno.verify_against(emg_edf)
print(f"Verification result: {'Passed' if result else 'Failed'}")

# 5. Detailed verification
results = compare_signals(emg_trigno, emg_edf, tolerance=0.01)
report_verification_results(results, verify_tolerance=0.01)

# 6. Visual verification of a few channels
plt.figure(figsize=(14, 8))
emg_trigno.plot_comparison(
    emg_edf,
    channels=list(emg_trigno.signals.columns)[:3],  # First 3 channels
    time_range=(0, 2),  # First 2 seconds
    detrend=True
)
plt.suptitle("Trigno CSV to EDF Conversion Verification", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
``` 