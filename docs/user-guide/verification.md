# Signal Verification

EMGIO provides tools to verify signal integrity when transferring data between formats. This is particularly useful when:

- Exporting EMG data to formats like EDF/BDF
- Sharing data with collaborators who use different software
- Creating archival copies of your data
- Testing new importers or exporters

## Verification Workflow

The standard way to verify signal integrity involves comparing an original Recording object with one that has been exported and then reloaded. This is done using the `compare_signals` and `report_verification_results` functions from the `emgio.analysis.verification` module.

```python
from emgio import Recording
from emgio.analysis.verification import compare_signals, report_verification_results

# Load original data
emg_original = Recording.from_file('raw_data.csv', importer='trigno')

# Export to EDF and reload
emg_original.to_edf('exported_data')
emg_reloaded = Recording.from_file('exported_data.edf')

# Compare signals with detailed metrics
results = compare_signals(emg_original, emg_reloaded, tolerance=0.01)

# Generate a detailed report (logs to the console)
is_identical = report_verification_results(results, verify_tolerance=0.01)

# Check the result
if is_identical:
    print("Verification passed: Signals are identical within tolerance")
else:
    print("Verification failed: Signal differences detected")

# Inspect results for specific channels
print("\nDetailed metrics per channel:")
for channel, metrics in results.items():
    if channel != 'channel_summary':
        print(f"  Channel {channel} - NRMSE: {metrics['nrmse']:.4f}")

```

## Visual Verification

You can visually compare original and reloaded signals using the `plot_comparison()` function from the `emgio.visualization.static` module:

```python
from emgio.visualization.static import plot_comparison
import matplotlib.pyplot as plt

# Plot comparison of original and reloaded signals
plot_comparison(emg_original, emg_reloaded)
plt.show()

# Plot specific channels only
plot_comparison(emg_original, emg_reloaded, channels=['EMG1', 'EMG2'])
plt.show()

# Customize the comparison
plot_comparison(
    emg_original,
    emg_reloaded,
    channels=['EMG1', 'EMG2'],
    time_range=(0, 5),  # First 5 seconds only
    detrend=True,       # Remove DC offset
    grid=True           # Add grid lines
)
plt.show()
```

## Advanced Verification

For more detailed verification, you can use the functions in the `verification` module directly:

```python
from emgio.analysis.verification import compare_signals, report_verification_results

# Compare signals with detailed metrics
results = compare_signals(emg_original, emg_reloaded, tolerance=0.01)

# Generate a detailed report (logs to the console)
is_identical = report_verification_results(results, verify_tolerance=0.01)

# Inspect results for specific channels
for channel, metrics in results.items():
    if channel != 'channel_summary':
        print(f"Channel {channel} - NRMSE: {metrics['nrmse']:.4f}")
```

## Channel Mapping

When channels might be renamed during the export/import process, you can provide a mapping:

```python
from emgio.analysis.verification import compare_signals, report_verification_results

# Define mapping from original channel names to reloaded channel names
channel_map = {
    'EMG_biceps': 'CH1',
    'EMG_triceps': 'CH2'
}

# Compare with channel mapping
results = compare_signals(emg_original, emg_reloaded, channel_map=channel_map)
report_verification_results(results, verify_tolerance=0.01)

# Visual comparison with channel mapping
from emgio.visualization.static import plot_comparison
import matplotlib.pyplot as plt
plot_comparison(emg_original, emg_reloaded, channel_map=channel_map)
plt.show()
```

## Understanding Verification Metrics

EMGIO uses the following metrics for signal verification:

- **Normalized RMSE (NRMSE)**: Root mean square error normalized by the peak-to-peak range of the original signal
- **Maximum Normalized Absolute Difference**: The maximum absolute difference between signals, normalized by the peak-to-peak range

A signal is considered identical if both metrics are below the tolerance threshold (default: 0.01 or 1%).

## Common Issues

If verification fails, check for:

1. **Different sampling rates**: Ensure both signals have the same sampling rate
2. **Precision loss**: Some formats (like EDF) have limited precision
3. **Filtering effects**: Some importers/exporters might apply filters
4. **Channel mapping**: Verify channels are correctly mapped
5. **Length differences**: Signals should have the same number of samples 