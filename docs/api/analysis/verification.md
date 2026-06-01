# Verification API

The `verification` module provides functions for verifying signal integrity after operations like export/import. It's particularly useful for ensuring data quality when transferring biosignal data between different formats.

## Module Documentation

::: biosigio.analysis.verification
    options:
      show_root_heading: true
      show_source: true
      members: true

## Key Functions Summary

- `compare_signals()`: Compare signals between two Recording objects using normalized metrics.
- `report_verification_results()`: Generate a detailed report based on verification results.

## Usage Examples

### Comparing Signals

```python
from biosigio import Recording
from biosigio.analysis.verification import compare_signals, report_verification_results

# Load original recording
rec_original = Recording.from_file("data.csv", importer="trigno")

# Export to EDF/BDF and reload. With the default format='auto', to_edf may pick
# either .edf or .bdf, so reload the path it actually wrote.
written_path = "exported_data.edf"  # or "exported_data.bdf" if BDF was selected
rec_original.to_edf("exported_data")
rec_reloaded = Recording.from_file(written_path)

# Compare signals
results = compare_signals(rec_original, rec_reloaded)

# Generate report
is_identical = report_verification_results(results, verify_tolerance=0.01)

# Check results
if is_identical:
    print("Verification passed: Signals are identical within tolerance")
else:
    print("Verification failed: Signal differences detected")
```

### Using Channel Mapping

When channels are renamed or reordered during export/import:

```python
# Define channel mapping from original to reloaded
channel_map = {
    'EMG1': 'CH1',  # Original channel 'EMG1' maps to reloaded channel 'CH1'
    'EMG2': 'CH2',
    'EMG3': 'CH3'
}

# Compare with mapping
results = compare_signals(rec_original, rec_reloaded, channel_map=channel_map)
is_identical = report_verification_results(results, verify_tolerance=0.01)
``` 