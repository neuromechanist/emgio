# Basic Usage

biosigIO provides a unified interface for working with EMG data from various systems. This page covers the core functionality that's common across all supported data formats.

## Loading Data

The main entry point for loading data is the `Recording.from_file()` method. This method automatically determines the correct importer based on the file extension, or you can specify the importer explicitly:

```python
from biosigio import Recording

# Automatic importer selection based on file extension
rec = Recording.from_file('data.csv')  # Will use CSV importer for CSV files
rec = Recording.from_file('data.edf')  # Will use EDF importer for EDF files
rec = Recording.from_file('data.set')  # Will use EEGLAB importer for SET files
rec = Recording.from_file('data.otb')  # Will use OTB importer for OTB files
rec = Recording.from_file('data.hea')  # Will use WFDB importer for WFDB files

# Explicit importer selection
rec = Recording.from_file('data.csv', importer='trigno')
rec = Recording.from_file('data.set', importer='eeglab')
rec = Recording.from_file('data.otb+', importer='otb')
rec = Recording.from_file('data.edf', importer='edf')
rec = Recording.from_file('data.csv', importer='csv')  # Generic CSV importer
rec = Recording.from_file('data.hea', importer='wfdb')  # WFDB importer
```

### Automatic File Type Inference

biosigIO automatically infers the appropriate importer based on file extension:

| Extension | Default Importer |
|-----------|------------------|
| `.csv`, `.txt` | `csv` (Generic CSV importer) |
| `.edf`, `.bdf` | `edf` (EDF/BDF importer) |
| `.set` | `eeglab` (EEGLAB importer) |
| `.otb`, `.otb+` | `otb` (OTB importer) |
| `.hea`, `.dat`, `.atr` | `wfdb` (WFDB importer) |

Automatic importer selection works for CSV/TXT files as well; the generic CSV
importer additionally includes specialized format detection that can identify
formats like Trigno CSV exports and route to the appropriate specialized
importer. WFDB datasets without a file extension must be loaded with an explicit
`importer='wfdb'`.

### Special Case: CSV Files

For CSV files with specialized formats like Trigno, the generic CSV importer will detect this and suggest using the appropriate specialized importer. If you still want to use the generic CSV importer, you can use the `force_csv` parameter:

```python
# Force using the generic CSV importer even for specialized formats
rec = Recording.from_file('trigno_data.csv', importer='csv', force_csv=True)
```

See the [Generic CSV Format](../formats/csv.md) documentation for more details on CSV import options.

## Accessing Data

Once you've loaded data, you can access the signals and metadata:

```python
# Access raw signal data (returns a pandas DataFrame)
signals = rec.signals

# Get information about channels
channels = rec.channels

# Get metadata
device = rec.get_metadata('device')
```

## Plotting Signals

biosigIO provides methods to visualize the EMG signals:

```python
# Plot all channels
rec.plot_signals()

# Plot specific channels
rec.plot_signals(['EMG1', 'EMG2'])

# Plot with time range (in seconds)
rec.plot_signals(time_range=(0, 5))

# Customize plot
rec.plot_signals(
    channels=['EMG1', 'EMG2'],
    time_range=(0, 5),
    title='EMG Signals',
    grid=True,
    detrend=False,
    offset_scale=0.8,  # Controls vertical spacing between signals
    uniform_scale=True  # Use same scale for all signals
)
```

## Verifying Signal Integrity

When exporting and reimporting data, you might want to verify that the signals remain intact. biosigIO provides two ways to verify signal integrity:

```python
# Method 1: Integrated verification during export
# Export to EDF with automatic verification
verification_results = rec_original.to_edf('output', verify=True)

# Method 2: Manual verification with more control
from biosigio.analysis.verification import compare_signals, report_verification_results
from biosigio.visualization.static import plot_comparison
import matplotlib.pyplot as plt

# Export to EDF and reload
rec_original.to_edf('output')
rec_reloaded = Recording.from_file('output.edf')

# Compare signals
results = compare_signals(rec_original, rec_reloaded)
is_identical = report_verification_results(results, verify_tolerance=0.01)
print(f"Verification {'passed' if is_identical else 'failed'}")

# Visual verification
plot_comparison(rec_original, rec_reloaded, channels=['EMG1', 'EMG2'])
plt.show()
```

For more detailed information about signal verification, see the [Signal Verification](verification.md) guide.

## Exporting Data

biosigIO can export data to EDF/BDF formats:

```python
# Export to EDF or BDF (format selected automatically)
rec.to_edf('output')  # Extension (.edf/.bdf) will be added automatically

# Force EDF format
rec.to_edf('output', format='edf')

# Force BDF format
rec.to_edf('output', format='bdf')

# Control the analysis method for format selection
rec.to_edf('output', method='svd')  # Use SVD analysis only
rec.to_edf('output', method='fft')  # Use FFT analysis only
rec.to_edf('output', method='both')  # Use both methods (default)
```

## Next Steps

After mastering these basics, you might want to explore:

- [Channel Selection](channel-selection.md) - Learn how to select and manipulate channels
- [EDF/BDF Format Selection](edf-bdf-selection.md) - Understanding how biosigIO selects the appropriate format
- [Metadata Handling](metadata.md) - Working with metadata in biosigIO
- [Signal Verification](verification.md) - Verifying signal integrity after export/import operations
