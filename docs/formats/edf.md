# EDF/BDF Format

EDF (European Data Format) and BDF (BioSemi Data Format) are standard formats for storing multichannel biological and physical signals. biosigIO supports both importing from and exporting to these formats.

## Format Description

### EDF Format

EDF (European Data Format) is a simple and flexible format for exchange and storage of multichannel biological and physical signals:

- 16-bit resolution (range: ±32767)
- Supports varying sampling rates between channels
- Includes metadata in the header
- Supports annotations and events

### BDF Format

BDF (BioSemi Data Format) is an extension of EDF that uses 24-bit resolution instead of 16-bit:

- 24-bit resolution (range: ±8,388,607)
- Higher dynamic range for signals with larger amplitude variations
- Otherwise identical to EDF

### File Structure

Both formats consist of:

1. **Header Record**: Contains metadata about the recording:
   - Version information
   - Patient and recording information
   - Date and time
   - Number of channels
   - Sampling rates
   - Calibration values
   
2. **Data Records**: Contains the actual signal data, organized in blocks.

## biosigIO's Approach to EDF/BDF

biosigIO can both import from and export to EDF/BDF formats. When exporting, biosigIO automatically determines which format to use based on the dynamic range of the data:

- If the signal's dynamic range is within 16-bit resolution (~90dB), it uses EDF
- If the signal requires greater precision, it uses BDF

This determination is made using:
- SVD (Singular Value Decomposition) analysis to determine the effective rank and energy distribution
- FFT (Fast Fourier Transform) analysis to evaluate the noise floor and signal-to-noise ratio

## Importer Implementation

The EDF importer in biosigIO (`biosigio.importers.edf`) uses the `pyedflib` package to:

1. Read the EDF/BDF file header to extract metadata
2. Load the signal data for all channels
3. Convert channel information to biosigIO's format
4. Handle different sampling rates across channels

## Exporter Implementation

The EDF exporter in biosigIO (`biosigio.exporters.edf`) also uses `pyedflib` to:

1. Automatically determine whether to use EDF or BDF based on signal characteristics
2. Generate appropriate header information
3. Scale the signals correctly for storage
4. Create a sidecar channels.tsv file with detailed channel metadata (BIDS-compatible)

## Code Example

```python
from biosigio import Recording

# Import from EDF file
emg = Recording.from_file('input.edf', importer='edf')

# Print basic information
print(f"Number of channels: {emg.get_n_channels()}")
print(f"Sampling frequency: {emg.get_sampling_frequency()} Hz")
print(f"Recording duration: {emg.get_duration()} seconds")

# Export to EDF/BDF (format selected automatically)
emg.to_edf('output')  # Will add .edf or .bdf extension automatically

# Force a specific format
emg.to_edf('output_edf', format='edf')  # Forces 16-bit EDF
emg.to_edf('output_bdf', format='bdf')  # Forces 24-bit BDF

# Export with verification to ensure data integrity 
emg.to_edf('output_verified', verify=True)
```

## Signal Verification

When exporting EMG data to EDF/BDF format, you can optionally verify the integrity of the exported data by setting the `verify` parameter to `True`:

```python
# Export with verification
verification_results = emg.to_edf('output', verify=True)

# Export with custom verification settings
verification_results = emg.to_edf(
    'output',
    verify=True,
    verify_tolerance=0.001,  # 0.1% tolerance
    verify_channel_map={'EMG1': 'CH1'},  # Custom channel mapping
    verify_plot=True  # Generate visualization of comparison
)
```

The verification process:
1. Exports the data to the specified file
2. Immediately reloads the file using the EDF importer
3. Compares the original and reloaded signals using normalization-based metrics
4. Logs a detailed verification report
5. Optionally generates a visual comparison if `verify_plot=True`
6. Returns the verification results as a dictionary

## Notes and Limitations

- The BIDS-compatible channels.tsv sidecar file includes detailed channel information
- Annotations in EDF files are preserved in biosigIO's metadata
- When importing from EDF, biosigIO attempts to identify channel types based on labels and signal characteristics
- When exporting to EDF/BDF, biosigIO automatically handles scaling to maximize precision
- EDF has limitations on channel naming (maximum 16 characters)
