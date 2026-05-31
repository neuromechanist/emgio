# EDF/BDF Exporter

The EDF/BDF exporter module in biosigIO provides functionality to export EMG data to EDF (European Data Format) or BDF (BioSemi Data Format) files.

## Module Documentation

::: biosigio.exporters.edf
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording

# Load data
emg = Recording.from_file('data.csv', importer='trigno')

# Export to EDF/BDF with automatic format selection
emg.to_edf('output')  # Will generate output.edf or output.bdf

# Force specific format
emg.to_edf('output_edf', format='edf')  # Forces 16-bit EDF
emg.to_edf('output_bdf', format='bdf')  # Forces 24-bit BDF
```

## Automatic Format Selection

A key feature of biosigIO's exporter is its ability to automatically determine whether to use EDF (16-bit) or BDF (24-bit) format based on the dynamic range of the data:

```python
# Control the analysis method for format selection
emg.to_edf('output', method='svd')  # Use SVD analysis only
emg.to_edf('output', method='fft')  # Use FFT analysis only 
emg.to_edf('output', method='both')  # Use both methods (default)

# Customize SVD parameters
emg.to_edf('output', method='svd', svd_rank=5)  # Manual rank cutoff

# Customize FFT parameters
emg.to_edf('output', 
           method='fft', 
           fft_noise_range=(0.1, 10))  # Manual frequency range for noise floor estimation
```

## Parameters

The `to_edf` method accepts the following parameters:

- **output_path (str)**: Path for the output file (without extension)
- **format (str, optional)**: Specify the format to use ('auto', 'edf', or 'bdf'). Default is 'auto'.
- **method (str, optional)**: Method for format selection ('svd', 'fft', or 'both'). Default is 'both'.
- **svd_rank (int, optional)**: Rank cutoff for SVD analysis. Default is None (automatic).
- **fft_noise_range (tuple, optional)**: Frequency range (min, max) for noise floor estimation in FFT. Default is None (automatic).
- **physical_min (float, optional)**: Physical minimum value. Default is None (automatic).
- **physical_max (float, optional)**: Physical maximum value. Default is None (automatic).
- **overwrite (bool, optional)**: Whether to overwrite existing files. Default is False.
- **additional_info (dict, optional)**: Additional information to include in the EDF header.

## Understanding Format Selection

The exporter uses two complementary approaches to determine the appropriate format:

### 1. SVD Analysis

Singular Value Decomposition (SVD) is used to:
- Estimate the effective dimensionality of the data
- Analyze the distribution of signal energy across components 
- Determine if the precision requirements can be satisfied by 16-bit representation

### 2. FFT Analysis

Fast Fourier Transform (FFT) analysis:
- Examines the frequency domain representation of the data
- Evaluates the noise floor and signal-to-noise ratio
- Helps determine if 16-bit precision is sufficient or if 24-bit is needed

## Output Files

When exporting, biosigIO generates the following files:

1. **Main data file**: Either `.edf` or `.bdf` extension depending on the format selected
2. **Channels metadata file**: A `{output_path}.channels.tsv` file with detailed channel information in BIDS-compatible format

Example channels.tsv file content:
```
name    type    units   sampling_frequency
EMG1    EMG     µV      2000
EMG2    EMG     µV      2000
ACC1    ACC     g       2000
```

## Additional Features

- **Channel scaling**: Signals are automatically scaled to maximize precision
- **Metadata preservation**: Subject, recording, and other metadata are included in the EDF header
- **BIDS compatibility**: The exporter follows BIDS conventions for metadata
- **Multi-channel support**: Handles multiple channel types with appropriate units
- **Different sampling rates**: Can handle channels with different sampling rates 