# EDF/BDF Format Selection

One of the key features of biosigIO is its ability to automatically determine whether to use the EDF (16-bit) or BDF (24-bit) format when exporting data. This decision is based on the dynamic range and precision requirements of your EMG data.

## Why Format Selection Matters

### EDF (European Data Format)
- 16-bit precision (range: ±32767)
- Widely supported by analysis software
- Smaller file size
- Suitable for most EMG recordings

### BDF (BioSemi Data Format)
- 24-bit precision (range: ±8,388,607)
- Required for high-precision recordings
- Larger file size
- Necessary when dynamic range exceeds 16-bit capabilities

Using 16-bit EDF when possible reduces storage requirements while maintaining sufficient precision for most analyses. However, when data contains very small signal components relative to the peak values, higher precision may be necessary to avoid quantization errors and preserve information.

## How biosigIO Selects the Format

biosigIO uses two complementary approaches to determine the appropriate format:

### 1. SVD (Singular Value Decomposition) Analysis

The SVD method decomposes the signal matrix to evaluate its effective rank and the distribution of signal energy:

```python
# Simplified implementation of SVD analysis
U, s, Vh = np.linalg.svd(signals, full_matrices=False)
effective_rank = np.sum(s > threshold)
```

This approach:
- Identifies the essential dimensions in the data
- Measures how much precision is needed to represent these dimensions
- Determines if 16-bit precision (90dB dynamic range) is sufficient

### 2. FFT (Fast Fourier Transform) Analysis

The FFT method examines the frequency domain representation:

```python
# Simplified implementation of FFT analysis
fft_result = np.fft.rfft(signals)
noise_floor = estimate_noise_floor(fft_result)
dynamic_range_db = 20 * np.log10(peak_amplitude / noise_floor)
```

This approach:
- Estimates the noise floor in the frequency domain
- Calculates the dynamic range in decibels
- Compares against the ~90dB limit of 16-bit data

## Controlling Format Selection

You can control the format selection process when exporting:

```python
# Use both methods (default)
emg.to_edf('output', method='both')

# Use only SVD analysis
emg.to_edf('output', method='svd', svd_rank=None)  # Auto threshold
emg.to_edf('output', method='svd', svd_rank=5)     # Manual threshold

# Use only FFT analysis
emg.to_edf('output', method='fft', fft_noise_range=None)    # Auto range
emg.to_edf('output', method='fft', fft_noise_range=(0.1, 10))  # Manual range

# Force a specific format
emg.to_edf('output', format='edf')  # Force EDF (16-bit)
emg.to_edf('output', format='bdf')  # Force BDF (24-bit)
```

## Understanding the Output

After export, biosigIO prints which format was selected. For the automatic
case the messages are (illustrative; the per-channel analysis lines vary with
your data):

```
Using EDF format (16-bit) based on signal analysis (precision within acceptable range).

EMG data exported to: output.edf
```

or:

```
Using BDF format (24-bit) based on signal analysis to preserve precision.
Reason: Channel 'EMG1': <reason from signal analysis>

EMG data exported to: output.bdf
```

Because `to_edf` auto-selects the extension, an `'output'` path becomes
`output.edf` or `output.bdf` depending on the analysis result.

## Best Practices

- Let biosigIO automatically determine the format when possible
- When in doubt, visualize your data to see if it contains very small but important signal components
- For critical applications, consider using the BDF format to ensure maximum precision
- If file size is a concern and precision is adequate, force the EDF format to save space
