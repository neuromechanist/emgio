# EMGIO

[![Tests](https://github.com/neuromechanist/emgio/actions/workflows/tests.yml/badge.svg)](https://github.com/neuromechanist/emgio/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/neuromechanist/emgio/branch/main/graph/badge.svg?token=63EDIA9TWD)](https://codecov.io/gh/neuromechanist/emgio)

A Python package for EMG data import/export and manipulation. This package provides a unified interface for working with EMG data from various systems (Trigno, EEGLAB, OTB, etc) and exporting to standardized formats like EDF and BDF with harmonized metadata.

The determination of the EDF/BDF format is based on the dynamic range of the data. If the data is within the range of 16-bit integers (~90dB), the EDF format is used. Otherwise, the BDF format is used. This is to ensure that the data is stored in the most efficient format possible. This determination is made automatically using SVD decomposition and/or FFT to determine the dynamic range of the data.

## Features

- Import EMG data from multiple systems:
  - EEGLAB set files (supported)
  - Delsys Trigno (supported)
  - OTB Systems (supported)
  - EDF (supported)
  - Noraxon (planned)
  
- Export to standardized formats:
  - EDF with channels.tsv metadata
- Data manipulation:
  - Channel selection
  - Metadata handling
  - Basic signal visualization
  - Raw data access and modification

## Installation

```bash
git clone https://github.com/neuromechanist/emgio.git
cd emgio
pip install .
```

## Usage

### Basic Example

```python
from emgio import EMG

# Load data from Trigno system
emg = EMG.from_file('data.csv', importer='trigno')

# Plot specific channels
emg.plot_signals(['EMG1', 'EMG2'])

# Export to EDF
emg.to_edf('output.edf')
```

### Channel Selection

```python
# Select specific channels
subset_emg = emg.select_channels(['EMG1', 'EMG2', 'ACC1'])

# Plot selected channels
subset_emg.plot_signals()
```

### Metadata Handling

```python
# Set metadata
emg.set_metadata('subject', 'S001')
emg.set_metadata('condition', 'resting')

# Get metadata
subject = emg.get_metadata('subject')
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/neuromechanist/emgio.git
cd emgio
```

2. Install for development:
```bash
pip install -e .
```

3. Install test dependencies (optional):
```bash
pip install -r test-requirements.txt
```

### Running Tests

Make sure you have installed the test dependencies first, then run:

```bash
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
