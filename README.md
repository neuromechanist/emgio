# EMGIO

[![PyPI version](https://badge.fury.io/py/emgio.svg)](https://badge.fury.io/py/emgio)
[![Tests](https://github.com/neuromechanist/emgio/actions/workflows/tests.yml/badge.svg)](https://github.com/neuromechanist/emgio/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/neuromechanist/emgio/branch/main/graph/badge.svg?token=63EDIA9TWD)](https://codecov.io/gh/neuromechanist/emgio)

A Python package for EMG data import/export and manipulation. This package provides a unified interface for working with EMG data from various systems (Trigno, EEGLAB, OTB, etc) and exporting to standardized formats like EDF and BDF with harmonized metadata.

The determination of the EDF/BDF format is based on the dynamic range of the data. If the data is within the range of 16-bit integers (~90dB), the EDF format is used. Otherwise, the BDF format is used. This is to ensure that the data is stored in the most efficient format possible. This determination is made automatically using SVD decomposition and/or FFT to determine the dynamic range of the data. (Alternatively, the user can override the format selection by explicitly indicating their desired format).

## Documentation

The documentation including installation instructions, examples, and API reference is available at [https://neuromechanist.github.io/emgio/](https://neuromechanist.github.io/emgio/).

## Features

- Import EMG data from multiple systems:
  - EEGLAB set files (supported)
  - Delsys Trigno (supported)
  - OTB Systems (supported)
  - EDF/BDF(+) (supported, including annotations)
  - WFDB (supported, including annotations)
  - XDF/Lab Streaming Layer (supported, multi-stream)
  - Generic CSV (supported with auto-detection)
  - Noraxon (planned)
  
- Smart import:
  - Automatic file format detection based on extension
  - Specialized format detection for CSV files
  - Custom importers for system-specific formats
  - Automatic annotation loading (WFDB, planned for EDF+/BDF+ and EEGLAB's .set files)
  - LSL timestamp preservation for XDF files (for synchronization)
  
- Export to standardized formats:
  - EDF/BDF(+) with channels.tsv metadata (automatically selects format based on signal properties, preserves annotations)
  
- Data manipulation:
  - Channel selection
  - Metadata handling
  - Event/Annotation handling (access, add)
  - Basic signal visualization
  - Raw data access and modification

## Installation

### From PyPI (recommended)

```bash
pip install emgio
```

### From source

```bash
git clone https://github.com/neuromechanist/emgio.git
cd emgio
pip install .
```

## Usage

### Basic Example

```python
from emgio import EMG

# Load data with automatic format detection
emg = EMG.from_file('data.csv')  # Format detected from file extension

# Load data with explicit importer
emg = EMG.from_file('data.csv', importer='trigno')

# Plot specific channels
emg.plot_signals(['EMG1', 'EMG2'])

# Export to EDF or BDF (format automatically determined)
emg.to_edf('output.edf')
```

### Generic CSV Import

```python
# Import a generic CSV file
emg = EMG.from_file('data.csv', importer='csv',
                   sample_frequency=1000,  # Required if no time column
                   has_header=True,        # Whether file has header row
                   channel_names=['EMG_L', 'EMG_R', 'ACC_X'])
```

### Channel Selection

```python
# Select specific channels
subset_emg = emg.select_channels(['EMG1', 'EMG2', 'ACC1'])

# Select all channels of a specific type
emg_only = emg.select_channels(channel_type='EMG')

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

## Acknowledgment
This project is partially supported by a Meta Reality Labs gift to @sccn and NIH 5R01NS047293.