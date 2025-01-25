# EMGIO

[![Tests](https://github.com/neuromechanist/emgio/actions/workflows/tests.yml/badge.svg)](https://github.com/neuromechanist/emgio/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/neuromechanist/emgio/branch/main/graph/badge.svg)](https://codecov.io/gh/neuromechanist/emgio)

A Python package for EMG data import/export and manipulation. This package provides a unified interface for working with EMG data from various systems (Trigno, Noraxon, OTB) and exporting to standardized formats like EDF.

## Features

- Import EMG data from multiple systems:
  - Delsys Trigno (supported)
  - Noraxon (planned)
  - OTB Systems (planned)
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
emg.select_channels(['EMG1', 'EMG2', 'ACC1'])

# Plot selected channels
emg.plot_signals()
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
