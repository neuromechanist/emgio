# Welcome to biosigIO

[![PyPI version](https://badge.fury.io/py/biosigio.svg)](https://badge.fury.io/py/biosigio)
[![Tests](https://github.com/neuromechanist/biosigio/actions/workflows/tests.yml/badge.svg)](https://github.com/neuromechanist/biosigio/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/neuromechanist/biosigio/branch/main/graph/badge.svg?token=63EDIA9TWD)](https://codecov.io/gh/neuromechanist/biosigio)

biosigIO is a Python package for EMG data import/export and manipulation. It provides a unified interface for working with EMG data from various systems and exporting to standardized formats with harmonized metadata.

## Why biosigIO?

Working with EMG data across multiple recording systems can be challenging due to:

- Different file formats
- Varied metadata structures
- Inconsistent channel naming
- Diverse sampling rates and filtering

biosigIO simplifies this process by providing a standardized interface for loading, manipulating, and exporting EMG data regardless of the original source.

## Key Features

- **Multi-system support**:
  - EEGLAB set files (supported)
  - Delsys Trigno (supported)
  - OTB Systems (supported)
  - EDF/BDF(+) (supported, including annotations)
  - WFDB (supported, including annotations)
  - XDF/Lab Streaming Layer (supported, multi-stream)
  - Generic CSV (supported with auto-detection)
  - Noraxon (planned)
  
- **Intelligent import**:
  - Automatic file format detection
  - Format-specific metadata extraction
  - Handling of specialized CSV formats
  - Automatic annotation loading (WFDB, planned for EDF+/BDF+ and EEGLAB's .set files)
  - LSL timestamp preservation for XDF files (for synchronization)
  
- **Intelligent export**:
  - Automatic determination of EDF/BDF format based on signal quality
  - Smart handling of precision requirements
  - BIDS-compatible metadata formatting
  - Annotation export (EDF+/BDF+)

- **Serialization & serving** (see [Serialization & Serving](formats/serialization.md)):
  - Parquet and Arrow/Feather: lossless columnar round-trip (analytics, fast IPC); `arrow` extra
  - Zarr: cloud-native serving store (viewing, inference, and training from one store), a derived downsampled copy; `zarr` extra
  
- **Data manipulation**:
  - Channel selection
  - Metadata handling
  - Event/Annotation handling (access, add)
  - Basic signal visualization
  - Raw data access and modification

## Quick Example

```python
from biosigio import Recording

# Load data with automatic format detection, will issue an error to indicate use of the `trigno` importer
emg = Recording.from_file('data.csv')  # Format detected from file extension

# Load data with explicit importer
emg = Recording.from_file('data.csv', importer='trigno')

# Plot specific channels
emg.plot_signals(['EMG1', 'EMG2'])

# Export to EDF/BDF (format automatically determined)
emg.to_edf('output.edf')  # Extension will be added if not provided
```

## Documentation Structure

This documentation is organized as follows:

- **User Guide**: Step-by-step instructions for using biosigIO
- **Data Formats**: Details about supported input/output formats
- **API Reference**: Complete documentation of classes and methods
- **Examples**: Practical examples for various use cases

## License

This project is licensed under the BSD 3-Clause License. 
