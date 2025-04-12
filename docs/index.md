# Welcome to EMGIO

[![Tests](https://github.com/neuromechanist/emgio/actions/workflows/tests.yml/badge.svg)](https://github.com/neuromechanist/emgio/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/neuromechanist/emgio/branch/main/graph/badge.svg?token=63EDIA9TWD)](https://codecov.io/gh/neuromechanist/emgio)

EMGIO is a Python package for EMG data import/export and manipulation. It provides a unified interface for working with EMG data from various systems and exporting to standardized formats with harmonized metadata.

## Why EMGIO?

Working with EMG data across multiple recording systems can be challenging due to:

- Different file formats
- Varied metadata structures
- Inconsistent channel naming
- Diverse sampling rates and filtering

EMGIO simplifies this process by providing a standardized interface for loading, manipulating, and exporting EMG data regardless of the original source.

## Key Features

- **Multi-system support**:
  - EEGLAB set files (supported)
  - Delsys Trigno (supported)
  - OTB Systems (supported)
  - EDF (supported)
  - Generic CSV (supported with auto-detection)
  - Noraxon (planned)
  
- **Intelligent import**:
  - Automatic file format detection
  - Format-specific metadata extraction
  - Handling of specialized CSV formats
  
- **Intelligent export**:
  - Automatic determination of EDF/BDF format based on signal quality
  - Smart handling of precision requirements
  - BIDS-compatible metadata formatting
  
- **Data manipulation**:
  - Channel selection
  - Metadata handling
  - Basic signal visualization
  - Raw data access and modification

## Quick Example

```python
from emgio import EMG

# Load data with automatic format detection, will issue an error to indicate use of the `trigno` importer
emg = EMG.from_file('data.csv')  # Format detected from file extension

# Load data with explicit importer
emg = EMG.from_file('data.csv', importer='trigno')

# Plot specific channels
emg.plot_signals(['EMG1', 'EMG2'])

# Export to EDF/BDF (format automatically determined)
emg.to_edf('output.edf')  # Extension will be added if not provided
```

## Documentation Structure

This documentation is organized as follows:

- **User Guide**: Step-by-step instructions for using EMGIO
- **Data Formats**: Details about supported input/output formats
- **API Reference**: Complete documentation of classes and methods
- **Examples**: Practical examples for various use cases

## License

This project is licensed under the BSD 3-Clause License. 