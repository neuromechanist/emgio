# Base Importer

The `BaseImporter` class is the abstract base class for all importers in EMGIO. It defines the interface that all specific importers must implement.

## Class Documentation

::: emgio.importers.base
    options:
      show_root_heading: true
      show_source: true
      members: true

## Creating a Custom Importer

To create a custom importer for a new format, you should inherit from `BaseImporter` and implement its abstract methods:

```python
from emgio.importers.base import BaseImporter
import pandas as pd

class MyCustomImporter(BaseImporter):
    def __init__(self, file_path, **kwargs):
        """Initialize the importer with a file path."""
        self.file_path = file_path
        self.kwargs = kwargs
        
    def load(self):
        """
        Load and parse the custom format file.
        
        Returns
        -------
        tuple
            A tuple containing (signals, channels, metadata)
            - signals: pandas DataFrame with signal data
            - channels: dictionary with channel information
            - metadata: dictionary with additional metadata
        """
        # Implement file loading logic for your custom format
        signals = pd.DataFrame(...)  # Load signal data
        
        # Create channels dictionary
        channels = {
            'CH1': {
                'channel_type': 'EMG',
                'physical_dimension': 'µV',
                'sample_frequency': 1000,
                # ...other channel metadata...
            },
            # ...additional channels...
        }
        
        # Create metadata dictionary
        metadata = {
            'subject': '001',
            'recording_date': '2023-01-01',
            # ...other metadata...
        }
        
        return signals, channels, metadata
```

## Required Return Values

The `load()` method must return a tuple of three elements:

1. **signals**: A pandas DataFrame containing the signal data with:
   - Columns named after channels
   - Rows representing time points

2. **channels**: A dictionary with channel information:
   ```python
   {
       'channel_name': {
           'channel_type': str,        # Type of channel (e.g., 'EMG', 'ACC')
           'physical_dimension': str,   # Unit (e.g., 'µV', 'mV', 'g')
           'sample_frequency': float,   # Sampling frequency in Hz
           # Optional additional fields
       },
       # ...more channels...
   }
   ```

3. **metadata**: A dictionary with additional information:
   ```python
   {
       'subject': str,            # Subject identifier
       'recording_date': str,     # Recording date
       'device': str,             # Recording device
       # ...other metadata fields...
   }
   ```

## Registering a Custom Importer

To make your custom importer available through `Recording.from_file()`, you need to register it:

```python
from emgio import Recording
from my_module import MyCustomImporter

# Register the importer with a name
Recording.register_importer('my_format', MyCustomImporter)

# Now you can use it
emg = Recording.from_file('data.custom', importer='my_format')
``` 