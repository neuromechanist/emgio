# WFDB Format

biosigIO supports loading data stored in the Waveform Database (WFDB) format, commonly used for physiological signals like ECG and EMG, especially in datasets available on PhysioNet.

## File Structure

Loading WFDB data typically involves three files sharing the same base name (e.g., `record1`):

- **Header File (`.hea`):** Contains metadata about the recording, such as signal names, units, sampling frequency, signal resolution, start time, and pointers to the data file(s).
- **Data File(s) (`.dat`):** Contains the actual signal samples in binary format as specified in the header.
- **Annotation File (`.atr`, optional):** Contains time-stamped event annotations associated with the recording. Common annotation files use the `.atr` extension, but other extensions might exist.

## Loading Data

To load a WFDB record, provide the path to the **header (`.hea`) file** or just the base record name (if the files are in the current directory or the path is configured) to `Recording.from_file`:

```python
from biosigio.core.emg import Recording

# Load using the header file path
emg = Recording.from_file('path/to/your/record.hea')

# Or load using just the base name (looks for files in the current directory)
# emg = Recording.from_file('record')
```

biosigIO uses the `wfdb` library (PyPI package `wfdb`) internally. It ships as a core dependency, so no separate installation is required.

## Annotation Handling

If an annotation file (e.g., `record.atr`) exists in the same directory and shares the same base name as the header file, the `WFDBImporter` will **automatically load** these annotations.

Loaded annotations are stored in the `emg.events` pandas DataFrame with the following columns:

- `onset`: The time of the event in seconds from the start of the recording.
- `duration`: The duration of the event in seconds (typically 0 for standard WFDB point annotations).
- `description`: A string describing the event, usually derived from the WFDB annotation symbol (e.g., `"WFDB Annotation: N"`).

See the [Metadata Handling](../user-guide/metadata.md#annotations-events) guide for more details on accessing and working with the `events` DataFrame.

If the annotation file is not found, a message will be stored in the `emg.metadata['annotation_status']` field. 