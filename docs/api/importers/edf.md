# EDF Importer

The `EDFImporter` class is responsible for importing EMG and other physiological data from EDF (European Data Format) and BDF (BioSemi Data Format) files.

## Class Documentation

::: biosigio.importers.edf
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording
from biosigio.importers.edf import EDFImporter

# Method 1: Using Recording.from_file (recommended)
rec = Recording.from_file('data.edf', importer='edf')  # Works for both .edf and .bdf

# Method 2: Using the importer directly
rec = EDFImporter().load('data.edf')
```

## File Format Support

The EDF importer supports:

1. EDF files (16-bit resolution)
2. BDF files (24-bit resolution)
3. EDF+ / BDF+ files with annotations (loaded into `Recording.events`)

The `.edf` and `.bdf` extensions are auto-detected by `Recording.from_file`.

## Channel Type Detection

The EDF importer identifies channel types from the channel label and transducer
fields in the EDF header (e.g., a label or transducer containing 'EMG' is
classified as `EMG`; 'EEG', 'ECG'/'EKG', 'EOG', 'ACC', 'GYRO', and 'TRIG' are
handled similarly, otherwise `OTHER`).

## Parameters

`EDFImporter().load(filepath)` takes:

- **filepath (str)**: Path to the EDF/BDF file.

The importer reads the entire file; it does not support partial/segment loading.

## Return Value

The `load()` method returns a single `Recording` object with:

1. **signals**: signal data with channels as columns.
2. **channels**: per-channel information including:
   - `channel_type`: type of channel (EMG, EEG, etc.)
   - `physical_dimension`: physical unit (e.g., 'µV', 'mV')
   - `sample_frequency`: sampling rate in Hz
   - `prefilter`: pre-filtering string from the header
   - `physical_min` / `physical_max`: physical value bounds
   - `digital_min` / `digital_max`: digital value bounds
   - `transducer`: transducer string from the header
3. **metadata**: fields parsed from the EDF header, including patient/recording
   info (`patientcode`, `patient_name`, `technician`, `equipment`, `startdate`,
   ...), `file_info` (`filetype`, `number_of_signals`, `file_duration`,
   `datarecord_duration`), and `source_file`.

Annotations are NOT stored in metadata; see "Working with Annotations" below.

## Implementation Details

The EDF importer uses the `pyedflib` package to:

1. Read the EDF/BDF file header to extract metadata.
2. Extract per-channel information and convert it to biosigIO's format.
3. Load the signal data into a pandas DataFrame.
4. Read any EDF+/BDF+ annotations into the Recording's events.

## Working with Annotations

EDF+ / BDF+ files can contain annotations that mark events in the data. The
importer reads them into `Recording.events`, a pandas DataFrame with `onset`,
`duration`, and `description` columns. When the Recording is exported back to
EDF/BDF, these events are written out again as EDF+ annotations.

```python
# Load an EDF+ file with annotations
rec = Recording.from_file('data.edf', importer='edf')

# Access annotations via the events DataFrame
for _, event in rec.events.iterrows():
    print(f"Event: {event['description']} at {event['onset']}s, "
          f"duration: {event['duration']}s")
```