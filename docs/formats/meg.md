# MEG (.fif / CTF .ds)

biosigIO supports loading magnetoencephalography (MEG) recordings through MNE-Python. MEG files capture the magnetic fields produced by neuronal activity using arrays of superconducting sensors, and a single file typically mixes several sensor types (magnetometers, gradiometers, reference sensors) alongside electroencephalography (EEG), electrooculography (EOG), electrocardiography (ECG), and stimulus/trigger channels.

## Installation

The MEG importer depends on MNE-Python, which is a heavy optional dependency and is not installed by default. Install it through the `meg` extra:

```bash
uv sync --extra meg
```

For an existing installation, you can install the extra directly:

```bash
uv pip install 'biosigio[meg]'
```

If MNE-Python is not installed, loading a MEG file raises an `ImportError` with the install hint above.

## File Structure

The importer accepts two MEG layouts:

- **Elekta/Neuromag FIF (`.fif`):** A single FIFF (Functional Imaging File Format) binary file. This is the native format used by MEGIN/Elekta/Neuromag systems and is also MNE-Python's general-purpose container.
- **CTF (`.ds`):** A directory (with a `.ds` extension) holding the data and metadata files written by CTF MEG systems. Pass the path to the `.ds` directory itself, not to an individual file inside it.

The file extension determines which MNE reader is used: `.fif` files are read with `read_raw_fif` and `.ds` directories are read with `read_raw_ctf`.

## Loading Data

To load a MEG recording, pass the path to the `.fif` file or `.ds` directory to `Recording.from_file`. Both extensions are auto-detected, so the importer is selected automatically:

```python
from biosigio import Recording

# Load an Elekta/Neuromag FIF file (importer inferred from .fif)
rec = Recording.from_file('sample.fif')

# Load a CTF .ds directory (importer inferred from .ds)
rec = Recording.from_file('recording.ds')
```

You can also select the MEG importer explicitly, which is useful if a file does not carry the expected extension:

```python
rec = Recording.from_file('sample.fif', importer='meg')
```

## Channel Types

MEG files contain many channel kinds in a single recording. The importer keeps each MNE channel type distinct rather than collapsing everything into one generic "MEG" type, so magnetometers, gradiometers, and reference sensors remain separable downstream. The mapping from MNE channel types to biosigIO channel types is:

| MNE type | biosigIO type | Description |
|----------|---------------|-------------|
| `mag` | `MEGMAG` | Magnetometer (also how MNE reports CTF axial gradiometers) |
| `grad` | `MEGGRADPLANAR` | Planar gradiometer |
| `ref_meg` | `MEGREFMAG` | MEG reference sensor |
| `eeg` | `EEG` | Electroencephalography |
| `seeg` | `SEEG` | Stereo-electroencephalography |
| `ecog` | `ECOG` | Electrocorticography |
| `dbs` | `DBS` | Deep brain stimulation |
| `eog` | `EOG` | Electrooculography |
| `ecg` | `ECG` | Electrocardiography |
| `emg` | `EMG` | Electromyography |
| `stim` | `TRIG` | Stimulus/trigger channel |
| `resp` | `RESP` | Respiration |
| `gsr` | `GSR` | Galvanic skin response |
| `temperature` | `TEMP` | Temperature |
| `bio`, `misc`, `syst`, `chpi`, `exci`, `ias` | `MISC` | Miscellaneous |

Any MNE type not listed above is mapped to `OTHER`.

## Physical Units

Each channel keeps the physical unit reported by the FIFF header. The importer reads the FIFF unit code from `raw.info['chs'][i]['unit']` and maps it to a dimension string:

| FIFF unit code | Physical dimension |
|----------------|--------------------|
| 107 | `V` (volts) |
| 112 | `T` (tesla) |
| 201 | `T/m` (tesla per meter) |

EEG, EOG, ECG, and similar electric channels carry volts (`V`); magnetometers and reference magnetometers carry tesla (`T`); planar gradiometers carry tesla per meter (`T/m`). Any unit code not listed is recorded as `n/a`. The underlying samples are read from MNE in SI units.

## Events

Stimulus/trigger channels (MNE type `stim`) are not only loaded as data channels; their transitions are also extracted as events. The importer calls MNE's `find_events` and converts each detected trigger into a row in `rec.events`, with onsets expressed relative to the start of the recording (the recording's `first_samp` offset is subtracted). The `rec.events` pandas DataFrame uses the columns:

- `onset`: Time of the trigger in seconds from the start of the recording.
- `duration`: Duration in seconds (`0.0` for these point triggers).
- `description`: The integer trigger code as a string (e.g., `"1"`, `"16"`).

If the file has no stim channel or no trigger transitions, no events are added. See the [Metadata Handling](../user-guide/metadata.md#annotations-events) guide for more on working with the `events` DataFrame.

## Metadata

Loaded MEG recordings include metadata such as:

- `source_file`: Path to the loaded `.fif` file or `.ds` directory.
- `number_of_signals`: Total channel count across all sensor types.

## Round-Trip to EDF/BDF

MEG sensors measure very small magnitudes; magnetometer signals are on the order of tesla (around 1e-12 T, femtotesla in practice) and planar gradiometers are in tesla per meter. Because biosigIO preserves the per-channel physical dimension and the full sample values, these small-magnitude channels round-trip correctly through EDF/BDF export and re-import. The automatic EDF/BDF format selection accommodates the wide dynamic range of MEG data, so exporting a loaded MEG recording and reading it back preserves the signal values and channel information.
