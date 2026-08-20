# MEG (.fif / CTF .ds / KIT / 4D-BTi)

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

The importer accepts four MEG layouts:

- **Elekta/Neuromag FIF (`.fif`):** A single FIFF (Functional Imaging File Format) binary file. This is the native format used by MEGIN/Elekta/Neuromag systems and is also MNE-Python's general-purpose container.
- **CTF (`.ds`):** A directory (with a `.ds` extension) holding the data and metadata files written by CTF MEG systems. Pass the path to the `.ds` directory itself, not to an individual file inside it.
- **KIT/Yokogawa/RICOH (`.con` / `.sqd` / `.kdf`):** A single binary file.
- **4D Neuroimaging/BTi:** A directory with **no file extension**. BIDS names it
  `sub-<label>[_ses-<label>]_task-<label>[_run-<index>]_meg/` and it holds the
  processed-data file (conventionally named `c,rfDC`) plus a `config` file and,
  usually, an `hs_file` head-shape file. Because there is no extension to key off,
  detection is content-based: a directory is recognized as BTi only if it directly
  contains (not in a subdirectory) both a file whose name starts with `c,rf` and a
  sibling `config` file. This two-signal check is deliberate -- matching on `config`
  alone would misfire on the `.datalad/config` that almost every datalad-tracked
  dataset repo has.

The file extension determines which MNE reader is used: `.fif` files are read with
`read_raw_fif`, `.ds` directories with `read_raw_ctf`, `.con`/`.sqd`/`.kdf` with
`read_raw_kit`, and a detected BTi directory with `read_raw_bti` (its PDF file is
passed as `pdf_fname`; `config`/`hs_file` are resolved as siblings, `hs_file` only
when present, since it is optional in BIDS).

## Loading Data

To load a MEG recording, pass the path to the file or directory to `Recording.from_file`. Every layout is auto-detected, so the importer is selected automatically:

```python
from biosigio import Recording

# Load an Elekta/Neuromag FIF file (importer inferred from .fif)
rec = Recording.from_file('sample.fif')

# Load a CTF .ds directory (importer inferred from .ds)
rec = Recording.from_file('recording.ds')

# Load a KIT .sqd file (importer inferred from .sqd)
rec = Recording.from_file('recording.sqd')

# Load a 4D/BTi directory (importer inferred from directory content)
rec = Recording.from_file('sub-01_task-rest_meg')
```

You can also select the MEG importer explicitly, which is useful if a file does not carry the expected extension (or you already know a directory is BTi and want to skip the content check):

```python
rec = Recording.from_file('sample.fif', importer='meg')
rec = Recording.from_file('sub-01_task-rest_meg', importer='meg')
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

- `source_file`: Path to the loaded file or directory.
- `number_of_signals`: Total channel count across all sensor types.

## Round-Trip to EDF/BDF

MEG sensors measure very small magnitudes; magnetometer signals are on the order of tesla (around 1e-12 T, femtotesla in practice) and planar gradiometers are in tesla per meter. Because biosigIO preserves the per-channel physical dimension and the full sample values, these small-magnitude channels round-trip correctly through EDF/BDF export and re-import. The automatic EDF/BDF format selection accommodates the wide dynamic range of MEG data, so exporting a loaded MEG recording and reading it back preserves the signal values and channel information.
