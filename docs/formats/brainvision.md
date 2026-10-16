# BrainVision (.vhdr)

biosigIO supports loading data stored in the BrainVision Core Data Format, the native format of BrainVision Recorder by Brain Products. It is a common distribution format for electroencephalography (EEG) and intracranial EEG (iEEG) datasets, including many BIDS (Brain Imaging Data Structure) datasets on OpenNeuro.

Reading is handled by MNE-Python, which reads the BrainVision triplet natively. MNE-Python is an optional, heavy dependency shared with the MEG (magnetoencephalography) importer, so it is installed through the `meg` extra:

```bash
uv sync --extra meg
# or, for an existing install:
uv pip install 'biosigio[meg]'
```

If MNE-Python is not installed, importing a BrainVision file raises an `ImportError` with this install hint.

## File Structure

A BrainVision recording is a triplet of files that share the same base name (e.g., `subject01`):

- **Header File (`.vhdr`):** A text file describing the recording: channel names, units, sampling interval, data orientation, and pointers to the marker and binary data files.
- **Marker File (`.vmrk`):** A text file listing time-stamped markers (events) such as stimulus and response triggers, segment boundaries, and recording-control notes.
- **Binary Data File (`.eeg` or `.dat`):** The actual signal samples in binary form, as specified in the header.

You always pass the **header (`.vhdr`) file** to biosigIO. MNE-Python resolves the sibling `.vmrk` and `.eeg`/`.dat` files from the `DataFile=` and `MarkerFile=` entries recorded in the header, so the three files must stay together in the same directory.

### Renamed files and stale header references

Renaming a BrainVision triplet (as BIDS conversion does) changes the file names on disk but not the `DataFile=`/`MarkerFile=` entries inside the header, so the header can name a `.eeg` or `.vmrk` that no longer exists. When a referenced file is missing and a sibling with the header's own stem exists (`<stem>.eeg`, then `<stem>.dat`, for the data; `<stem>.vmrk` for the markers), biosigIO reads a patched temporary copy of the header that points at the sibling. Every file a non-empty `DataFile=` or `MarkerFile=` entry of its `[Common Infos]` section resolves to (not only the stale one) is staged beside the copy under a fixed ASCII name, `data<ext>` or `marker.vmrk` (a symbolic link where the host allows one, else a hard link, else a byte copy), and the copy names it by that name. The copy therefore never has to spell the dataset's own directory, which may hold characters the header's codepage cannot encode (a Greek or CJK name under a cp1252 header, or bytes that are not valid UTF-8); the rest of the header, its encoding, its `Codepage=` entry and its line endings are kept. The dataset's own files are never modified. The streaming Zarr export (`stream_to_zarr`) uses the same resolution and keeps the staged files until the source is closed, since MNE reads the data file lazily.

A sibling is matched by exact name first, then with only the extension's case ignored (`<stem>.EEG` or `<stem>.VMRK` on a case-sensitive filesystem), but only when exactly one file matches; two files whose extensions differ only in case are never guessed between. The stem itself must be the header's own: it is compared exactly, except that the same text in a different Unicode normalization (an NFD `é` against an NFC one) is accepted, so `SUB-01.eeg` is never read as the data of `sub-01.vhdr`. MNE picks its marker reader by the exact `.vmrk` suffix, which the staged name `marker.vmrk` always has.

A successful recovery is logged at info level and recorded in the recording's metadata under `brainvision_header_recovered`, for example `{"DataFile": {"referenced": "old.eeg", "used": "sub-01_eeg.eeg"}}`; the streaming export writes the same entry into the store's `recording_metadata`.

A header whose references all exist is passed to MNE unchanged, and so is one where neither the named file nor a same-stem sibling exists, so that MNE reports the missing file.

Read errors are typed the same way on both paths: the importer and the streaming export classify an MNE failure with `classify_read_error`, so a corrupt header raises a `FileReadError` (or a subclass) from either path, and a message that quoted the temporary header copy or a staged file names the real `.vhdr`, or the dataset file the staged one stood for, instead. An error the operating system raises for this host rather than for the file (permission denied, an I/O error, a full disk, a stale network handle) is not typed: it propagates as the raw `OSError`, as it does from every importer (see `biosigio.is_host_condition`).

## Truncated Data Files

A binary data file whose size shows that it was cut short is refused by default with `CorruptFileError` (code `corrupt_or_truncated`), by the importer and by the streaming Zarr export alike.
A frame is one sample of every channel: the channel count times the bytes per sample of `BinaryFormat=`.
Without a positive `DataPoints=` in the header's `[Common Infos]` section, the file size must be a whole number of frames.
A trailing partial frame is refused by default, because MNE-Python would otherwise drop it, and any marker that falls in it, without a word.
This is a known refusal, and it also applies to a file with extra trailing bytes that are not a frame.
The opt-in for it is described under [Reading a File with a Partial Trailing Frame](#reading-a-file-with-a-partial-trailing-frame).
With a positive `DataPoints=`, the file must hold at least that many frames: a shorter file is refused, and a longer one is read as MNE-Python reads it, since MNE-Python sizes the recording from the file and ignores `DataPoints=` for binary data.
Vectorized data (each channel stored whole, one after another) always has a size of channels times points times bytes per sample, so a vectorized file that does not hold whole frames is refused whatever `DataPoints=` says, even if it is longer than `DataPoints=` declares.
A `DataPoints=` that is absent, empty, `0` or not an integer counts as not declared.
For `.ahdr` headers, MNE-Python counts one channel more than the header names, and so does the frame.
The error message names the data file by its file name, never its directory.
Some cut files cannot be detected.
With no `DataPoints=`, a file cut exactly at a frame boundary has no count to be compared against and still reads: multiplexed data reads as a shorter recording, while in vectorized data (each channel stored whole, one after another) the channel boundaries move, so every channel after the first is corrupted.
A vectorized file with whole extra frames beyond its `DataPoints=` is not detected either, because it cannot be told from a stale count: if the count is right, MNE-Python sizes the channel blocks from the file and the channels after the first are corrupted, and if the count is stale, the read is correct.
ASCII data has no fixed frame and is not checked, and a header that cannot be parsed or a data file that cannot be opened is left to MNE-Python, which raises its own error.

## Reading a File with a Partial Trailing Frame

A data file that ends in a partial frame is refused by default.
When the whole frames before the partial one are what you want, pass `allow_partial_frame=True`:

```python
rec = Recording.from_file('path/to/subject01.vhdr', allow_partial_frame=True)
```

`stream_to_zarr` takes the same keyword, and so does `BrainVisionImporter.load`.
`Recording.from_file` and `stream_to_zarr` ignore it for every format other than BrainVision, as `mixed_rate` is ignored outside EDF and BDF.
The other importer classes accept it for BrainVision only: `EDFImporter.load(path, allow_partial_frame=True)` raises `TypeError`.

The option covers one case only: multiplexed data with a trailing partial frame and no positive `DataPoints=`.
There, biosigIO reads the whole frames as MNE-Python does, which drops the partial bytes and any marker that falls in them, and says so.
It emits a `UserWarning` that names the data file by its file name and gives the number of dropped bytes.
It also records that number in the recording's metadata as `brainvision_dropped_bytes`, so it survives the read; the streaming export writes the same key into the store's `recording_metadata`.

The option does not change any other refusal.
A vectorized file is still refused, with a message saying that `allow_partial_frame` does not apply to vectorized data, because a cut there misaligns the channels after the first.
A file smaller than a positive `DataPoints=` is still refused, and so is a file that holds no whole frame.
With the flag on or off, a multiplexed file with a positive `DataPoints=` that ends in a partial frame is read as MNE-Python reads it, with no warning and no key.
Without the option, nothing changes.

## Loading Data

Provide the path to the `.vhdr` header file to `Recording.from_file`. The `.vhdr` extension is recognized automatically, so the importer is inferred:

```python
from biosigio import Recording

# Load a BrainVision recording (pass the .vhdr header)
rec = Recording.from_file('path/to/subject01.vhdr')
```

You can also select the importer explicitly with the `importer` argument:

```python
rec = Recording.from_file('path/to/subject01.vhdr', importer='brainvision')
```

## Channel Types and Units

Channels are read through the channel-type and unit mapping shared by all MNE-backed importers (MEG and BrainVision). Each MNE channel type maps to a distinct biosigIO channel type, so different sensor streams are preserved rather than collapsed together. The mapping includes, among others:

| MNE type | biosigIO channel type |
|----------|----------------------|
| `eeg` | `EEG` |
| `seeg` | `SEEG` |
| `ecog` | `ECOG` |
| `dbs` | `DBS` |
| `eog` | `EOG` |
| `ecg` | `ECG` |
| `emg` | `EMG` |
| `stim` | `TRIG` |
| `resp` | `RESP` |
| `misc`, `bio`, `syst` | `MISC` |

Any MNE type not in the mapping becomes `OTHER`.

Physical units are derived from the FIFF unit code that MNE assigns to each channel. Voltage channels are reported with a physical dimension of `V`; channels without a recognized unit code use `n/a`.

## Marker (Event) Handling

MNE-Python exposes the `.vmrk` markers as annotations on the loaded recording. biosigIO reads these markers into the `rec.events` pandas DataFrame, sorted by onset, with the following columns:

- `onset`: The time of the marker in seconds from the start of the recording.
- `duration`: The duration of the marker in seconds (0 for instantaneous markers).
- `description`: A string describing the marker, taken from the BrainVision marker label.

```python
rec = Recording.from_file('subject01.vhdr')

# Inspect the markers read from the .vmrk file
print(rec.events.head())
```

If the recording contains no markers, `rec.events` is left at its default empty value.

See the [Metadata Handling](../user-guide/metadata.md#annotations-events) guide for more details on accessing and working with the `events` DataFrame.

## Metadata

Loaded BrainVision recordings include metadata such as:

- `source_file`: Path to the `.vhdr` header file passed to `from_file`.
- `number_of_signals`: The number of channels read from the recording.
- `brainvision_header_recovered`: Present only when the header named missing files and same-stem siblings were read instead (see [Renamed files and stale header references](#renamed-files-and-stale-header-references)).
- `brainvision_dropped_bytes`: Present only when `allow_partial_frame=True` read a multiplexed file that ends in a partial frame; the number of trailing bytes that were dropped (see [Reading a File with a Partial Trailing Frame](#reading-a-file-with-a-partial-trailing-frame)).

```python
rec = Recording.from_file('subject01.vhdr')
print(rec.get_metadata('source_file'))
print(rec.get_metadata('number_of_signals'))
```

## Requirements

The BrainVision importer requires MNE-Python, installed through the `meg` extra (`uv sync --extra meg`). This extra is shared with the MEG importer, so installing it enables both. No additional package is needed to read the BrainVision triplet; MNE reads it natively.
