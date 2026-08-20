# MEF3 (.mefd)

biosigIO supports loading intracranial EEG (iEEG) recordings stored in the
Multiscale Electrophysiology Format v3 (MEF3), used by, among others, the Mayo
Systems Electrophysiology Lab. A MEF3 recording is a **session directory** with
a `.mefd` extension, not a single file.

Reading is handled by MNE-Python's `read_raw_mef`, which delegates the actual
MEF3 parsing to the optional `pymef` package (a Python wrapper around the MEF3
C reference library). Both are needed, and MEF3 needs a **newer MNE than the
rest of the MNE-backed importers**: `read_raw_mef` was added in MNE 1.12, while
MEG/BrainVision only need MNE 1.6+. To avoid forcing that newer floor onto every
MEG/BrainVision user, MEF3 has its own `mef3` extra:

```bash
uv sync --extra mef3
# or, for an existing install:
uv pip install 'biosigio[mef3]'
```

If MNE-Python is missing, or installed but older than 1.12, or `pymef` is
missing, importing a `.mefd` recording raises a clear `ImportError` naming the
exact requirement and the install command above.

## File Structure

A `.mefd` session is a directory tree:

```
<name>.mefd/
  <CHANNEL>.timd/
    <CHANNEL>-000000.segd/
      *.tdat   # data
      *.tidx   # index
      *.tmet   # metadata
  ...           # one .timd directory per channel
```

A real session can hold well over a hundred `.timd` channel directories. Pass
the path to the `.mefd` directory itself, not to anything inside it.

## Loading Data

Provide the path to the `.mefd` directory to `Recording.from_file`. The `.mefd`
extension is recognized automatically (the same way CTF's `.ds` is), so the
importer is inferred:

```python
from biosigio import Recording

rec = Recording.from_file('sub-01_task-rest_ieeg.mefd')
```

You can also select the importer explicitly:

```python
rec = Recording.from_file('sub-01_task-rest_ieeg.mefd', importer='mef3')
```

Encrypted MEF3 sessions take a password through the importer's `password`
keyword argument (empty string, the default, for unencrypted data -- the common
case).

## Channel Types and Units

MNE assigns every channel the `seeg` type by default (MEF3 does not encode a
per-channel modality distinction the way BIDS `_channels.tsv` does), which maps
to biosigIO's `SEEG` channel type. If a recording is actually ECoG or DBS,
reassign the channel type after loading. Physical units come from each
channel's MEF3 `units_description`/`units_conversion_factor` metadata, which MNE
converts to volts; biosigIO records the resulting FIFF unit code as `V`.

## Events

MEF3's internal records and table-of-contents (TOC) gaps are exposed by MNE as
annotations on the loaded recording, the same way BrainVision's `.vmrk` markers
are. biosigIO reads these into the `rec.events` pandas DataFrame (`onset`,
`duration`, `description`, sorted by onset). If a session carries no
records/gaps, `rec.events` is left at its default empty value.

## Metadata

Loaded MEF3 recordings include metadata such as:

- `source_file`: Path to the `.mefd` directory passed to `from_file`.
- `number_of_signals`: The number of channels read from the session.

## Streaming (large recordings)

MEF3 iEEG sessions can be multi-gigabyte. `stream_to_zarr` reads `.mefd`
recordings through the same bounded-memory streaming path as `.fif`/`.vhdr`/CTF
`.ds` (MNE's `preload=False`), so converting a large session to the Zarr serving
format does not require loading it into memory all at once.

## Requirements

The MEF3 importer requires `mne>=1.12` and `pymef`, installed together through
the `mef3` extra (`uv sync --extra mef3`). This is a stricter requirement than
the `meg` extra (`mne>=1.6`), kept separate so installing `meg` alone never
forces the newer MNE version.
