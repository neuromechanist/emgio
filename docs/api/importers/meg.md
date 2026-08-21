# MEG Importer

Imports MEG recordings (`.fif`, CTF `.ds`, KIT `.con`/`.sqd`/`.kdf`, and 4D/BTi --
a directory with no extension, detected by content) via MNE-Python into a
`Recording`, mapping MNE channel types to distinct biosigIO types (MEGMAG /
MEGGRADPLANAR / MEGREFMAG, plus EEG/EOG/ECG/stim/...) and FIFF physical units.
Stim triggers become events. Requires the `meg` extra (MNE).

See [MEG & BrainVision](../../formats/meg.md) for details.

## Module Documentation

::: biosigio.importers.meg
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording

rec = Recording.from_file("sub-01_task-rest_meg.fif")  # or a CTF .ds directory
rec = Recording.from_file("recording.fif", importer="meg")
```
