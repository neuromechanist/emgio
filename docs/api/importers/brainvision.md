# BrainVision Importer

Imports BrainVision recordings (`.vhdr`, with sibling `.vmrk`/`.eeg`) via
MNE-Python into a `Recording`. `.vmrk` markers become events. Shares the MNE
channel-type/unit mapping with the MEG importer. Requires the `meg` extra (MNE).

See [MEG & BrainVision](../../formats/brainvision.md) for details.

## Module Documentation

::: biosigio.importers.brainvision
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording

rec = Recording.from_file("sub-01_task-rest_eeg.vhdr")
rec = Recording.from_file("recording.vhdr", importer="brainvision")
```
