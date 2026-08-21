# MEF3 Importer

Imports MEF3 iEEG recordings (`.mefd` session directories) via MNE-Python +
pymef into a `Recording`. Channels default to `SEEG`; MEF3 records/TOC gaps
become events, the same way BrainVision's `.vmrk` markers do. Requires the
`mef3` extra (`mne>=1.12` plus `pymef` -- a stricter requirement than the `meg`
extra's `mne>=1.6`).

See [MEF3](../../formats/mef3.md) for details.

## Module Documentation

::: biosigio.importers.mef3
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording

rec = Recording.from_file("sub-01_task-rest_ieeg.mefd")
rec = Recording.from_file("sub-01_task-rest_ieeg.mefd", importer="mef3")
```
