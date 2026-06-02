# BIDS Helpers

Helpers for the Brain Imaging Data Structure (BIDS): locate a sibling
`_channels.tsv` / `_events.tsv` next to a recording, apply its per-channel
types/units (`apply_channels_tsv`), and load its authoritative event table into
`rec.events` (`apply_events_tsv`). `apply_channels_tsv` is applied automatically
by `Recording.from_file` (unless `bids_channels="off"`); `apply_events_tsv` is
called explicitly when the sidecar events should override the data file's own
markers.

## Module Documentation

::: biosigio.bids
    options:
      show_root_heading: true
      show_source: true
      members: true
