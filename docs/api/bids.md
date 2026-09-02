# BIDS Helpers

Helpers for the Brain Imaging Data Structure (BIDS): locate a sibling
`_channels.tsv` / `_events.tsv` next to a recording, apply its per-channel
types/units (`apply_channels_tsv`), and load its authoritative event table into
`rec.events` (`apply_events_tsv`). `apply_channels_tsv` is applied automatically
by `Recording.from_file` (unless `bids_channels="off"`); `apply_events_tsv` is
called explicitly when the sidecar events should override the data file's own
markers.

## Units are converted, not relabelled

The `units` column describes the numbers, so adopting it **rescales the
samples** (see [Physical Units](units.md)). An MNE-backed importer returns SI
volts; a sidecar declaring `uV` therefore multiplies that channel by 10^6 and
then sets the label, leaving values and unit in agreement. It is a no-op
wherever the importer already reports the sidecar's unit (EDF via pyedflib) and
idempotent everywhere, so applying a sidecar twice cannot double-convert.

When the two units are not convertible -- different quantities, or a spelling
neither side can parse (`n/a`, `a.u.`) -- the importer's values *and* its label
are kept, and the sidecar's claim is recorded as `channels[label]["bids_unit"]`
with a warning. Nothing is ever relabelled without being converted.

## Module Documentation

::: biosigio.bids
    options:
      show_root_heading: true
      show_source: true
      members: true
