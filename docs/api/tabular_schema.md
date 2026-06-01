# Tabular Schema (serialization)

The canonical, versioned `biosigio` schema shared by the Parquet/Arrow exporters
and the Zarr store's metadata: it serializes a `Recording` (signals + channels +
events + metadata) self-describingly and round-trips datetimes/numpy via typed
envelopes. See [Serialization & Serving](../formats/serialization.md).

## Module Documentation

::: biosigio.tabular_schema
    options:
      show_root_heading: true
      show_source: true
      members: true
