# Tabular Exporter (Parquet / Arrow)

Exports a `Recording` to the columnar biosigIO formats: Parquet (analytics) and
Arrow/Feather (fast zero-copy IPC). Both are lossless and round-trip via
`Recording.from_file`. Requires the `arrow` extra (pyarrow).

See [Serialization & Serving Formats](../../formats/serialization.md) for when to
use each format.

## Module Documentation

::: emgio.exporters.tabular
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from emgio import Recording

rec = Recording.from_file("data.edf")
rec.to_parquet("out.parquet")   # analytics (DuckDB/Polars/pandas/Spark)
rec.to_arrow("out.feather")     # fast zero-copy IPC

# Lossless round-trip (auto-detected by extension)
rt = Recording.from_file("out.parquet")
```

## Canonical Schema

Both formats embed a self-describing, versioned `biosigio` metadata blob (see
[`emgio.tabular_schema`](../../formats/serialization.md)) carrying channels,
events, and recording metadata, so the file reconstructs the `Recording` exactly.
