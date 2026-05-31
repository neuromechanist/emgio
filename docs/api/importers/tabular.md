# Tabular Importer (Parquet / Arrow)

Reads the columnar biosigIO formats (`.parquet`, `.feather`, `.arrow`) back into a
`Recording`, reconstructing signals, channels, events, and metadata losslessly
from the self-describing `biosigio` schema blob. Requires the `arrow` extra.

## Module Documentation

::: biosigio.importers.tabular
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording

# Auto-detected by extension, or force importer="tabular"
rec = Recording.from_file("out.parquet")
rec = Recording.from_file("out.feather", importer="tabular")
```
