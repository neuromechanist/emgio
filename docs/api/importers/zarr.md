# Zarr Importer (serving store)

Reconstructs a `Recording` from a biosigIO Zarr store, dequantizing `level 0`
(`physical = digital * scale + offset`) and restoring channels, events, and
metadata. The store is a derived serving copy, so reconstruction is at the store's
canonical (possibly downsampled) rate, not the original full-rate signal. Requires
the `zarr` extra.

## Module Documentation

::: biosigio.importers.zarr
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording

rec = Recording.from_file("out.zarr")

# A multi-rate store holds several (modality, rate) groups; pick one:
eeg = Recording.from_file("rec.zarr", importer="zarr", group="eeg_250hz")
```
