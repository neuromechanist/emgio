# Zarr Exporter (serving store)

Exports a `Recording` to a sharded Zarr v3 serving store: an anti-aliased,
per-modality-resampled `level 0` inference signal plus a min/max view pyramid for
rendering. A derived serving copy, not an archival source. Requires the `zarr`
extra (zarr v3).

See [Zarr Serving Store](../../formats/zarr.md) for the on-disk store contract and
serving model.

## Module Documentation

::: emgio.exporters.zarr
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from emgio import Recording

rec = Recording.from_file("data.edf")
rec.to_zarr("out.zarr")                 # int16 by default (per-channel scale/offset)
rec.to_zarr("lossless.zarr", dtype="float32")
```
