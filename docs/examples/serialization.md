# Serialization Round-Trip Examples

This example demonstrates how to serialize a `Recording` to the biosigIO columnar
and serving formats and read it back: Parquet and Arrow/Feather for lossless,
analytics-friendly round trips, and Zarr for a cloud-native serving store with a
multi-resolution view pyramid.

These formats need optional dependencies. Parquet and Arrow/Feather use the
`arrow` extra (the `pyarrow` package); Zarr uses the `zarr` extra (zarr v3):

```bash
# With uv (editable dev install)
uv sync --extra arrow      # Parquet / Arrow / Feather
uv sync --extra zarr       # Zarr serving store

# Or into an existing environment (UV-only project; use uv pip)
uv pip install 'biosigio[arrow]'
uv pip install 'biosigio[zarr]'
```

The examples below build a small `Recording` in memory so they run without any
external data file. In practice you would load a recording from any supported
format (EDF, WFDB, XDF, ...) and serialize that instead.

```python
import numpy as np
from biosigio import Recording

# Build a small two-channel recording (1 second at 1000 Hz)
fs = 1000.0
t = np.arange(fs) / fs
rec = Recording()
rec.add_channel('EMG1', np.sin(2 * np.pi * 20 * t), fs, 'mV', 'EMG')
rec.add_channel('EMG2', np.sin(2 * np.pi * 35 * t), fs, 'mV', 'EMG')
rec.add_event(onset=0.25, duration=0.10, description='burst')
```

## Parquet Round-Trip

`Recording.to_parquet()` writes a single self-describing columnar table: signal
channels become columns (the time index is preserved), and channels, events, and
recording metadata travel in the file's schema metadata. The round trip is
**lossless and bit-exact**; sample values are stored verbatim, so no quantization
or resampling occurs.

```python
# Export
rec.to_parquet('out.parquet')

# Read back; the '.parquet' extension selects the tabular importer automatically
restored = Recording.from_file('out.parquet')

# Signals are recovered exactly
assert np.array_equal(rec.signals.to_numpy(), restored.signals.to_numpy())
print(list(restored.channels.keys()))   # ['EMG1', 'EMG2']
print(restored.events.iloc[0]['description'])  # 'burst'
```

Parquet is a good choice when you want to query signals with analytics engines
(DuckDB, Polars, pandas, Spark) while keeping the full biosigIO metadata intact.

## Arrow / Feather Round-Trip

`Recording.to_arrow()` writes the same self-describing schema as Parquet, but in
the Arrow/Feather inter-process communication (IPC) format for fast, zero-copy
reads. It is also lossless and bit-exact.

```python
# Export (.feather or .arrow are both accepted)
rec.to_arrow('out.feather')

# Read back; the extension selects the tabular importer automatically
restored = Recording.from_file('out.feather')

assert np.array_equal(rec.signals.to_numpy(), restored.signals.to_numpy())
```

Both `to_parquet()` and `to_arrow()` return the written file path.

## Zarr Export and Read

`Recording.to_zarr()` writes a sharded Zarr v3 *serving* store. This is a derived
serving copy, not an archival source: `level 0` of each channel group is the
anti-aliased, per-modality-resampled inference signal, with a min/max render
pyramid stacked above it for fast viewing. Channels are grouped by
`(modality, native rate)`, so a single-rate EMG recording produces one group
named like `emg_1000hz`.

By default the store uses `int16` with a per-channel scale and offset (half the
bytes of `float32`). Pass `dtype='float32'` for a lossless store.

```python
# Lossless store (float32). The default dtype='int16' is smaller but quantized.
rec.to_zarr('out.zarr', dtype='float32')

# Read back; the '.zarr' extension selects the Zarr importer automatically.
# With a single channel group, no group= selector is needed.
served = Recording.from_file('out.zarr')
print(list(served.channels.keys()))   # ['EMG1', 'EMG2']
```

### Reading the serving signal, not the original

The Zarr read returns the **downsampled serving signal**, not the original
full-rate recording. `Recording.from_file` reconstructs `level 0` of the chosen
group at the store's canonical (possibly downsampled) rate. The view pyramid
(`view/*`) is render-only and is never read back as signal. Treat the source
recording (for example a BIDS or EDF archive) as authoritative when you need the
exact acquisition signal.

```python
# level 0 rate is the per-modality canonical rate (EMG caps at 1000 Hz by default),
# capped to never exceed the native rate.
fs_served = served.channels['EMG1']['sample_frequency']
print(fs_served)   # 1000.0 here; lower than native for, e.g., a 2048 Hz source
```

### Selecting a group in a multi-rate store

A store may hold several `(modality, rate)` groups that cannot share biosigio's
single time grid (for example an EEG group at 250 Hz and an EMG group at
1000 Hz). When more than one group is present, pass the `group=` selector to
choose which one to reconstruct:

```python
# Explicit importer plus group selector for a multi-rate store
rec = Recording.from_file('rec.zarr', importer='zarr', group='emg_1000hz')
eeg = Recording.from_file('rec.zarr', importer='zarr', group='eeg_250hz')
```

If you omit `group=` on a multi-group store, `from_file` raises a `ValueError`
listing the available group names.

## Choosing a Format

For when to reach for Parquet, Arrow/Feather, or Zarr, including the
lossless-versus-serving trade-offs and the on-disk store contract, see the
[Serialization Formats](../formats/serialization.md) reference.
