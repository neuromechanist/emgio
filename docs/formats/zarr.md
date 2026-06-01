# Zarr Serving Store

biosigIO can export a recording to a biosigIO Zarr serving store: a single, cloud-native store that serves three downstream jobs from one conversion, namely fast interactive viewing on thin clients, edge and batch inference, and training-time streaming. This page documents the store contract and the serving model so the consumers of the store (a web viewer, an inference service, a training loader, a batch converter) can read it correctly. Those consumers are built elsewhere; biosigIO only produces and reads the store.

## Overview

The store is a sharded Zarr version 3 (v3) store with one group per (modality, native rate) pair. Level 0 of every group is the anti-aliased inference signal, resampled to a per-modality canonical rate, with a min/max view pyramid above it for rendering.

The store is a derived serving copy, not an archive. The Brain Imaging Data Structure (BIDS) dataset, and the European Data Format (EDF) files alongside it, remain the authoritative source of truth and the citable artifact. The Zarr store is reproducible from them, and the downsampling parameters are recorded in the store attributes so the derivation is documented.

Zarr is an optional dependency (the `zarr` extra, which installs zarr v3) and is imported lazily. Install it with one of:

```bash
uv sync --extra zarr           # development install
uv pip install 'biosigio[zarr]'   # existing environment
```

Write a store with `Recording.to_zarr`:

```python
from biosigio import Recording

rec = Recording.from_file('recording.edf')

# Defaults: int16 storage, per-modality rate caps, min/max view pyramid
store_path = rec.to_zarr('recording.zarr')
print(store_path)  # 'recording.zarr'
```

## Store layout

One Zarr v3 root group holds the whole recording. Both the writer and the reader honor this layout:

```
/                              root group
  attrs: biosigio_version, format="biosigio-zarr", format_version,
         source_format, modality_rates, dtype, view_downsample,
         anti_alias_filter, channel_groups, recording_metadata, created_utc

  <modality>_<rate>hz/         one group per (modality, native rate)
    attrs: modality, rate, original_rate, n_channels, n_samples, channels[]
    0                          (n_ch, n_time) level-0 signal, sharded
      attrs: level=0, rate, downsample_factor=1, kind="signal",
             usable_for_inference, scale[], offset[],
             physical_formula="physical = digital * scale + offset"
    view/                      min/max render pyramid (not sharded)
      1, 2, ...                (2, n_ch, n_time_L), axis0 = [min, max]
        attrs: level, downsample_factor, rate_effective,
               kind="minmax_envelope", usable_for_inference=false

  events/
    onset (f64), duration (f64), code (i32)
    attrs: label_map {code: description}, n_events
```

The group name encodes the served (target) rate, not the source rate, for example `eeg_250hz` even when the source was recorded at 500 Hz. The two rates are both kept in the group attributes:

- `group.attrs['rate']` is the served target rate (the rate of level 0).
- `group.attrs['original_rate']` is the integer-rounded native rate used as the grouping key.
- Each entry in `group.attrs['channels'][].original_rate` is the true per-channel acquisition rate as a float, so a non-integer source rate (for example 511.7 Hz) is preserved even though the group key rounds it (metadata loss is data loss).

The level-0 array carries the per-channel `scale` and `offset`. Reconstruct physical units with:

```
physical = digital * scale + offset
```

Per-channel metadata lives in each group's `channels` attribute, one object per channel with these keys:

`label`, `channel_type`, `modality`, `unit`, `prefilter`, `original_rate`, `target_rate`, `anti_aliased`, `usable_for_inference`, `scale`, `offset`, `row_index`.

The `row_index` maps a channel to its row in the `(n_ch, n_time)` level-0 array.

## Level 0 vs view/*

The two array tiers exist for different jobs and are downsampled by deliberately different rules:

- **Level 0** is the anti-aliased inference signal, one sample per time step at the group's canonical rate. Inference and training read this tier. Its `usable_for_inference` flag is `true` for groups that contain at least one continuous channel and `false` for a group made up entirely of discrete channels.
- **view/\*** are min/max render envelopes. Each level stores two values per downsample bin, the minimum and the maximum, on axis 0 (`[min, max]`). Envelopes preserve visible transients so brief spikes do not vanish when zoomed out, which is why they are nonlinear and are flagged `usable_for_inference=false`. Never train or run inference on the view tier.

The two downsampling philosophies coexist on purpose: anti-aliased polyphase resampling for level 0, min/max binning for view/*.

## Per-modality rates and dtype

**Rate caps.** Level 0 is resampled to `target = min(native, cap)` per modality, and the store never upsamples. The default caps are:

| Modality | Default cap (Hz) |
|---|---|
| EEG | 250 |
| MEG | 250 |
| iEEG (SEEG, ECoG, DBS) | 1000 |
| EMG | 1000 |
| Other (BEH, MISC, ...) | native (no cap) |

A modality absent from the cap table keeps its native rate. Override the caps per call with the `modality_rates` argument:

```python
# Keep MEG high-gamma by raising its cap; EEG stays at the default
rec.to_zarr('recording.zarr', modality_rates={'MEG': 1000, 'EEG': 250})
```

**Storage dtype.** The default `dtype="int16"` stores each channel with a per-channel `scale` and `offset` (half the bytes of float32; consumers cast on read). int16 cannot represent NaN or inf, so the exporter rejects non-finite samples rather than silently corrupting them; use `dtype="float32"` (lossless, and it keeps NaN) when a channel carries gaps:

```python
rec.to_zarr('recording.zarr', dtype='float32')
```

**Discrete channels.** Trigger and clock channel types (`TRIG`, `SYSCLOCK`, `CTRL`) are resampled by nearest sample with no anti-alias filter so step edges survive, and they are flagged `usable_for_inference=false`.

**Heterogeneous channels.** Channels are grouped by (modality, native rate), so each array stays length-consistent. A common single-rate recording yields one group per modality; a genuinely mixed-rate source yields one group per rate, which is faithful rather than silently resampled together.

## Reading a store

Read a store back into a `Recording` with `Recording.from_file`. The `.zarr` extension is auto-detected, or pass `importer='zarr'` explicitly:

```python
from biosigio import Recording

# Extension-inferred
rec = Recording.from_file('recording.zarr')

# Explicit importer
rec = Recording.from_file('recording.zarr', importer='zarr')
```

The importer reads level 0 of one group, applies the `physical = digital * scale + offset` dequantization, and restores channels, events, and recording metadata. The view pyramid is render-only and is never read here.

Because the groups in a store can sit at different rates that cannot share biosigIO's single time grid, the importer reconstructs one group at a time. When a store holds a single group it is selected automatically; when it holds more than one, pass the `group=` selector by group name. Calling `from_file` on a multi-group store without `group=` raises an error that lists the available groups:

```python
# Multi-group store: choose which (modality, rate) group to reconstruct
rec = Recording.from_file('recording.zarr', importer='zarr', group='eeg_250hz')
```

Reconstruction is at the store's canonical (possibly downsampled) level-0 rate, not the original full-rate signal, because the store is the serving copy and BIDS remains the source of truth.

## Format version and provenance

The root group attributes carry the format tag and version so a reader can recognize and version-check a store:

- `format` is `"biosigio-zarr"`.
- `format_version` is currently `2`. In version 2, `recording_metadata` is a native JSON object, with non-JSON-native values such as datetimes carried in typed envelopes, so a browser or zarrita reader can consume it without a second parse. Version 1 stored `recording_metadata` as a JSON string; the reader still accepts version 1 stores.

A reader should reject a store whose `format_version` is newer than the one it supports rather than guess at an unknown layout. The biosigIO importer does exactly this: it raises if the store's version exceeds the version this build reads.

## Serving model (external)

The viewing, inference, training, and batch-conversion layers live outside biosigIO. biosigIO only writes and reads the store; the sections below summarize the intended serving model so those external consumers can be built against the same contract.

**Reads need no backend.** A Zarr v3 store is objects plus JSON, so a browser reader (zarrita) streams it directly from object storage over HTTPS using ranged GETs. The viewing path has no decode service.

**Cross-Origin Resource Sharing (CORS) is the one hard requirement.** The bucket must allow browser GET, HEAD, and Range requests, and expose the ETag and Content-Length headers.

**A Content Delivery Network (CDN) is recommended for a public production viewer.** A viewport pulls several small chunk objects; an edge CDN with HTTP/2 or HTTP/3 multiplexing shares one connection across those fetches, caches the small, shared coarse pyramid levels at the edge, and collapses request count and origin egress. The viewer code is identical with or without a CDN, since zarrita reads object URLs; ship on object storage plus CORS, then front it with a CDN by changing the base URL. The hot viewing path stays CDN-friendly because the view levels are not sharded.

**Reader contracts** for the three consumers:

- **Viewer:** read the root and group attributes, pick the pyramid level whose `rate_effective` puts roughly one sample per screen pixel, fetch the chunks covering the viewport, and render the min/max band using each channel's `scale` and `offset`. At maximum zoom, read level 0 directly. Group the display by modality and overlay events from the `events` group.
- **Inference:** read level 0 of the relevant group, apply scale and offset, and window the signal. Never read view/*. For a lower analysis rate, derive it on read from level 0 with an anti-aliased resample rather than persisting a third tier.
- **Training:** iterate shards of level 0 sequentially, one object per shard and many windows per object, shuffling at the shard level plus a within-shard buffer.

## Default parameters

The exporter defaults follow the store specification. Override any of them as keyword arguments to `to_zarr`:

| Parameter | Default | Meaning |
|---|---|---|
| `modality_rates` | EEG 250, MEG 250, iEEG 1000, EMG 1000 | Per-modality canonical rate cap |
| `dtype` | `int16` | Storage type; `float32` for lossless (keeps NaN) |
| `view_downsample` | 4 | Time decimation factor between pyramid levels |
| `min_view_samples` | 512 | Stop building pyramid levels at this length |
| `chunk_seconds` | 4 | Random-access grain |
| `shard_seconds` | 300 | Sequential-read grain, rounded to whole chunks |
| `compressor_level` | 5 | zstd compression level (Blosc codec) |

```python
# A lossless store with a shorter sequential-read grain
rec.to_zarr('recording.zarr', dtype='float32', shard_seconds=120)
```
