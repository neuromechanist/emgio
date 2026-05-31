# Biosignal Zarr Serving Format

**PRD and engineering specification**

| | |
|---|---|
| Status | Draft for engineering review |
| Owner | Yahya Shirazi |
| Applies to | `biosigio` (formerly `emgio`), NEMAR serving layer |
| Reference code | `zarr_exporter.py` (target path `biosigio/exporters/zarr.py`) |

---

## 1. Summary

We need one derived data format that serves three jobs from a single conversion: fast viewing on edge devices, inference at the edge and in batch, and training-time streaming. Today `biosigio` exports only EDF and EDF+ (events as annotations), and NEMAR holds BIDS data on object storage. EDF is an archival container, not a serving format, so neither viewing nor ML streams from it well.

The decision is to add a **sharded Zarr v3 exporter with a min/max view pyramid**. The BIDS dataset stays the source of truth. The Zarr store is a derived serving copy that all three consumers read directly from object storage, with no decode backend required for reads.

This document records the decisions, the on-disk contract, the signal-processing rules, the serving model, and the work plan.

---

## 2. Goals and non-goals

**Goals**

- One conversion and one store per recording. No separate viewing and training pipelines.
- Interactive viewing of arbitrarily long recordings on thin clients, with bounded bytes per view.
- Inference-ready signal at a sensible per-modality rate, readable in the browser and in Python.
- Training-time throughput comparable to a sharded sequential format.
- Reproducible downsampling, with provenance in metadata.

**Non-goals**

- Replacing BIDS or EDF. BIDS remains the archival source of truth and the citable artifact.
- Replacing XDF or LSL. Those stay the acquisition and live-stream layer. This format is downstream of acquisition.
- A general analysis API. The store is a typed array contract that clients read directly.

---

## 3. Why Zarr, and why not the alternatives

We evaluated the formats already in or adjacent to the field.

| Format | Role it solves well | Why not the single serving store |
|---|---|---|
| EDF, EDF+ | Archival interchange, current biosigio output | Record-interleaved, no index, no multi-resolution, not browser-native |
| XDF | LSL acquisition and live multi-stream recording | Append-only, no index, in-memory load, no browser reader. NEMAR data is not even in XDF |
| WebDataset | Training throughput (sequential shards) | No random access, no pyramids, cannot serve interactive viewing |
| Parquet, Arrow | Events, annotations, channel and dataset metadata | Wrong shape for dense multichannel signal and view pyramids |
| **Zarr v3 (sharded)** | **Chosen** | Chunked random access plus shard-level sequential reads, multi-resolution, cloud-native, browser readers exist |

The deciding feature is **Zarr v3 sharding**. A shard is one object holding many chunks plus an index. Training reads a whole shard sequentially in one request and gets many windows. Viewing and inference range-read a single chunk inside that shard. The same bytes serve both the random and the sequential access patterns, which is what lets one store cover all three jobs.

Parquet and Arrow still have a role around the signal: events and metadata. We keep those lightweight and do not force them into the array store.

---

## 4. Store contract

One Zarr v3 group per recording. The reader and writer both honor this layout.

```
/                              root group
  attrs: biosigio_version, format="biosigio-zarr", format_version,
         source_format, modality_rates, dtype, view_downsample,
         anti_alias_filter, channel_groups, recording_metadata, created_utc

  <modality>_<rate>hz/         one group per (modality, native rate)
    attrs: modality, rate, original_rate, n_channels, n_samples, channels[]
    0                          (n_ch, n_time) base signal, sharded
      attrs: level=0, rate, downsample_factor=1, kind="signal",
             usable_for_inference=true, scale[], offset[],
             physical_formula="physical = digital * scale + offset"
    view/                      min/max render pyramid (not sharded)
      1, 2, ...                (2, n_ch, n_time_L), axis0 = [min, max]
        attrs: level, downsample_factor, rate_effective,
               kind="minmax_envelope", usable_for_inference=false

  events/
    onset (f64), duration (f64), code (i32)
    attrs: label_map {code: description}, n_events
```

Per-channel metadata (in each group's `channels` attribute):

`label, channel_type, modality, unit, prefilter, original_rate, target_rate, anti_aliased, usable_for_inference, scale, offset, row_index`

Key contract points:

- **`level 0` is the inference signal.** Anti-aliased, resampled to the canonical rate, one sample per time step. Inference and training read this.
- **`view/*` are render envelopes.** Min/max per downsample bin, two values per bin, nonlinear. They are flagged `usable_for_inference=false`. Clients must not train or infer on them.
- **`physical = digital * scale + offset`**, per channel, when `dtype="int16"`. For `float32` the scale is 1 and offset is 0.
- **Events** are stored as onset, duration, and an integer code per event, with a portable code to label map. This stays compact when there are many events with few unique labels.

### Implementation notes (as shipped in emgio 0.6.0)

These clarify the contract to match the reference implementation (`emgio/exporters/zarr.py`, `emgio/importers/zarr.py`). Readers built to this PRD should rely on them:

- **`format_version` is `2`** (`format = "biosigio-zarr"`). Readers should accept a version less than or equal to the one they know and reject a newer one. Version 2 stores `recording_metadata` as a **native JSON object** (directly readable by a browser/zarrita client); version 1 stored it as a JSON string. Both encode non-JSON values (datetime, date, numpy) as typed envelopes `{"__biosigio_type__": "datetime"|"date", "value": ...}`; a reader detects and decodes these.
- **Group directory names encode the TARGET (served) rate**, not the native rate (e.g. a 500 Hz native EEG group is named `eeg_250hz`). Address groups by the names in the root `channel_groups` attribute, never by reconstructing from a native rate. The native rate is recovered from the group `original_rate` attribute.
- **`original_rate` has two scopes:** the group-level `original_rate` is the integer-rounded grouping key, while the per-channel `channels[].original_rate` is the true (possibly fractional) acquisition rate. Use the per-channel value for exact provenance.
- **The `level 0` array also carries an `anti_aliased` boolean** (group-level summary; true if any channel was anti-alias resampled) in addition to the per-channel `channels[].anti_aliased`.
- **`usable_for_inference` on `level 0`** is true for groups with at least one continuous channel and **false for a group composed solely of discrete channels** (TRIG/SYSCLOCK/CTRL).
- **`int16` storage rejects non-finite samples** (NaN/inf) with a clear error; use `dtype="float32"` to preserve NaN gaps. A reader can therefore assume an `int16` store has no NaN.
- **Resampling and grouping operate on rates rounded to the nearest integer Hz**, so for a fractional acquisition rate the effective polyphase ratio is approximate (the exact native rate is preserved in `channels[].original_rate`).
- The root carries one extra free-text `note` attribute (human-readable provenance); readers may ignore it.

---

## 5. Signal-processing requirements

These are correctness rules, not preferences.

**Per-modality canonical rate.** `target = min(native, cap)`, never upsample.

| Modality | Default cap (Hz) | Rationale |
|---|---|---|
| EEG | 250 | Scalp content sits below ~100 Hz; 250 keeps a clean passband to ~100 Hz |
| MEG | 250 | Same default as EEG. See open decision in section 8 |
| iEEG (SEEG, ECoG, DBS) | 1000 | Ripples and high-gamma live at 80 to 250 Hz and above |
| EMG | 1000 | Surface and HD-EMG carry content to several hundred Hz |
| Other (BEH, MISC) | native | No cap; kept as recorded |

The cap table is a single configurable parameter, not hard-coded behavior.

**Anti-aliasing.** Downsampling `level 0` uses polyphase resampling (`scipy.signal.resample_poly`). Decimating without a low-pass would fold line-noise harmonics and muscle into band. This is the analysis data, so it must be spectrally clean.

**View pyramid uses min/max, not anti-aliased decimation.** Envelopes preserve visible transients so spikes do not vanish when zoomed out. This is why view levels are not inference-usable. The two downsampling philosophies coexist on purpose: anti-aliased for `level 0`, min/max for `view/*`.

**Discrete channels.** Trigger and clock types (`TRIG`, `SYSCLOCK`, `CTRL`) are resampled by nearest sample, with no anti-alias filter, so step edges survive. They are flagged `usable_for_inference=false`.

**Heterogeneous channels.** BIDS allows mixed channel types and, less often, mixed native rates within a modality. Channels are grouped by `(modality, native rate)` so every array stays length-consistent. The common single-rate recording yields one group per modality. A genuinely mixed-rate source yields one group per rate, which is faithful rather than silently resampled together.

---

## 6. Serving and deployment

**Reads need no backend.** A Zarr v3 store is objects plus JSON. A browser reader (zarrita) streams it directly from object storage over HTTPS with ranged GETs. This is a direct payoff of choosing Zarr: the viewing path has no decode service.

**CORS is the one hard requirement.** The bucket must allow browser GET, HEAD, and Range, and expose ETag and Content-Length.

**CDN is recommended for the public production viewer**, for reasons specific to this workload:

1. Many small requests. A viewport pulls several chunk objects. A CDN serves from an edge POP with HTTP/2 or HTTP/3 multiplexing, so many chunk fetches share one connection.
2. Hot, shared coarse levels. Overview pyramid levels and `zarr.json` are tiny and identical across users, so they cache at the edge with a near-total hit rate.
3. Cost. Browser viewing generates many small GETs. A CDN collapses request count and origin egress, the dominant cost for a public archive. Cloudflare R2 plus its CDN, whose model has no egress fees, is worth evaluating against S3 plus CloudFront.

**The viewer code is identical with or without a CDN**, since zarrita reads object URLs. Ship on S3 plus CORS, then front it with a CDN by changing the base URL. The store is also already CDN-friendly: view levels are not sharded, so the hot viewing path is plain one-object-per-chunk GETs with no shard-index double-fetch. Sharding stays on the cold throughput path (`level 0` and training).

**Caching.** The store is write-once per dataset version. Set immutable, long max-age headers on chunks and version the store prefix with a build hash or dataset version, so invalidation is never needed. Keep `zarr.json` on a shorter TTL or version it too.

**Restricted data.** Public datasets get a public-read bucket behind the CDN. Access-controlled datasets cannot be public, so use signed cookies at the edge over a whole dataset prefix, rather than per-object signed URLs.

---

## 7. Reader contract

**Viewer.** Read root and group attrs. For each group, pick the level whose `rate_effective` puts about one sample per screen pixel, fetch the chunks covering the viewport, and render the min/max band using per-channel `scale` and `offset`. At maximum zoom read `level 0` directly. Group display by modality. Overlay events from the `events` group.

**Inference.** Read `level 0` of the relevant group, apply scale and offset, window. Never read `view/*`. For a lower analysis rate (for example 100 or 128 Hz models), derive it on read from `level 0` with an anti-aliased resample rather than persisting a third tier, unless that rate becomes a high-volume edge path.

**Training.** Iterate shards of `level 0` sequentially, one object per shard, many windows per object. Shuffle at shard level plus a within-shard buffer. Use tensorstore for async, concurrent reads into the training loop.

---

## 8. Open decisions and risks

| Item | Decision needed | Default for now |
|---|---|---|
| MEG rate | MEG at 250 Hz drops high-gamma. Raise the cap or set per-dataset | 250 Hz, flagged |
| int16 vs float32 | int16 halves bytes with small quantization loss; float32 is lossless | int16, float32 available per call |
| Low-rate inference tier | Persist a 100 or 128 Hz tier, or derive on read | Derive on read |
| Window and chunk coupling | Base chunk size sets the random-access grain | Small base chunk, compose larger windows |
| Conversion scale | Compute and storage budget for converting the NEMAR archive | Batch job, per-dataset, idempotent |
| Event table size | Code plus map is fine for typical sizes; very large tables need review | Code plus map |

---

## 9. Work plan

| Milestone | Deliverable | Status |
|---|---|---|
| M0 | Zarr writer (`zarr_exporter.py`), synthetic roundtrip verified | Done |
| M1 | `Recording.to_zarr` wiring, plus unit and roundtrip tests in repo style | Done (`emgio/exporters/zarr.py`, `test_zarr.py`) |
| M2 | `from_zarr` reader for verification and braindecode and training integration | Done (`emgio/importers/zarr.py`, `importer="zarr"` + `.zarr` autodetect) |
| M3 | Batch conversion over NEMAR BIDS: per-dataset, idempotent, versioned output prefix, manifest, all four modalities | |
| M4 | Web viewer: zarrita plus WebGL, pyramid level selection, events overlay | |
| M5 | Serving infra: bucket layout, CORS, CDN, cache headers, auth for restricted data | |
| M6 | Optional derive-on-read low-rate inference tier | |

Cross-cutting: validation that reconstructed `level 0` matches the source within tolerance, and CI.

The reference writer already implements per-modality resampling, anti-aliased `level 0`, the min/max pyramid, int16 scaling, heterogeneous-channel grouping, discrete-channel handling, events, and provenance attrs. Wiring it into the core mirrors the existing `to_edf` entry point:

```python
def to_zarr(self, filepath: str, **kwargs) -> str:
    """Export to a sharded Zarr store with a min/max view pyramid."""
    from ..exporters.zarr import ZarrExporter
    if self.signals is None:
        raise ValueError("No signals loaded")
    return ZarrExporter.export(self, filepath, **kwargs)
```

---

## 10. Acceptance criteria

- One conversion produces viewing, inference, and training capability from a single store. No second pipeline exists.
- An hour-long recording opens at overview zoom within an interactive budget and a bounded transfer, independent of recording length.
- `level 0` reconstruction matches the anti-aliased source within tolerance (int16 quantization step or better).
- Shard streaming throughput is comparable to a WebDataset baseline on the same data.
- Metadata flags prevent training or inference on view envelopes.
- The viewer reads the store directly from object storage with no decode backend.

---

## Appendix A. Default parameters

| Parameter | Default | Meaning |
|---|---|---|
| `modality_rates` | EEG 250, MEG 250, iEEG 1000, EMG 1000 | Per-modality canonical rate cap |
| `dtype` | `int16` | Storage type; `float32` for lossless |
| `view_downsample` | 4 | Time decimation factor between pyramid levels |
| `min_view_samples` | 512 | Stop building pyramid levels at this length |
| `chunk_seconds` | 4 | Random-access grain |
| `shard_seconds` | 300 | Sequential-read grain, rounded to whole chunks |
| `compressor_level` | 5 | zstd level (Blosc codec) |

## Appendix B. Bucket CORS, minimum

Allow `GET` and `HEAD`, allow the `Range` request header, and expose `ETag`, `Content-Length`, and `Content-Range`. Set immutable, long max-age cache headers on chunk objects and version the store prefix so invalidation is never required.
