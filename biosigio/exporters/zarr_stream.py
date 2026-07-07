"""Streaming Zarr export for large recordings (bounded RAM).

The in-memory path (:meth:`ZarrExporter.export` via ``Recording.from_file`` +
``to_zarr``) holds the whole recording at float64: MNE ``preload=True``, then
``raw.get_data()``, then a pandas DataFrame, then a de-fragmenting ``.copy()`` --
peaking at 2-3x the recording size. An 18 GB iEEG recording (BrainVision) needs
~150 GB that way and OOMs the converter (worse under the Hallu cron's ``--jobs``
fan-out). The #95 fix only streamed the *exporter*; the *importer* still loads
everything.

This path keeps peak RAM bounded by ~one channel, independent of recording size:

1. Open the source LAZILY with MNE (``preload=False``) -- works for the formats
   NEMAR serves (``.fif`` / ``.vhdr`` / ``.set`` / ``.edf`` / ``.bdf`` / CTF
   ``.ds``).
2. Pass 1: stream the source in time-windows (``raw.get_data(start, stop)``) into
   a channel-major float32 memmap on scratch. RAM = one time-window; disk I/O is a
   single sequential pass (a per-channel read of a multiplexed source would instead
   scan the file once per channel).
3. Pass 2: for each channel, read its full row from the memmap, resample +
   quantize it, and write it straight into the Zarr arrays (base row + its slice of
   each min/max pyramid level). RAM = one channel.

The resulting store is structurally identical to :meth:`ZarrExporter.export` and
numerically equal within int16 quantization. The downsampling parameters and
provenance are recorded in attrs, same as the in-memory path. Requires both the
``zarr`` and ``meg`` (MNE) extras.
"""

from __future__ import annotations

import datetime as _dt
import os
import tempfile
from typing import cast

import numpy as np

from ..core.modality import infer_modality_from_channel_type
from ..importers._mne_common import _FIFF_UNIT_TO_DIM, _MNE_TYPE_TO_biosigIO, require_mne
from ..tabular_schema import metadata_to_mapping
from ..version import __version__ as _BIOSIGIO_VERSION
from .zarr import (
    _DISCRETE_TYPES,
    DEFAULT_MODALITY_RATES,
    FORMAT,
    FORMAT_VERSION,
    _build_minmax_pyramid,
    _chunk_shard_time,
    _quantize_int16_channel,
    _resample_channel,
    _target_rate,
    require_zarr,
)


def _pyramid_level_lengths(
    n_time: int, factor: int, min_samples: int, max_levels: int
) -> list[int]:
    """Output length of each min/max pyramid level for a base of ``n_time`` samples.

    Mirrors the level-stopping rule in :func:`_build_minmax_pyramid` so the view
    Zarr arrays can be created up front (before any channel is written) and then
    filled one channel-row at a time.
    """
    lengths: list[int] = []
    n = n_time
    for _ in range(max_levels):
        if n <= min_samples:
            break
        n_out = n // factor
        if n_out < 1:
            break
        lengths.append(n_out)
        n = n_out
    return lengths


# --- Lazy stream sources (#944) -----------------------------------------------
# stream_to_zarr needs, per recording: sfreq, n_samples, and per-channel
# (label / biosigIO type / unit) metadata, plus windowed float64 blocks. MNE
# (preload=False) covers .fif/.vhdr/.set/CTF. EDF/BDF go through pyedflib INSTEAD
# -- biosigIO's in-memory path reads EDF via pyedflib, and MNE disagrees on EDF
# unit scaling (an EDF whose physical dimension is unknown: MNE assumes Volts,
# pyedflib keeps the file's dimension), so streaming EDF via MNE would NOT match a
# re-run on the in-memory path. pyedflib.readSignal(chn, start, n) is numerically
# identical to readSignal(chn) sliced, so the streamed store matches the in-memory
# store exactly. (EDF-via-stream is not used in production today, so switching its
# reader changes no existing store.)


class _MneSource:
    """MNE-backed lazy source for the formats MNE reads faithfully
    (.fif/.vhdr/.set/CTF)."""

    def __init__(self, filepath: str, force_modality: str | None):
        mne = require_mne()
        self._raw = mne.io.read_raw(filepath, preload=False, verbose="ERROR")
        self.sfreq = float(self._raw.info["sfreq"])
        self.n_samples = int(self._raw.n_times)
        types = self._raw.get_channel_types()
        self.channels: list[dict] = []
        for i, name in enumerate(self._raw.ch_names):
            ctype = _MNE_TYPE_TO_biosigIO.get(types[i], "OTHER")
            self.channels.append(
                {
                    "idx": i,
                    "label": name,
                    "channel_type": ctype,
                    "modality": force_modality or infer_modality_from_channel_type(ctype),
                    "unit": _FIFF_UNIT_TO_DIM.get(int(self._raw.info["chs"][i]["unit"]), "n/a"),
                }
            )

    def read(self, picks: list[int], start: int, stop: int) -> np.ndarray:
        return self._raw.get_data(picks=picks, start=start, stop=stop)

    def close(self) -> None:
        pass


class _EdfSource:
    """pyedflib-backed lazy source for .edf/.bdf, matching biosigIO's in-memory
    EDF importer (physical-dimension units, label/transducer channel typing).

    Uniform per-channel rate ONLY: a mixed-rate EDF raises MixedSamplingRateError
    (same as the importer's default), so it stays on the in-memory resample path
    rather than being silently mis-gridded by a single-rate streaming window."""

    def __init__(self, filepath: str, force_modality: str | None):
        import pyedflib

        from ..exceptions import MixedSamplingRateError
        from ..importers.edf import EDFImporter

        self._reader = pyedflib.EdfReader(filepath)
        headers = self._reader.getSignalHeaders()
        nsamps = self._reader.getNSamples()
        rates = {float(h["sample_frequency"]) for h in headers}
        if len(rates) > 1:
            self._reader.close()
            raise MixedSamplingRateError(
                "EDF/BDF recording has mixed per-channel sampling rates "
                f"({sorted(rates)} Hz); the streaming path needs a uniform grid. The "
                'in-memory path handles this with mixed_rate="resample".'
            )
        typer = EDFImporter()._determine_channel_type
        self.sfreq = float(headers[0]["sample_frequency"]) if headers else 0.0
        self.n_samples = int(nsamps[0]) if len(nsamps) else 0
        self.channels = []
        for i, h in enumerate(headers):
            label = cast(str, h["label"]).strip()
            transducer = cast(str, h.get("transducer", "")).strip()
            ctype = typer(label, transducer)
            self.channels.append(
                {
                    "idx": i,
                    "label": label,
                    "channel_type": ctype,
                    "modality": force_modality or infer_modality_from_channel_type(ctype),
                    "unit": cast(str, h.get("dimension", "")).strip() or "n/a",
                }
            )

    def read(self, picks: list[int], start: int, stop: int) -> np.ndarray:
        n = stop - start
        return np.stack([self._reader.readSignal(i, start, n) for i in picks]).astype(np.float64)

    def close(self) -> None:
        self._reader.close()


def _open_stream_source(filepath: str, force_modality: str | None):
    """Open a lazy source, reading EDF/BDF via pyedflib (importer parity) and
    everything else via MNE. #944"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".edf", ".bdf"):
        return _EdfSource(filepath, force_modality)
    return _MneSource(filepath, force_modality)


def stream_to_zarr(
    filepath: str,
    store_path: str,
    *,
    force_modality: str | None = None,
    modality_rates: dict[str, int] | None = None,
    dtype: str = "int16",
    events_df=None,
    recording_metadata: dict | None = None,
    view_downsample: int = 4,
    min_view_samples: int = 512,
    max_view_levels: int = 12,
    chunk_seconds: float = 4.0,
    shard_seconds: float = 300.0,
    compressor_level: int = 5,
    read_chunk_seconds: float = 30.0,
    scratch_dir: str | None = None,
) -> str:
    """Stream ``filepath`` to a Zarr serving store with bounded peak memory.

    Args:
        filepath: Source recording MNE can open (``.fif``/``.vhdr``/``.set``/
            ``.edf``/``.bdf``/CTF ``.ds``).
        store_path: Output store path (``.zarr`` appended if missing).
        force_modality: If set (e.g. ``"IEEG"`` from a BIDS suffix), assign every
            channel this modality so the recording lands in one coherent group at
            the modality's rate cap -- matching the NEMAR driver's suffix-driven
            grouping. If None, modality is inferred per channel from its type.
        modality_rates: Per-modality rate cap (defaults to
            :data:`~biosigio.exporters.zarr.DEFAULT_MODALITY_RATES`).
        dtype: ``"int16"`` (scaled) or ``"float32"`` (lossless).
        events_df: Optional events table (onset/duration/description).
        recording_metadata: Optional metadata dict stored in the store root attrs.
        read_chunk_seconds: Time-window size for the streaming transpose pass.
        scratch_dir: Directory for the temporary channel-major memmap (defaults to
            the system temp dir). Point this at fast local scratch.

    Returns:
        The store path written.
    """
    if dtype not in ("int16", "float32"):
        raise ValueError("dtype must be 'int16' or 'float32'")
    zarr = require_zarr()
    from zarr.codecs import BloscCodec

    rates = dict(DEFAULT_MODALITY_RATES if modality_rates is None else modality_rates)
    if not store_path.endswith(".zarr"):
        store_path = store_path + ".zarr"
    out_np = np.int16 if dtype == "int16" else np.float32

    # EDF/BDF read via pyedflib (importer parity), everything else lazily via MNE.
    src = _open_stream_source(filepath, force_modality)
    sfreq = src.sfreq
    n_samples = src.n_samples
    if n_samples == 0 or len(src.channels) == 0:
        src.close()
        raise ValueError("No signals loaded")

    # Per-channel metadata; group by (modality, native rate). The source is
    # single-rate (a mixed-rate EDF is rejected in _EdfSource), so every channel
    # shares `sfreq`; force_modality collapses to one group.
    groups: dict[tuple[str, int], list[dict]] = {}
    for info in src.channels:
        groups.setdefault((str(info["modality"]), int(round(sfreq))), []).append(info)

    store = zarr.storage.LocalStore(store_path)
    root = zarr.create_group(store=store, overwrite=True)
    compressors = [BloscCodec(cname="zstd", clevel=compressor_level)]
    written_groups: list[str] = []

    with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp:
        for (modality, native_rate), members in groups.items():
            target_rate = _target_rate(native_rate, modality, rates)
            n_ch = len(members)
            picks = [ci["idx"] for ci in members]

            # Pass 1: stream the source into a channel-major float32 memmap on scratch.
            # RAM here is one read window (n_ch x read_chunk), not the whole signal.
            mm_path = os.path.join(tmp, f"{modality}_{native_rate}.f32")
            mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(n_ch, n_samples))
            read_chunk = max(1, int(round(read_chunk_seconds * native_rate)))
            for s in range(0, n_samples, read_chunk):
                e = min(n_samples, s + read_chunk)
                block = src.read(picks, s, e)  # (n_ch, e-s) float64
                mm[:, s:e] = block.astype(np.float32)
            mm.flush()

            n_time = max(0, int(round(n_samples * target_rate / native_rate)))
            gname = f"{modality.lower()}_{int(round(target_rate))}hz"
            if gname in written_groups:  # single native rate per modality -> shouldn't collide
                raise ValueError(f"Zarr group name collision: {gname!r}")

            grp = root.create_group(gname)
            chunk_t, shard_t = _chunk_shard_time(n_time, target_rate, chunk_seconds, shard_seconds)
            a0 = grp.create_array(
                "0",
                shape=(n_ch, n_time),
                chunks=(n_ch, chunk_t),
                shards=(n_ch, shard_t),
                dtype=out_np,
                compressors=compressors,
            )
            view = grp.create_group("view")
            level_lengths = _pyramid_level_lengths(
                n_time, view_downsample, min_view_samples, max_view_levels
            )
            view_arrays = []
            for li, length in enumerate(level_lengths, start=1):
                eff = view_downsample**li
                ct = max(1, min(length, int(round(chunk_seconds * target_rate / eff))))
                av = view.create_array(
                    str(li),
                    shape=(2, n_ch, length),
                    chunks=(2, n_ch, ct),
                    dtype=out_np,
                    compressors=compressors,
                )
                view_arrays.append((li, eff, av))

            # Pass 2: one channel at a time -- resample, quantize, write the base row
            # and each pyramid level's row. Peak RAM is a single channel.
            scale = np.ones(n_ch, dtype=np.float64)
            offset = np.zeros(n_ch, dtype=np.float64)
            chan_meta = []
            for i, ci in enumerate(members):
                ctype = str(ci["channel_type"]).upper()
                discrete = ctype in _DISCRETE_TYPES
                x = np.asarray(mm[i], dtype=np.float64)
                y = _resample_channel(x, native_rate, target_rate, discrete=discrete)
                n_nonfinite = 0
                if dtype == "int16":
                    q, scale[i], offset[i], n_nonfinite = _quantize_int16_channel(y)
                else:
                    q = y.astype(np.float32)
                a0[i, :] = q
                if view_arrays:
                    pyramid = _build_minmax_pyramid(
                        q[np.newaxis, :], view_downsample, min_view_samples, max_view_levels
                    )
                    for (_li, _eff, av), lvl in zip(view_arrays, pyramid, strict=False):
                        av[:, i, :] = lvl[:, 0, :]
                meta = {
                    "label": ci["label"],
                    "channel_type": ctype,
                    "modality": modality,
                    "unit": ci["unit"],
                    "prefilter": "n/a",
                    "original_rate": float(native_rate),
                    "target_rate": float(target_rate),
                    "anti_aliased": bool((not discrete) and target_rate < native_rate),
                    # A zero-filled NaN/inf channel is lossy: viewable, never inferable.
                    "usable_for_inference": (not discrete) and n_nonfinite == 0,
                    "scale": float(scale[i]),
                    "offset": float(offset[i]),
                    "row_index": i,
                }
                if n_nonfinite:
                    meta["nonfinite_samples"] = n_nonfinite
                chan_meta.append(meta)
                del x, y, q
            del mm  # release the memmap before the temp dir is reclaimed

            grp.attrs.update(
                {
                    "modality": modality,
                    "rate": float(target_rate),
                    "original_rate": float(native_rate),
                    "n_channels": int(n_ch),
                    "n_samples": int(n_time),
                    "channels": chan_meta,
                }
            )
            a0.attrs.update(
                {
                    "level": 0,
                    "rate": float(target_rate),
                    "downsample_factor": 1,
                    "kind": "signal",
                    "anti_aliased": any(m["anti_aliased"] for m in chan_meta),
                    "usable_for_inference": any(m["usable_for_inference"] for m in chan_meta),
                    "scale": scale.tolist(),
                    "offset": offset.tolist(),
                    "physical_formula": "physical = digital * scale + offset",
                }
            )
            for li, eff, av in view_arrays:
                av.attrs.update(
                    {
                        "level": li,
                        "downsample_factor": int(eff),
                        "rate_effective": float(target_rate) / eff,
                        "kind": "minmax_envelope",
                        "axis0": ["min", "max"],
                        "usable_for_inference": False,
                    }
                )
            written_groups.append(gname)

        # Events as onset/duration/code arrays + a portable code->label map.
        eg = root.create_group("events")
        ev = events_df
        if ev is not None and len(ev) > 0:
            onset = ev["onset"].to_numpy(dtype=np.float64)
            duration = ev["duration"].to_numpy(dtype=np.float64)
            descs = ev["description"].astype(str).to_numpy()
            uniques = list(dict.fromkeys(descs.tolist()))
            code_of = {d: i for i, d in enumerate(uniques)}
            codes = np.array([code_of[d] for d in descs], dtype=np.int32)
            for name, arr in (("onset", onset), ("duration", duration), ("code", codes)):
                a = eg.create_array(name, shape=arr.shape, chunks=arr.shape, dtype=arr.dtype)
                a[:] = arr
            eg.attrs["label_map"] = {str(i): d for i, d in enumerate(uniques)}
            eg.attrs["n_events"] = int(len(ev))
        else:
            eg.attrs["label_map"] = {}
            eg.attrs["n_events"] = 0

        meta = dict(recording_metadata or {})
        meta.setdefault("source_file", filepath)
        meta.setdefault("number_of_signals", len(src.channels))
        meta.setdefault("streamed", True)
        root.attrs.update(
            {
                "biosigio_version": _BIOSIGIO_VERSION,
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "source_format": meta.get("source_format", "n/a"),
                "modality_rates": rates,
                "dtype": dtype,
                "view_downsample": view_downsample,
                "anti_alias_filter": "scipy.signal.resample_poly (polyphase FIR)",
                "channel_groups": written_groups,
                "recording_metadata": metadata_to_mapping(meta),
                "created_utc": _dt.datetime.now(_dt.UTC).isoformat(),
                "note": (
                    "Derived serving copy (streamed, bounded-memory conversion). level 0 "
                    "of each group is the anti-aliased inference signal; view/* are min/max "
                    "render envelopes (not for inference). BIDS source remains authoritative."
                ),
            }
        )
    src.close()
    return store_path
