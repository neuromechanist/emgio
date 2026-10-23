import contextlib
import math
import os
import tempfile
import warnings
import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
import pyedflib

from ..analysis.signal import analyze_signal, determine_format_suitability
from ..core.emg import Recording
from ..core.modality import to_bids_channels_tsv_type

# EDF and BDF share the same 256-byte-per-signal header layout: physical_min and
# physical_max each occupy an 8-character ASCII field in BOTH formats (BDF's gain
# is the 24-bit digital sample, not a wider physical field). edflib cuts a physical
# value off at this many characters of fixed-notation text, so the character budget
# is always 8.
_PHYS_FIELD_CHARS = 8


# EDF/BDF store a signal label in a 16-character ASCII field.
_LABEL_FIELD_CHARS = 16


def _edf_signal_labels(names: list[str]) -> tuple[list[str], dict[str, str]]:
    """Fit channel labels to the 16-character EDF/BDF field without repeating one.

    Truncating to 16 characters can give two channels the same label: every
    Delsys Trigno channel of one sensor (``Mini sensor 10: ACC.X 10``,
    ``Mini sensor 10: ACC.Y 10``, ...) becomes ``Mini sensor 10: ``, and a reader
    keyed by label then keeps one of them or renames them all. A label that fits
    is written as it is. Truncated labels that collide are numbered the way the
    importers number repeated labels (``-0``, ``-1``, ..., then ``-a`` .. ``-z``),
    with the stem shortened so the suffix still fits, and never onto a label
    another channel is written under.

    Returns:
        ``(labels, renames)``: the labels to write, in channel order, and
        ``{written_label: channel_label}`` for every truncated label that had to
        be numbered (empty when truncation alone kept them distinct).

    Raises:
        ValueError: If every candidate suffix collides with a written label.
    """
    out = [name[:_LABEL_FIELD_CHARS] for name in names]
    renames: dict[str, str] = {}
    for stem in [label for label in dict.fromkeys(out) if out.count(label) > 1]:
        positions = [
            i for i, label in enumerate(out) if label == stem and len(names[i]) > _LABEL_FIELD_CHARS
        ]
        for idx, pos in enumerate(positions):
            for suffix in (str(idx), *"abcdefghijklmnopqrstuvwxyz"):
                candidate = f"{stem[: _LABEL_FIELD_CHARS - len(suffix) - 1]}-{suffix}"
                if candidate not in out:
                    break
            else:
                raise ValueError(
                    f"Channel {names[pos]!r} cannot be given a unique "
                    f"{_LABEL_FIELD_CHARS}-character EDF/BDF label"
                )
            out[pos] = candidate
            renames[candidate] = names[pos]
    return out, renames


def _fixed_text(value: float) -> str:
    """The shortest fixed-notation decimal text of ``value``, without an exponent.

    edflib writes a physical bound in fixed notation, never as ``1e-05`` (which is
    what ``str()`` gives below 1e-4): ``-1e-06`` needs nine characters as
    ``-0.000001``, and the 8-character field then keeps ``-0.00000``, a bound of
    zero. This is the text whose length has to fit the field.
    """
    return format(Decimal(repr(float(value))).normalize(), "f")


def _fit_physical_bound(value: float, *, round_up: bool, max_chars: int = _PHYS_FIELD_CHARS):
    """Round a physical bound OUTWARD to fit the EDF/BDF 8-char header field.

    The header stores physical_min/physical_max as short ASCII, in fixed notation
    (see :func:`_fixed_text`), and a value that needs more characters is cut off,
    which can move the stored bound INSIDE the data window. To guarantee the stored
    window can bracket the signal we round AWAY from the data: ``round_up=True``
    returns the smallest representable value ``>= value`` (use for physical_max);
    ``round_up=False`` returns the largest ``<= value`` (use for physical_min).
    The grid is as fine as the field allows, so a bound below 1e-4 in magnitude
    can only be held to a few digits, and a negative one to one digit less than a
    positive one because of the sign. The result's fixed-notation text fits
    ``max_chars`` whenever the magnitude is representable; an integer part with
    more than ``max_chars`` digits (|value| >= 1e8, or >= 1e7 once a sign is added)
    cannot fit, so the tightest outward value is returned and the exporter rejects
    it with a clear unit-rescale error rather than letting the header cut it off.
    Containment (the bracket) always holds. Because the digital range is mapped onto
    [physical_min, physical_max], outward rounding is an affine rescale of the
    reconstruction and barely affects correlation; clipping (which this prevents)
    is what destroys signal fidelity. A double whose text still reads back inside
    the window after the header is written (a binary expansion just below the
    decimal) is settled by :func:`_settle_physical_bounds`.
    """
    if value == 0 or not math.isfinite(value):
        return 0.0
    rounding = ROUND_CEILING if round_up else ROUND_FLOOR
    dec = Decimal(repr(float(value)))
    # Prefer the finest grid that still fits (tightest bracket), stepping down to
    # coarser rounding until the fixed-notation text fits the field.
    for places in range(max_chars, -1, -1):
        quantum = Decimal(1).scaleb(-places)
        try:
            rounded = dec.quantize(quantum, rounding=rounding)
        except InvalidOperation:
            continue
        out = int(rounded) if rounded == rounded.to_integral_value() else float(rounded)
        # float() can land one ULP on the wrong side of the decimal; if so, take
        # one more whole quantum outward so the bracket invariant always holds.
        if round_up and out < value:
            out = float(rounded + quantum)
        elif not round_up and out > value:
            out = float(rounded - quantum)
        if len(_fixed_text(out)) <= max_chars:
            return out
    # Extreme magnitude (more integer digits than the field holds): fall back to
    # a single significant figure, still rounded outward.
    quantum = Decimal(1).scaleb(dec.adjusted())
    rounded = dec.quantize(quantum, rounding=rounding)
    return int(rounded) if rounded == rounded.to_integral_value() else float(rounded)


# Tight percentile band used as the robust window only for a (near-)constant
# bulk, where the median absolute deviation is zero and cannot define a scale.
_CONSTANT_BULK_PERCENTILES = (0.1, 99.9)


def _resolve_physical_window(
    signal: np.ndarray,
    use_bdf: bool,
    clip_outliers,
    outlier_sigmas: float,
    min_effective_bits: float,
) -> tuple:
    """Decide the physical window [lo, hi] the header bounds will bracket.

    EDF/BDF map the whole digital range onto [physical_min, physical_max], so a
    single extreme outlier inflates the range and starves the bulk signal of
    quantization levels. The robust window is the min/max of the *inliers* -
    samples within ``outlier_sigmas`` robust standard deviations
    (1.4826 x median-absolute-deviation) of the median - so only genuine
    singularities fall outside it, not a fixed fraction of legitimate samples.
    For a (near-)constant bulk the MAD is zero, so a tight percentile band is
    used instead, which still isolates sparse spikes on a flat channel.

    - ``clip_outliers=False``: always the full data range (purely lossless).
    - ``clip_outliers="auto"`` (default): the full range, UNLESS keeping it would
      push the bulk below ``min_effective_bits`` of resolution at the chosen
      format, in which case the singular outliers are clipped to the robust
      window so the recording survives.
    - ``clip_outliers=True``: clip to the robust window whenever any sample lies
      outside it (a no-op when there are no outliers).

    Returns ``(lo, hi, n_clipped, max_excursion)``; ``n_clipped`` counts the
    samples that will saturate and ``max_excursion`` is how far the worst one
    lies beyond the window (both only for the caller's warning/report).
    """
    finite = signal[np.isfinite(signal)]
    if finite.size == 0:
        return 0.0, 0.0, 0, 0.0
    smin = float(np.min(finite))
    smax = float(np.max(finite))
    full_range = smax - smin
    if clip_outliers is False or full_range <= 0.0:
        return smin, smax, 0, 0.0

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad > 0.0:
        threshold = outlier_sigmas * 1.4826 * mad
        inliers = finite[np.abs(finite - median) <= threshold]
        if inliers.size == 0:
            inliers = finite
        lo, hi = float(np.min(inliers)), float(np.max(inliers))
    else:
        lo, hi = (float(v) for v in np.percentile(finite, _CONSTANT_BULK_PERCENTILES))
    core_range = hi - lo
    if core_range <= 0.0:
        return smin, smax, 0, 0.0

    bits = 24 if use_bdf else 16
    # Effective bits the bulk would retain if the full range were kept: every
    # doubling of full/core range costs one bit of bulk resolution.
    eff_bits_full = bits - math.log2(full_range / core_range) if full_range > core_range else bits
    if clip_outliers == "auto" and eff_bits_full >= min_effective_bits:
        return smin, smax, 0, 0.0  # the format absorbs the range; stay lossless

    n_clipped = int(np.count_nonzero((finite < lo) | (finite > hi)))
    if n_clipped == 0:
        return smin, smax, 0, 0.0  # nothing actually outside the window
    max_excursion = max(smax - hi, lo - smin, 0.0)
    return lo, hi, n_clipped, float(max_excursion)


def _determine_scaling_factors(
    signal_min: float, signal_max: float, use_bdf: bool = False
) -> tuple:
    """Compute EDF/BDF header scaling for a physical window [signal_min, signal_max].

    Returns ``(physical_min, physical_max, digital_min, digital_max,
    scaling_factor)``. physical_min/physical_max are rounded OUTWARD to the
    8-char header field so the stored window always brackets the input window;
    this is what stops pyedflib from saturating (clipping) the signal, which was
    the source of the per-channel corruption in issue #61. The caller chooses the
    window (full range by default, or a robust window when clipping genuine
    outliers via :func:`_resolve_physical_window`). ``scaling_factor`` is
    informational only: the exporter quantizes with :func:`_quantize_to_digital`
    from the stored bounds and does not use this value.

    Args:
        signal_min: Minimum physical value the window must contain.
        signal_max: Maximum physical value the window must contain.
        use_bdf: Whether to use BDF (24-bit) digital range.

    Returns:
        tuple: (physical_min, physical_max, digital_min, digital_max, scaling_factor)
    """
    if np.isnan(signal_min):
        signal_min = -1e-6
    if np.isnan(signal_max):
        signal_max = 1e-6
    if signal_min > signal_max:
        signal_min, signal_max = signal_max, signal_min

    if use_bdf:
        digital_min, digital_max = -8388608, 8388607  # 24-bit
    else:
        digital_min, digital_max = -32768, 32767  # 16-bit
    digital_range = digital_max - digital_min

    if np.isclose(signal_min, signal_max) and not np.isclose(signal_min, 0.0):
        # Constant non-zero signal: a 1% margin on each side keeps a valid,
        # signal-scaled range that brackets the value.
        margin = abs(signal_min) * 0.01
        phys_min = _fit_physical_bound(signal_min - margin, round_up=False)
        phys_max = _fit_physical_bound(signal_max + margin, round_up=True)
    else:
        # Any other window, including a constant zero channel and a channel whose
        # values all lie within 1e-8 of zero (float residue, or Tesla-scale MEG): the
        # smallest range the 8-character header text can hold around it. Bounds that
        # fit are 0 and 1e-6 above zero and -1e-5 below it ('-0.000001' needs nine
        # characters and would be stored as -0.00000), so the window is [0, 1e-6]
        # for non-negative data, [-1e-5, 0] for non-positive data and [-1e-5, 1e-6]
        # across zero.
        phys_min = _fit_physical_bound(signal_min, round_up=False)
        phys_max = _fit_physical_bound(signal_max, round_up=True)

    # Guarantee a strictly positive physical range (pyedflib requires
    # physical_min != physical_max), widening outward if rounding collapsed it.
    if not phys_min < phys_max:
        bump = max(abs(phys_min), abs(phys_max), 1e-6) * 1e-3
        phys_max = _fit_physical_bound(phys_min + bump, round_up=True)
        if not phys_min < phys_max:
            phys_max = phys_min + bump

    scaling_factor = (digital_range - 1) / (phys_max - phys_min)
    return phys_min, phys_max, digital_min, digital_max, scaling_factor


# EDF/BDF header layout (https://www.edfplus.info/specs/edf.html): 256 fixed bytes
# whose last 4 hold the number of signals, then one block per field, each with one
# entry per signal: label 16, transducer 80, physical dimension 8, physical minimum
# 8, physical maximum 8, ... The same layout holds for BDF.
_FIXED_HEADER_BYTES = 256
_SIGNAL_COUNT_AT = 252
_PHYS_MIN_BLOCK_AT = 16 + 80 + 8  # bytes of the fields before it, per signal
_PHYS_MAX_BLOCK_AT = _PHYS_MIN_BLOCK_AT + _PHYS_FIELD_CHARS

# Rounds of widening a bound that still reads back inside its window: round k moves
# it 2**k units in the last place outward, so a binary expansion just below the
# decimal (one step) and a text cut off a digit late (a few) are both absorbed.
_MAX_BOUND_ROUNDS = 8


def _read_header_bounds(path: str, n_signals: int) -> list[tuple[float, float]]:
    """Physical minimum and maximum of the first ``n_signals`` signals, as stored.

    Parses the header text, which is what every reader scales by (pyedflib and MNE
    read these fields as decimal text), not the doubles the writer was given. The text
    is parsed with Python's ``float()``; pyedflib parses the same text with edflib's own
    parser, which can land one unit in the last place away (it reads ``2.53135`` as
    ``2.5313499999999998``), so a bound read through pyedflib may differ from the one
    returned here by an ulp, which is far below an LSB.
    """
    with open(path, "rb") as fh:
        n_total = int(fh.read(_FIXED_HEADER_BYTES)[_SIGNAL_COUNT_AT:])
        fields = fh.read(n_total * (_PHYS_MAX_BLOCK_AT + _PHYS_FIELD_CHARS))

    def field(block_at: int, i: int) -> float:
        start = n_total * block_at + i * _PHYS_FIELD_CHARS
        return float(fields[start : start + _PHYS_FIELD_CHARS].decode("ascii"))

    return [(field(_PHYS_MIN_BLOCK_AT, i), field(_PHYS_MAX_BLOCK_AT, i)) for i in range(n_signals)]


def _verify_header_bounds(path: str, expected: list[tuple[float, float]]) -> None:
    """Raise ``ValueError`` unless the header of ``path`` stores exactly ``expected``.

    The samples are rounded against the bounds read back from a scratch header; a file
    whose own header says otherwise would read back miscalibrated, so it is refused.
    """
    stored = _read_header_bounds(path, len(expected))
    if stored != expected:
        raise ValueError(
            f"The header of {path} stores physical bounds {stored} other than the ones "
            f"its samples were scaled by, {expected}; the file would read back "
            "miscalibrated."
        )


@contextlib.contextmanager
def _quiet_bound_warnings():
    """Silence pyedflib's "will be truncated" warning about long physical bounds.

    pyedflib warns when ``str()`` of a bound exceeds 8 characters. A bound widened
    by :func:`_settle_physical_bounds` always does (``95382.65000000001`` is stored
    as ``95382.65``); what the header really holds is checked by reading it back.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Physical m(?:inimum|aximum) for channel", category=UserWarning
        )
        yield


def _stored_physical_bounds(
    channels: list[dict], filetype: int, scratch_dir: str
) -> list[tuple[float, float]]:
    """The physical bounds a header built from ``channels`` will hold.

    edflib writes each bound as fixed-notation text cut off at 8 characters, from its
    binary expansion, so the text is not always the decimal that was meant: 95382.65
    is stored as ``95382.64``, and -53820.7 as ``-53820.6``. The header is written to a
    scratch file (no samples) and parsed back, so the answer is what edflib does
    rather than a model of it. The scratch file lives in ``scratch_dir``, which the
    export can write to anyway (the system temporary directory may not be writable),
    and is removed before returning or raising.
    """
    fd, path = tempfile.mkstemp(prefix=".biosigio-bounds-", dir=scratch_dir)
    os.close(fd)
    try:
        writer = pyedflib.EdfWriter(path, len(channels), file_type=filetype)
        try:
            with _quiet_bound_warnings():
                writer.setSignalHeaders(channels)
        finally:
            writer.close()
        return _read_header_bounds(path, len(channels))
    finally:
        os.unlink(path)


def _step_outward(value: float, toward: float, ulps: int) -> float:
    """``value`` moved ``ulps`` representable doubles toward ``toward``."""
    for _ in range(ulps):
        value = float(np.nextafter(value, toward))
    return value


def _settle_physical_bounds(
    channels: list[dict], windows: list[tuple[float, float]], filetype: int, scratch_dir: str
) -> list[tuple[float, float]]:
    """Make the bounds a header stores bracket each channel's data window.

    A reader scales the digital range onto the STORED bounds, so a stored bound
    inside the window saturates the extreme samples and shifts the scale of the
    whole channel (more than ten LSB at the extremes). Starting from the outward
    rounded bounds in ``channels`` (``physical_min``/``physical_max``, updated in
    place), the header is written to a scratch file in ``scratch_dir`` and read back
    (:func:`_stored_physical_bounds`); every bound that reads back inside its window
    is moved outward by growing steps in the last place and checked again.

    Returns:
        The ``(physical_min, physical_max)`` each channel's header will hold.

    Raises:
        ValueError: If a stored bound still sits inside its window after
            ``_MAX_BOUND_ROUNDS`` rounds, instead of writing a miscalibrated file.
    """
    if not channels:
        return []
    attempt = 0
    while True:
        stored = _stored_physical_bounds(channels, filetype, scratch_dir)
        low_short = [s_lo > w_lo for (s_lo, _), (w_lo, _) in zip(stored, windows, strict=True)]
        high_short = [s_hi < w_hi for (_, s_hi), (_, w_hi) in zip(stored, windows, strict=True)]
        flat = [s_lo >= s_hi for s_lo, s_hi in stored]
        failing = [i for i in range(len(channels)) if low_short[i] or high_short[i] or flat[i]]
        if not failing:
            return stored
        if attempt == _MAX_BOUND_ROUNDS:
            details = "; ".join(
                f"{channels[i]['label']!r}: header stores [{stored[i][0]}, {stored[i][1]}] "
                f"for data in [{windows[i][0]}, {windows[i][1]}]"
                for i in failing
            )
            raise ValueError(
                "The EDF/BDF header cannot store physical bounds that bracket the data "
                f"window ({details}). Rescale the channel to a unit that suits its magnitude."
            )
        for i in failing:
            ch = channels[i]
            if low_short[i] or flat[i]:
                ch["physical_min"] = _step_outward(float(ch["physical_min"]), -np.inf, 2**attempt)
            if high_short[i] or flat[i]:
                ch["physical_max"] = _step_outward(float(ch["physical_max"]), np.inf, 2**attempt)
        attempt += 1


def _quantize_to_digital(
    values: np.ndarray, phys_min: float, phys_max: float, dig_min: int, dig_max: int
) -> np.ndarray:
    """Round physical ``values`` to the nearest digital level of the stored scale.

    A reader maps digital ``d`` back to ``phys_min + (d - dig_min) * lsb`` with
    ``lsb = (phys_max - phys_min) / (dig_max - dig_min)``, from the bounds the header
    STORES (pass those, not the doubles the writer was given). Rounding to the nearest
    level keeps every value that lies within the bounds within half an LSB of itself;
    pyedflib's own conversion truncates toward zero, which leaves up to one LSB. A
    value outside the bounds saturates at the end of the digital range.
    """
    lsb = (phys_max - phys_min) / (dig_max - dig_min)
    levels = np.rint(dig_min + (np.clip(values, phys_min, phys_max) - phys_min) / lsb)
    return np.clip(levels, dig_min, dig_max).astype(np.int32)


# A channel whose data span covers fewer stored LSB than this keeps too few digital
# levels to tell its samples apart.
_MIN_RESOLVED_LSB = 4
# Channels named in the warning about such channels; the rest are counted.
_MAX_NAMED_CHANNELS = 5


class _Unresolved(NamedTuple):
    """A channel too narrow for its stored scale, as reported by the export warning."""

    name: str
    unit: str
    span: float
    lsb: float
    offset: float | None  # the window's midpoint when subtracting it would resolve the channel


def _unresolved_channels(
    names: list[str],
    channels: list[dict],
    windows: list[tuple[float, float]],
    stored: list[tuple[float, float]],
) -> list[_Unresolved]:
    """Channels whose data window spans fewer than ``_MIN_RESOLVED_LSB`` stored LSB.

    The header can only hold bounds a few digits wide, so a window that is narrow
    against its own magnitude gets bounds far wider than the data. Two things cause it,
    with different remedies: a signal far below 1e-5 in its unit (MEG in tesla; nothing
    narrower than ``[0, 1e-6]`` around zero can be stored) needs a smaller unit, and a
    small variation on a large offset (410986 mV, give or take 5e-6) needs the offset
    subtracted. ``offset`` is set when the same span centred on zero would be resolved,
    and is ``None`` when it would not.

    A constant channel has no variation to lose and is not listed; a channel that varies
    by a float residue (values of 0 and 1e-12, say) does lose it and is.

    Args:
        names: The recording's channel names, which the warning uses (not the EDF labels,
            which are cut to 16 characters and may be numbered).
        channels: The signal headers, in the same order.
        windows: Each channel's data window.
        stored: The bounds each header will hold.

    Returns:
        One entry per such channel, in channel order.
    """
    out = []
    for name, ch, (w_lo, w_hi), (s_lo, s_hi) in zip(names, channels, windows, stored, strict=True):
        digital_range = ch["digital_max"] - ch["digital_min"]
        lsb = (s_hi - s_lo) / digital_range
        span = w_hi - w_lo
        if not 0 < span < _MIN_RESOLVED_LSB * lsb:
            continue
        centred_low = _fit_physical_bound(-span / 2, round_up=False)
        centred_high = _fit_physical_bound(span / 2, round_up=True)
        centred_lsb = (centred_high - centred_low) / digital_range
        offset = (w_lo + w_hi) / 2 if span >= _MIN_RESOLVED_LSB * centred_lsb else None
        out.append(_Unresolved(name, str(ch["dimension"]), span, lsb, offset))
    return out


def _unresolved_warning(unresolved: list[_Unresolved]) -> str:
    """The warning text for :func:`_unresolved_channels`, naming up to five channels."""
    entries = []
    for name, unit, span, lsb, offset in unresolved[:_MAX_NAMED_CHANNELS]:
        entry = f"{name!r} spans {span:.3g} {unit} = {span / lsb:.2g} LSB (LSB {lsb:.3g} {unit})"
        if offset is None:
            entry += f": rescale it to a smaller unit than {unit}"
        else:
            entry += (
                f": its variation is small against its offset (about {offset:.6g} {unit}), "
                "so subtract the offset before export"
            )
        entries.append(entry)
    named = "; ".join(entries)
    more = len(unresolved) - _MAX_NAMED_CHANNELS
    if more > 0:
        named += f"; and {more} more"
    return (
        f"{len(unresolved)} channel(s) span fewer than {_MIN_RESOLVED_LSB} stored LSB, so "
        f"their samples fall on a few digital levels or one: {named}. The header holds each "
        "physical bound as 8 characters, which limits how narrow a range it can store."
    )


def _pad_to_whole_records(levels: np.ndarray, samples_per_record: int, fill: int) -> np.ndarray:
    """``levels`` extended with ``fill`` to a whole number of data records.

    EDF/BDF hold whole data records, so a recording that ends inside one is padded.
    pyedflib pads digital samples with digital 0, the middle of the window, which a
    reader sees as a mid-scale value that was never recorded. The caller passes the
    level of physical 0 instead, so the padding reads back as 0 (the nearest bound when
    0 lies outside the window), as it did when pyedflib converted physical samples.
    """
    pad = -len(levels) % samples_per_record
    if pad == 0:
        return levels
    return np.concatenate([levels, np.full(pad, fill, dtype=levels.dtype)])


def summarize_channels(channels: dict, analyses: dict) -> str:
    """
    Generate a summary of channel characteristics grouped by type.

    Args:
        channels: Dictionary of channel information
        analyses: Dictionary of signal analyses

    Returns:
        str: Formatted summary string
    """
    # Group channels by type
    type_groups = {}
    for ch_name, ch_info in channels.items():
        ch_type = ch_info.get("channel_type", "Unknown")
        if ch_type not in type_groups:
            type_groups[ch_type] = {
                "channels": [],
                "ranges": [],
                "dynamic_ranges": [],
                "snrs": [],
                "formats": [],
                "unit": ch_info.get("physical_dimension", "Unknown"),
            }
        type_groups[ch_type]["channels"].append(ch_name)

        analysis = analyses.get(ch_name, {})
        if not analysis.get("is_zero", False):
            type_groups[ch_type]["ranges"].append(analysis.get("range", 0))
            type_groups[ch_type]["dynamic_ranges"].append(analysis.get("dynamic_range_db", 0))
            type_groups[ch_type]["snrs"].append(analysis.get("snr_db", 0))
            type_groups[ch_type]["formats"].append(
                "BDF" if analysis.get("use_bdf", False) else "EDF"
            )

    # Generate summary
    summary = []
    for ch_type, data in type_groups.items():
        ranges = np.array(data["ranges"])
        dynamic_ranges = np.array(data["dynamic_ranges"])
        snrs = np.array(data["snrs"])
        formats = data["formats"]

        if len(ranges) > 0:
            summary.append(f"\nChannel Type: {ch_type} ({len(data['channels'])} channels)")
            summary.append(
                f"Range: {np.min(ranges):.2f} to {np.max(ranges):.2f} "
                f"(mean: {np.mean(ranges):.2f}) {data['unit']}"
            )
            summary.append(
                f"Dynamic Range: {np.min(dynamic_ranges):.1f} to "
                f"{np.max(dynamic_ranges):.1f} (mean: {np.mean(dynamic_ranges):.1f}) dB"
            )
            summary.append(
                f"SNR: {np.min(snrs):.1f} to {np.max(snrs):.1f} (mean: {np.mean(snrs):.1f}) dB"
            )

            edf_count = formats.count("EDF")
            bdf_count = formats.count("BDF")
            summary.append(
                f"Format: {edf_count} channels using EDF, {bdf_count} channels using BDF"
            )
        else:
            summary.append(f"\nChannel Type: {ch_type} ({len(data['channels'])} channels)")
            summary.append("All channels contain zero signal")

    return "\n".join(summary)


def _resolve_start_datetime(rec: Recording, start_datetime=None):
    """Decide the start date/time the EDF/BDF header will carry.

    pyedflib's ``EdfWriter`` defaults the header start to ``datetime.now()``,
    so two exports of the same recording differed in the header and outputs
    were not reproducible, and the conversion date sat in a field readers take
    as the acquisition time. BIDS expects EDF headers anonymized and the
    acquisition time in ``scans.tsv``.

    Resolution order:

    1. an explicit ``start_datetime`` given to :meth:`EDFExporter.export`;
    2. ``rec.metadata["startdate"]`` when the importer filled it (a
       :class:`~datetime.datetime`, or an ISO-format string);
    3. otherwise the anonymized ``01.01.85 00.00.00`` the EDF specification
       defines for an unknown start date.
    """
    if start_datetime is not None:
        if not isinstance(start_datetime, datetime.datetime):
            raise ValueError(
                "start_datetime must be a datetime.datetime, got "
                f"{type(start_datetime).__name__}"
            )
        return start_datetime
    value = rec.metadata.get("startdate")
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.datetime(1985, 1, 1)


class EDFExporter:
    """Exporter for EDF format with channels.tsv generation."""

    @staticmethod
    def export(
        rec: Recording,
        filepath: str,
        precision_threshold: float = 0.01,
        method: str = "both",
        fft_noise_range: tuple | None = None,
        svd_rank: int | None = None,
        format: Literal["auto", "edf", "bdf"] = "auto",
        bypass_analysis: bool = False,
        events_df: pd.DataFrame | None = None,
        create_channels_tsv: bool = True,
        start_datetime: datetime.datetime | None = None,
        clip_outliers: bool | str = "auto",
        outlier_sigmas: float = 8.0,
        min_effective_bits: float = 10.0,
        **kwargs,
    ) -> str:
        """
        Export EMG data to EDF/BDF format with optional BIDS-compliant channels.tsv file.

        Every channel gets its own physical window. Each sample is rounded to the nearest
        digital level of the scale the header stores, so the file reads back within half a
        least significant bit (LSB) of the given values, with the LSB
        ``(physical_max - physical_min) / (digital_max - digital_min)`` taken from the
        header as stored. The header keeps each bound as 8 characters of fixed-notation
        text, so the exporter reads it back and widens a bound that is stored inside the
        data window. Samples outside the window (see ``clip_outliers``) saturate at its
        ends.

        Args:
            rec: Recording object containing the data
            filepath: Path to save the EDF/BDF file
            precision_threshold: Maximum acceptable precision loss percentage (default: 0.01%)
            method: Method for signal analysis ('svd', 'fft', or 'both')
                'svd': Uses Singular Value Decomposition for noise floor estimation
                'fft': Uses Fast Fourier Transform for noise floor estimation
                'both': Uses both methods and takes the minimum noise floor (default)
            fft_noise_range: Optional tuple (min_freq, max_freq) specifying frequency range for noise in FFT method
            svd_rank: Optional manual rank cutoff for signal/noise separation in SVD method
            format: Format to use ('auto', 'edf', or 'bdf'). Default is 'auto'.
                    If 'edf' or 'bdf' is specified, that format will be used directly.
                    If 'auto', the format (EDF/16-bit or BDF/24-bit) is chosen based
                    on signal analysis to minimize precision loss while preferring EDF
                    if sufficient.
            bypass_analysis: If True, skip the signal analysis step. Requires format
                             to be explicitly set to 'edf' or 'bdf'. (default: False)
            events_df: Optional DataFrame containing events/annotations to write.
                     Columns should include 'onset', 'duration', 'description'.
                     If None or empty, no annotations are written.
            create_channels_tsv: If True, create a BIDS-compliant channels.tsv file (default: True)
            start_datetime: Explicit start date/time for the EDF/BDF header. When
                None (default), the header start comes from
                ``rec.metadata["startdate"]`` if the importer filled it, else the
                anonymized ``01.01.85 00.00.00`` the EDF specification defines for
                an unknown start. Without this, pyedflib writes ``datetime.now()``
                into the header, so two exports of the same recording differ (#174).
            clip_outliers: Singularity handling for the per-channel physical window.
                'auto' (default): keep the full data range losslessly, but clip rare
                extreme outliers to a robust percentile window ONLY when keeping them
                would drop the bulk signal below ``min_effective_bits`` of resolution
                at the chosen format (a loud warning reports what was clipped). True:
                always clip to the robust window. False: never clip (full range,
                lossless even if a single outlier starves the bulk of resolution).
            outlier_sigmas: Robust z-score threshold for the inlier window: samples
                within ``outlier_sigmas`` x 1.4826 x median-absolute-deviation of the
                median are inliers; the window is their min/max so only genuine
                singularities are clipped (default 8.0).
            min_effective_bits: Resolution floor (in bits) the bulk signal must keep
                under 'auto'; outliers are clipped only if the full range would push
                it below this (default 10.0, ~60 dB). Kept low so BDF's 24-bit range
                preserves moderate outliers losslessly and only true singularities
                (or forced/low-res EDF) trigger clipping.
            **kwargs: Additional arguments for the exporter
        """
        if rec.signals is None:
            raise ValueError("No signals to export")

        print("\nSignal Analysis:")
        print("--------------")

        # Initialize format decision variables
        use_bdf = False
        bdf_reason = ""

        # --- Format Decision and Bypass Check ---
        if bypass_analysis and format.lower() == "auto":
            raise ValueError("Cannot bypass analysis when format is set to 'auto'.")

        if format.lower() == "bdf":
            use_bdf = True
            if not bypass_analysis:
                print("\nUser specified BDF format (24-bit).")
            else:
                # Log critical only if bypassing, already logged in Recording.to_edf
                pass  # logging.log(logging.CRITICAL, "Skipping analysis, using specified BDF format.")
        elif format.lower() == "edf":
            use_bdf = False
            if not bypass_analysis:
                print("\nUser specified EDF format (16-bit).")
            else:
                # Log critical only if bypassing, already logged in Recording.to_edf
                pass  # logging.log(logging.CRITICAL, "Skipping analysis, using specified EDF format.")
        elif format.lower() != "auto":
            warnings.warn(
                f"Unknown format: {format}. Valid options are 'auto', 'edf', or 'bdf'. Using 'auto'.",
                stacklevel=2,
            )
            format = "auto"  # Default to auto if invalid format given
            bypass_analysis = False  # Cannot bypass if format is auto

        signal_analyses = {}
        signal_info_strings = []

        # --- Conditional Signal Analysis ---
        if not bypass_analysis:
            # Analyze signals (needed for summary and potentially for 'auto' format decision)
            for ch_name in rec.channels:
                signal = rec.signals[ch_name].values
                ch_info = rec.channels[ch_name]

                # Analyze signal characteristics
                analysis = analyze_signal(
                    signal, method=method, fft_noise_range=fft_noise_range, svd_rank=svd_rank
                )
                recommend_bdf, reason, snr = determine_format_suitability(signal, analysis)
                analysis["snr"] = snr
                analysis["recommend_bdf"] = recommend_bdf
                analysis["reason"] = reason
                signal_analyses[ch_name] = analysis  # Store analysis for later summary

                # If format is 'auto', check if any channel recommends BDF
                if format == "auto" and recommend_bdf:
                    use_bdf = True  # Switch to BDF if any channel needs it
                    if not bdf_reason:  # Capture the first reason
                        bdf_reason = f"Channel '{ch_name}': {reason}"

                # Prepare info string for printing later
                signal_info_strings.append(
                    f"\n  {ch_name}:"
                    f"\n    Range: {analysis['range']:.8g} {ch_info['physical_dimension']}"
                    f"\n    Dynamic Range: {analysis['dynamic_range_db']:.1f} dB"
                    f"\n    Noise Floor: {analysis['noise_floor']:.2e} {ch_info['physical_dimension']}"
                    f"\n    SNR: {snr:.1f} dB"
                    f"\n    Method: {analysis.get('method', 'svd')}"
                    f"\n    Recommended Format: {'BDF' if recommend_bdf else 'EDF'} ({reason})"
                )

            # Print analysis details after deciding the format
            for info_str in signal_info_strings:
                print(info_str)

            # Final format decision message for 'auto' mode
            if format == "auto":
                if use_bdf:
                    print(
                        "\nUsing BDF format (24-bit) based on signal analysis to preserve precision."
                    )
                    print(f"Reason: {bdf_reason}")
                    warnings.warn(
                        f"Using BDF format based on signal analysis. Reason: {bdf_reason}",
                        stacklevel=2,
                    )
                else:
                    print(
                        "\nUsing EDF format (16-bit) based on signal analysis (precision within acceptable range)."
                    )
        # else: # bypass_analysis is True - logging handled in Recording.to_edf
        #     pass # logging.log(logging.CRITICAL, "Signal analysis bypassed.")

        # Set file format and create writer
        # Initialize BIDS-compliant channels.tsv data structure
        # Required columns in BIDS order: name, type, units
        channels_tsv_data = {
            "name": [],
            "type": [],
            "units": [],
            "sampling_frequency": [],
            "reference": [],
            "status": [],
        }
        channel_info_list: list[dict[str, Any]] = []
        data_windows: list[tuple[float, float]] = []

        # EDF/BDF export requires a single sampling rate across channels: biosigio stores
        # all channels on one uniform-length grid, and pyedflib's writeSamples produces
        # an unreadable file when per-channel record counts differ. Fail loudly instead
        # of writing a corrupt file; mixed-rate sources (e.g. Trigno EMG + ACC) must be
        # resampled to a common rate before export.
        distinct_rates = {int(rec.channels[ch]["sample_frequency"]) for ch in rec.channels}
        if len(distinct_rates) > 1:
            raise ValueError(
                "EDF/BDF export requires a single sampling rate across all channels, but "
                f"multiple were found: {sorted(distinct_rates)} Hz. Resample channels to a "
                "common rate before exporting."
            )

        if use_bdf:
            filepath = os.path.splitext(filepath)[0] + ".bdf"
            filetype = pyedflib.FILETYPE_BDFPLUS
        else:
            filepath = os.path.splitext(filepath)[0] + ".edf"
            filetype = pyedflib.FILETYPE_EDFPLUS

        writer = pyedflib.EdfWriter(filepath, len(rec.channels), file_type=filetype)

        try:
            # The header's start date/time must not be the wall clock at export
            # time: same recording in, same bytes out. BIDS expects anonymized
            # headers; the explicit argument lets a caller set the acquisition
            # start when the recording genuinely knows it (#174).
            writer.setStartdatetime(_resolve_start_datetime(rec, start_datetime))
            # MEMORY OPTIMIZATION: Two-pass approach to avoid holding all signals in memory
            edf_labels, label_renames = _edf_signal_labels(list(rec.channels))
            if label_renames:
                warnings.warn(
                    f"{len(label_renames)} channel label(s) are not unique once truncated to "
                    f"{_LABEL_FIELD_CHARS} characters for EDF/BDF; writing them as "
                    + ", ".join(f"{new!r} ({old!r})" for new, old in label_renames.items()),
                    stacklevel=2,
                )

            # Pass 1: Collect headers only (compute min/max without copying data)
            for ch_index, ch_name in enumerate(rec.channels):
                signal = rec.signals[ch_name].values
                ch_info = rec.channels[ch_name]

                # Resolve the physical window the bounds will bracket. Handle the
                # empty/all-NaN edge case, then choose the window (full range, or a
                # robust window when 'auto'/True clips genuine singularities).
                if signal.size == 0 or np.all(np.isnan(signal)):
                    warnings.warn(
                        f"Channel '{ch_name}' has an empty or all-NaN signal. "
                        "Using default min/max of 0.0 for scaling.",
                        stacklevel=2,
                    )
                    win_lo, win_hi, n_clipped, max_excursion = 0.0, 0.0, 0, 0.0
                else:
                    win_lo, win_hi, n_clipped, max_excursion = _resolve_physical_window(
                        signal, use_bdf, clip_outliers, outlier_sigmas, min_effective_bits
                    )
                    if n_clipped > 0:
                        unit = ch_info["physical_dimension"]
                        warnings.warn(
                            f"Channel '{ch_name}': {n_clipped} outlier sample(s) "
                            f"({100.0 * n_clipped / signal.size:.4f}%) will saturate to the "
                            f"robust window [{win_lo:.6g}, {win_hi:.6g}] {unit}, preserving "
                            f"{24 if use_bdf else 16}-bit resolution for the bulk signal "
                            f"(max excursion {max_excursion:.6g} {unit}). "
                            "Pass clip_outliers=False to keep the full range instead.",
                            stacklevel=2,
                        )

                data_windows.append((float(win_lo), float(win_hi)))

                # Calculate scaling factors for header based on the chosen format (use_bdf).
                # physical_min/max are rounded outward to bracket the window, so pyedflib
                # never silently clips the bulk signal (issue #61).
                # scaling_factor is informational (pyedflib derives its own from the
                # physical/digital ranges); the bounds are what matter for fidelity.
                phys_min, phys_max, dig_min, dig_max, _scaling = _determine_scaling_factors(
                    win_lo, win_hi, use_bdf=use_bdf
                )

                # EDF/BDF store physical_min/max as 8-char ASCII. A magnitude needing
                # more digits (|value| >= 1e8, or >= 1e7 once a sign is added) cannot be
                # represented: pyedflib would truncate it and silently scale the channel
                # by powers of ten (reintroducing the #61 corruption) or abort the write.
                # Fail loudly here instead, before any bytes are written, with the only
                # real remedy - rescale the channel to a coarser unit.
                if (
                    len(_fixed_text(phys_min)) > _PHYS_FIELD_CHARS
                    or len(_fixed_text(phys_max)) > _PHYS_FIELD_CHARS
                ):
                    raise ValueError(
                        f"Channel '{ch_name}': physical range [{phys_min}, {phys_max}] "
                        f"{ch_info['physical_dimension']} needs more than {_PHYS_FIELD_CHARS} "
                        "characters and cannot be stored in the EDF/BDF header without corrupting "
                        "the values. Rescale this channel to a coarser unit (e.g. uV -> mV -> V) "
                        "so the magnitude fits."
                    )

                # Prepare channel header dictionary
                ch_dict = {
                    "label": edf_labels[ch_index],  # fits the 16-char field, unique
                    "dimension": ch_info["physical_dimension"],
                    "sample_frequency": int(ch_info["sample_frequency"]),
                    "physical_max": phys_max,
                    "physical_min": phys_min,
                    "digital_max": dig_max,
                    "digital_min": dig_min,
                    "prefilter": ch_info["prefilter"],
                    "transducer": f"{ch_info.get('channel_type', 'Unknown')} sensor",
                }
                channel_info_list.append(ch_dict)

                # Add to BIDS-compliant channels.tsv data. A row keeps the
                # channel's full label, as released, even when the file stores it
                # truncated; only a channel whose truncated label collided, and
                # so was numbered, is named by the label written to the file,
                # because its full label no longer tells its row apart from the
                # other channels that share the truncated stem.
                written = edf_labels[ch_index]
                channels_tsv_data["name"].append(
                    written.rstrip() if written in label_renames else ch_name
                )

                # Channels carry a validated channel_type from the modality
                # vocabulary, so use it directly for channels.tsv. This preserves
                # genuine BIDS types (EEG/SEEG/ECOG/...) instead of flattening
                # everything but a short whitelist to MISC.
                bids_type = to_bids_channels_tsv_type(ch_info.get("channel_type", "OTHER"))
                channels_tsv_data["type"].append(bids_type)
                channels_tsv_data["units"].append(ch_info["physical_dimension"])
                channels_tsv_data["sampling_frequency"].append(ch_info["sample_frequency"])
                channels_tsv_data["reference"].append("n/a")
                channels_tsv_data["status"].append("good")

            # The header keeps each bound as 8 characters of text, cut off from its
            # binary expansion, so a bound can be stored INSIDE the window it was
            # rounded to bracket (issue #170). Read the text back and widen any such
            # bound before the writer sees it.
            stored_bounds = _settle_physical_bounds(
                channel_info_list,
                data_windows,
                filetype,
                scratch_dir=os.path.dirname(os.path.abspath(filepath)),
            )
            unresolved = _unresolved_channels(
                list(rec.channels), channel_info_list, data_windows, stored_bounds
            )
            if unresolved:
                warnings.warn(_unresolved_warning(unresolved), stacklevel=2)

            # Set all headers before writing
            with _quiet_bound_warnings():
                writer.setSignalHeaders(channel_info_list)

            # Pass 2: Write every data record for all signals at once.
            # pyedflib's writePhysicalSamples() writes exactly ONE data record
            # (sample_frequency samples) per call, so calling it once per channel with
            # the full array silently truncated every export to a single record
            # (one second). writeSamples() emits all records for every signal.
            # The samples go in as digital levels: pyedflib's own physical-to-digital
            # conversion truncates toward zero (up to one LSB of error) and scales by
            # the bounds as given, not as the header stores them (issue #170). Rounding
            # to the nearest level against the stored bounds keeps every value within
            # half an LSB of the scale a reader will use.
            # IMPORTANT: order must match setSignalHeaders; rec.channels preserves
            # insertion order (Python 3.7+) and both passes iterate it.
            signals_to_write = []
            for i, (ch_name, ch_dict, (phys_min, phys_max)) in enumerate(
                zip(rec.channels, channel_info_list, stored_bounds, strict=True)
            ):
                levels = _quantize_to_digital(
                    np.nan_to_num(rec.signals[ch_name].values, nan=0.0).astype(
                        np.float64, copy=False
                    ),
                    phys_min,
                    phys_max,
                    ch_dict["digital_min"],
                    ch_dict["digital_max"],
                )
                # A recording that ends inside a data record is padded to the end of
                # it; the padding is physical 0, not pyedflib's digital 0 (mid-window).
                zero_level = _quantize_to_digital(
                    np.zeros(1),
                    phys_min,
                    phys_max,
                    ch_dict["digital_min"],
                    ch_dict["digital_max"],
                )[0]
                signals_to_write.append(
                    _pad_to_whole_records(levels, writer.get_smp_per_record(i), int(zero_level))
                )
            writer.writeSamples(signals_to_write, digital=True)

            # Write annotations if provided
            if events_df is not None and not events_df.empty:
                for _index, row in events_df.iterrows():
                    try:
                        # pyedflib uses onset, duration, description
                        onset = float(row["onset"])
                        duration = float(row["duration"])
                        description = str(row["description"])
                        # Write annotation for all channels (-1)
                        writer.writeAnnotation(onset, duration, description)
                    except KeyError as e:
                        warnings.warn(
                            f"Skipping event due to missing column: {e}. Event data: {row}",
                            stacklevel=2,
                        )
                    except (TypeError, ValueError) as e:
                        warnings.warn(
                            f"Skipping event due to invalid data type: {e}. Event data: {row}",
                            stacklevel=2,
                        )

            # Explicitly flush and close the writer to ensure all data is written
            writer.close()

            # The samples were rounded against the bounds read back from a scratch
            # header; refuse a file whose own header says otherwise.
            _verify_header_bounds(filepath, stored_bounds)

            # Wait a moment to ensure file system operations are complete
            import time

            time.sleep(0.1)

            # Verify the file exists and has the correct size
            if not os.path.exists(filepath):
                raise OSError(f"File {filepath} was not created")

            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise OSError(f"File {filepath} was created but is empty")

            # Generate BIDS-compliant channels.tsv file if requested
            if create_channels_tsv:
                channels_tsv_path = os.path.splitext(filepath)[0] + "_channels.tsv"
                # Create DataFrame with columns in BIDS-specified order
                # Required columns first: name, type, units
                # Then optional columns in the order they appear in data
                ordered_columns = ["name", "type", "units"]
                optional_columns = [
                    col for col in channels_tsv_data.keys() if col not in ordered_columns
                ]
                column_order = ordered_columns + optional_columns

                channels_df = pd.DataFrame(channels_tsv_data)
                channels_df = channels_df[column_order]
                channels_df.to_csv(channels_tsv_path, sep="\t", index=False, na_rep="n/a")
                print(f"\nBIDS-compliant channels metadata saved to: {channels_tsv_path}")

            # Print summary using stored analyses, only if analysis was performed
            if not bypass_analysis:
                # We need to adapt summarize_channels call slightly or assume it uses the analyses dict
                # Let's refine the analyses dict passed to summarize_channels
                summary_analyses = {}
                for ch_name, analysis in signal_analyses.items():
                    summary_analyses[ch_name] = {
                        "range": analysis["range"],
                        "dynamic_range_db": analysis["dynamic_range_db"],
                        "snr_db": analysis["snr"],
                        "use_bdf": use_bdf,  # Use the final decision for the whole file
                    }

                summary = summarize_channels(cast(dict, rec.channels), summary_analyses)
                print("\nSummary:")
                print(summary)
            else:
                print("\nSummary skipped as signal analysis was bypassed.")

            print(f"\nEMG data exported to: {filepath}")
            return filepath
        except Exception as e:
            # The writer is closed unconditionally in the finally block below; here
            # we only remove the partially written file so a failed export leaves no
            # corrupt output behind.
            import time

            time.sleep(0.1)

            if "filepath" in locals() and os.path.exists(filepath):
                try:
                    os.unlink(filepath)
                    print(f"Cleaned up partially written file: {filepath}")
                except Exception as unlink_e:
                    print(f"Error during cleanup of {filepath}: {unlink_e}")

            raise e
        finally:
            if writer is not None:
                writer.close()  # Ensure writer is closed
