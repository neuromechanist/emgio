"""Physical-unit parsing and conversion for channel ``physical_dimension`` strings.

A channel carries its values and its unit label as two independent pieces of
metadata, and nothing in the data model forces them to agree. They must, so any
code that adopts a unit from an external source (a BIDS ``channels.tsv``, a
user-supplied override) has to convert the samples at the same moment it sets
the label. :func:`conversion_factor` is the arithmetic behind that; see
:func:`biosigio.bids.apply_channels_tsv` for the caller that motivated it
(issue #122).

Every unit here is a decimal-prefixed base symbol, so a unit is fully described
by ``(base symbol, decimal exponent)`` and a conversion is ``10 ** (from - to)``.
Keeping the exponent an ``int`` rather than a float multiplier matters: ``1e-6``
is not exactly representable in binary, so ``1.0 / 1e-6`` is
``1000000.0000000001`` while ``10.0 ** 6`` is exactly ``1000000.0``. A volts ->
microvolts rescale therefore introduces no error of its own.

Parsing is deliberately **case-sensitive**. ``m`` (milli) and ``M`` (mega) differ
by 10^9, and a lenient parser that treated ``MV`` as millivolts would turn a
label mismatch into a silent numeric one -- the exact failure mode this module
exists to prevent. An unrecognized spelling returns ``None`` (not convertible)
so the caller can keep the importer's values and warn, which is always safe.
"""

from __future__ import annotations

# Decimal SI prefixes, as exponents of ten. Case-sensitive: 'm' is milli and 'M'
# is mega. The micro sign has three spellings in the wild -- MICRO SIGN U+00B5,
# GREEK SMALL LETTER MU U+03BC, and the ASCII fallback 'u' -- and BIDS
# channels.tsv files use all three, so all three are accepted.
_PREFIX_EXPONENTS: dict[str, int] = {
    "f": -15,
    "p": -12,
    "n": -9,
    "µ": -6,  # MICRO SIGN
    "μ": -6,  # GREEK SMALL LETTER MU
    "u": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "": 0,
    "k": 3,
    "M": 6,
    "G": 9,
}

# Base symbol -> (canonical quantity symbol, exponent relative to that quantity).
# Two symbols share a quantity only when a conversion between them is exact and
# unambiguous: T/cm is 10^2 T/m, so an MEG gradiometer in fT/cm converts to T/m.
# Quantities that merely look related (deg/s vs rad/s) are kept distinct, so a
# sidecar claiming one over the other is reported as non-convertible rather than
# rescaled by a factor this module would have to guess.
_BASE_UNITS: dict[str, tuple[str, int]] = {
    "V": ("V", 0),  # electric potential: EEG, iEEG, EMG, EOG, ECG
    "T": ("T", 0),  # magnetic flux density: MEG magnetometers
    "T/m": ("T/m", 0),  # MEG planar gradiometers
    "T/cm": ("T/m", 2),  # MEG axial gradiometers (CTF/4D convention)
    "A": ("A", 0),  # current: stimulation channels
    "S": ("S", 0),  # conductance: GSR/EDA (typically uS)
    "Ohm": ("Ohm", 0),  # impedance / respiration belts
    "ohm": ("Ohm", 0),
    "Ω": ("Ohm", 0),  # GREEK CAPITAL LETTER OMEGA
    "N": ("N", 0),  # force: EMG force/dynamometer channels
    "g": ("g", 0),  # accelerometers (units of gravity)
    "m": ("m", 0),  # displacement
    "s": ("s", 0),  # time
    "Hz": ("Hz", 0),  # frequency
    "deg/s": ("deg/s", 0),  # gyroscopes
    "rad/s": ("rad/s", 0),
}

# Longest symbol first, so "T/cm" is tried before "m" and "deg/s" before "s".
_BASE_SYMBOLS: tuple[str, ...] = tuple(sorted(_BASE_UNITS, key=len, reverse=True))

# Spellings that explicitly assert "no unit here". They must never parse: an
# unset unit carries no numeric claim, so there is nothing to convert from.
# Matched case-insensitively, so nothing here may collide with a real unit under
# case folding -- notably "na" is absent because it would swallow "nA".
_NON_UNITS = frozenset({"n/a", "none", "unknown", "a.u.", "a.u", "arbitrary", "-", "?"})

# pyedflib hands back the micro sign as these two mojibake bytes for EDF headers
# written in a non-UTF-8 codepage (see biosigio.importers._edf_tolerant).
_MANGLED_MICRO = "\x83\xca"


def parse_unit(unit: str | None) -> tuple[str, int] | None:
    """Split a unit string into its quantity and its decimal exponent.

    Args:
        unit: A physical unit label, e.g. ``"uV"``, ``"mV"``, ``"fT/cm"``.

    Returns:
        ``(quantity, exponent)`` such that one of ``unit`` equals
        ``10 ** exponent`` of ``quantity`` -- e.g. ``"uV" -> ("V", -6)`` and
        ``"fT/cm" -> ("T/m", -13)``. Returns **None** for an empty string, an
        explicit non-unit (``"n/a"``, ``"a.u."``), or any spelling this module
        does not recognize; callers treat None as "not convertible".

    Examples:
        >>> parse_unit("uV")
        ('V', -6)
        >>> parse_unit("V")
        ('V', 0)
        >>> parse_unit("a.u.") is None
        True
    """
    if not unit:
        return None
    text = unit.replace(_MANGLED_MICRO, "µ").strip()
    if not text or text.lower() in _NON_UNITS:
        return None
    for symbol in _BASE_SYMBOLS:
        if not text.endswith(symbol):
            continue
        prefix_exponent = _PREFIX_EXPONENTS.get(text[: len(text) - len(symbol)])
        if prefix_exponent is None:
            continue
        quantity, base_exponent = _BASE_UNITS[symbol]
        return quantity, prefix_exponent + base_exponent
    return None


def conversion_factor(from_unit: str | None, to_unit: str | None) -> float | None:
    """The multiplier that re-expresses a value from ``from_unit`` in ``to_unit``.

    Args:
        from_unit: The unit the values are currently in.
        to_unit: The unit the values should be expressed in.

    Returns:
        A float ``k`` such that ``value_in_to_unit == value_in_from_unit * k``,
        or **None** when either unit is unparsable or the two measure different
        quantities. None means "do not touch the values"; it is never a
        conversion of 1.0 in disguise.

    Examples:
        >>> conversion_factor("V", "uV")
        1000000.0
        >>> conversion_factor("uV", "uV")
        1.0
        >>> conversion_factor("V", "T") is None
        True
    """
    source = parse_unit(from_unit)
    target = parse_unit(to_unit)
    if source is None or target is None or source[0] != target[0]:
        return None
    return 10.0 ** (source[1] - target[1])
