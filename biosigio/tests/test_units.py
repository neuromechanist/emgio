"""Tests for the physical-unit conversion table (issue #122).

:mod:`biosigio.units` is the arithmetic behind adopting a BIDS ``channels.tsv``
``units`` column, so a wrong entry here is a silent order-of-magnitude error in
every recording that carries a sidecar. NO MOCKS: the module is pure arithmetic
over strings and is exercised directly.
"""

import pytest

from biosigio.units import conversion_factor, parse_unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Electric potential, the case that matters for EEG/iEEG/EMG.
        ("V", ("V", 0)),
        ("mV", ("V", -3)),
        ("uV", ("V", -6)),
        ("µV", ("V", -6)),  # MICRO SIGN
        ("μV", ("V", -6)),  # GREEK SMALL LETTER MU
        ("nV", ("V", -9)),
        ("kV", ("V", 3)),
        ("MV", ("V", 6)),
        # Magnetic units, for MEG.
        ("T", ("T", 0)),
        ("fT", ("T", -15)),
        ("pT", ("T", -12)),
        ("T/m", ("T/m", 0)),
        ("fT/cm", ("T/m", -13)),  # 10^-15 T per 10^-2 m
        # Aux / peripheral units.
        ("uS", ("S", -6)),
        ("mA", ("A", -3)),
        ("ms", ("s", -3)),
        ("deg/s", ("deg/s", 0)),
        ("g", ("g", 0)),
        ("mg", ("g", -3)),
        ("kOhm", ("Ohm", 3)),
        # Whitespace and the pyedflib micro-sign mojibake are normalized.
        ("  uV  ", ("V", -6)),
        ("\x83\xcaV", ("V", -6)),
    ],
)
def test_parse_unit_splits_prefix_from_quantity(text, expected):
    assert parse_unit(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "n/a",
        "N/A",
        "a.u.",
        "arbitrary",
        "unknown",
        "count",  # a tally, not a physical unit
        "rad",  # a real unit, but not one this table converts
        "Volt",  # spelled out, deliberately not accepted
        "mmHg",
        "V/m",
    ],
)
def test_parse_unit_rejects_non_units_and_unknown_spellings(text):
    assert parse_unit(text) is None


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("V", "uV", 1e6),  # the issue #122 case: MNE volts -> BIDS microvolts
        ("uV", "V", 1e-6),
        ("mV", "uV", 1e3),
        ("uV", "mV", 1e-3),
        ("V", "V", 1.0),
        ("uV", "µV", 1.0),  # pure spelling difference
        ("μV", "uV", 1.0),
        ("T", "fT", 1e15),
        ("T/m", "fT/cm", 1e13),
        ("kOhm", "Ohm", 1e3),
    ],
)
def test_conversion_factor_table(source, target, expected):
    assert conversion_factor(source, target) == pytest.approx(expected, rel=1e-15)


def test_conversion_factor_is_exact_not_merely_close():
    """Powers of ten must be exact, or every rescale adds error of its own.

    ``1.0 / 1e-6`` is ``1000000.0000000001``; ``10.0 ** 6`` is ``1000000.0``.
    The module computes the latter, and this pins that.
    """
    assert conversion_factor("V", "uV") == 1000000.0
    assert conversion_factor("uV", "V") == 1e-6
    assert conversion_factor("V", "mV") == 1000.0


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("V", "T"),  # different quantities
        ("V", "a.u."),
        ("n/a", "uV"),  # the importer made no claim; nothing to convert from
        ("uV", "n/a"),
        ("V", "count"),
        ("deg/s", "rad/s"),  # related but not the same quantity: never guessed
        ("V", None),
        (None, "V"),
    ],
)
def test_conversion_factor_is_none_when_not_convertible(source, target):
    assert conversion_factor(source, target) is None


def test_prefix_parsing_is_case_sensitive():
    """'m' is milli and 'M' is mega; conflating them is a 10^9 error."""
    assert conversion_factor("mV", "V") == 1e-3
    assert conversion_factor("MV", "V") == 1e6
    assert parse_unit("MV") != parse_unit("mV")
    # A lenient parser would read these as volts; they must not parse at all.
    assert parse_unit("uv") is None
    assert parse_unit("Uv") is None


@pytest.mark.parametrize(
    ("source", "target"),
    [("V", "uV"), ("mV", "nV"), ("T", "fT"), ("uS", "S"), ("T/m", "fT/cm")],
)
def test_conversion_factors_are_reciprocal(source, target):
    forward = conversion_factor(source, target)
    backward = conversion_factor(target, source)
    assert forward * backward == pytest.approx(1.0, rel=1e-12)
