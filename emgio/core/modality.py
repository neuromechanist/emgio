"""BIDS-aligned channel-type and modality vocabularies for biosignal data.

This module is the single source of truth for the two-axis model that replaces
the historical single ``channel_type`` string:

- ``channel_type``: the fine-grained `Brain Imaging Data Structure (BIDS)
  <https://bids-specification.readthedocs.io/>`_ channel ``type``
  (e.g. ``EEG``, ``EMG``, ``ECG``, ``ACC``).
- ``modality``: the coarse recording modality (``EEG``, ``EMG``, ``IEEG``,
  ``MEG``, ``BEH``, ``MISC``).

There is no silent default; callers must provide an explicit channel type, and
unknown values raise rather than being relabelled.
"""

from __future__ import annotations

# BIDS channel `type` values plus device-domain types common in wearable EMG.
# Uppercase canonical forms.
VALID_CHANNEL_TYPES: frozenset[str] = frozenset(
    {
        # Neural
        "EEG",
        "SEEG",
        "ECOG",
        "DBS",
        "MEGMAG",
        "MEGGRADAXIAL",
        "MEGGRADPLANAR",
        "MEGREFMAG",
        "MEGREFGRADAXIAL",
        "MEGREFGRADPLANAR",
        # Muscle / cardiac / ocular
        "EMG",
        "ECG",
        "EKG",
        "EOG",
        "VEOG",
        "HEOG",
        # Reference / trigger / misc
        "REF",
        "TRIG",
        "MISC",
        "OTHER",
        # Device-domain (wearables, IMUs, peripherals)
        "ACC",
        "GYRO",
        "QUAT",
        "CTRL",
        "MAGN",
        "RESP",
        "GSR",
        "TEMP",
        "PPG",
        "SYSCLOCK",
    }
)

VALID_MODALITIES: frozenset[str] = frozenset({"EEG", "EMG", "IEEG", "MEG", "BEH", "MISC"})

# Explicit channel_type -> modality map. Any valid channel_type not listed here
# resolves to MISC (a deterministic mapping, not a guess).
_CHANNEL_TYPE_TO_MODALITY: dict[str, str] = {
    "EEG": "EEG",
    "EMG": "EMG",
    "SEEG": "IEEG",
    "ECOG": "IEEG",
    "DBS": "IEEG",
    "MEGMAG": "MEG",
    "MEGGRADAXIAL": "MEG",
    "MEGGRADPLANAR": "MEG",
    "MEGREFMAG": "MEG",
    "MEGREFGRADAXIAL": "MEG",
    "MEGREFGRADPLANAR": "MEG",
}


def validate_channel_type(channel_type: str) -> str:
    """Normalize and validate a channel type against :data:`VALID_CHANNEL_TYPES`.

    Args:
        channel_type: A channel type string (case-insensitive).

    Returns:
        The canonical uppercase channel type.

    Raises:
        ValueError: If ``channel_type`` is empty, ``n/a``, or not a known type.
    """
    ct = channel_type.strip().upper()
    if ct in ("", "N/A", "NA"):
        raise ValueError(
            "channel_type 'n/a' is not allowed; use 'OTHER' or 'MISC' for an unknown type."
        )
    if ct not in VALID_CHANNEL_TYPES:
        raise ValueError(
            f"Unknown channel_type {channel_type!r}. Valid channel types: "
            f"{sorted(VALID_CHANNEL_TYPES)}"
        )
    return ct


def validate_modality(modality: str) -> str:
    """Normalize and validate a modality against :data:`VALID_MODALITIES`.

    Args:
        modality: A modality string (case-insensitive).

    Returns:
        The canonical uppercase modality.

    Raises:
        ValueError: If ``modality`` is not a known modality.
    """
    m = modality.strip().upper()
    if m not in VALID_MODALITIES:
        raise ValueError(
            f"Unknown modality {modality!r}. Valid modalities: {sorted(VALID_MODALITIES)}"
        )
    return m


def infer_modality_from_channel_type(channel_type: str) -> str:
    """Derive the coarse modality for a channel type.

    The mapping is deterministic: neural types map to EEG/IEEG/MEG, ``EMG`` maps
    to EMG, and every other valid type (ECG, EOG, ACC, TRIG, ...) maps to MISC.

    Args:
        channel_type: A channel type string (case-insensitive); validated first.

    Returns:
        One of :data:`VALID_MODALITIES`.

    Raises:
        ValueError: If ``channel_type`` is not a known type.
    """
    ct = validate_channel_type(channel_type)
    return _CHANNEL_TYPE_TO_MODALITY.get(ct, "MISC")
