"""Channel-type properties shared across the package.

The modality vocabulary itself lives in :mod:`biosigio.core.modality`; this holds
the small behavioural facts about a type that more than one subsystem has to
agree on. Right now that is exactly one: which types carry **discrete codes**
rather than a sampled physical quantity.
"""

from __future__ import annotations

# Channel types whose samples are codes, not measurements: an event value, a
# clock tick, a control state. Two subsystems must treat them the same way, and
# for the same reason -- arithmetic that is correct for a waveform is meaningless
# for a code:
#
# - the Zarr exporters skip anti-aliased resampling for them (interpolating
#   between trigger codes 5 and 7 invents a 6 that never happened);
# - :func:`biosigio.bids.apply_channels_tsv` never rescales them, whatever a BIDS
#   ``channels.tsv`` declares for their ``units`` (issue #122). MNE labels stim
#   channels with the FIFF volts code while they hold integer codes, so a sidecar
#   declaring ``mV`` would otherwise turn codes 5/3/7 into 5000/3000/7000.
#
# Keeping one definition means a type added here cannot be discrete for the
# exporter and continuous for the sidecar.
DISCRETE_CHANNEL_TYPES: frozenset[str] = frozenset({"TRIG", "SYSCLOCK", "CTRL"})
