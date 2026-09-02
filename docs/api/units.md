# Physical Units

Parsing and conversion for the unit strings a channel carries in
`physical_dimension`. `parse_unit` splits a decimal-prefixed unit into its
quantity and exponent (`"uV" -> ("V", -6)`); `conversion_factor` gives the exact
power-of-ten multiplier between two units of the same quantity, or `None` when
they are not convertible.

This is what lets `bids.apply_channels_tsv` adopt a BIDS `channels.tsv` `units`
column by **converting** the samples rather than relabelling them (issue #122).
Parsing is case-sensitive on purpose: `m` is milli and `M` is mega, and a
lenient reading of `MV` as millivolts would be a 10^9 error in the values.

## Module Documentation

::: biosigio.units
    options:
      show_root_heading: true
      show_source: true
      members: true
