# Neo Importer (proprietary electrophysiology)

Imports proprietary electrophysiology acquisition formats via python-neo: Intan
(`.rhd/.rhs`), Blackrock (`.ns1`-`.ns6`), Spike2/CED (`.smr/.smrx`), Plexon
(`.plx/.pl2`), Micromed (`.trc`), Neuralynx (`.ncs`), and more. neo models a
recording as signal streams (channel groups sharing a rate); same-rate streams
merge, and multi-rate files need an explicit `stream=` selector. neo carries no
BIDS channel type, so channels default to `OTHER` (override with `channel_type=`
or a sibling `_channels.tsv`). Requires the `neo` extra.

See [Proprietary electrophysiology (Neo)](../../formats/neo.md) for details.

## Module Documentation

::: biosigio.importers.neo
    options:
      show_root_heading: true
      show_source: true
      members: true

## Usage Example

```python
from biosigio import Recording

rec = Recording.from_file("recording.rhd")                       # Intan
rec = Recording.from_file("recording.ns5", stream="lfp")          # pick a rate-group
rec = Recording.from_file("recording.rhd", channel_type="SEEG")   # label the channels
```
