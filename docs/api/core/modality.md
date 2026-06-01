# Modality Vocabulary

The modality vocabulary backs the per-channel `channel_type` / `modality`
contract used by `Recording.add_channel` and `set_channel`: it validates channel
types and coarse modalities (EEG, EMG, IEEG, MEG, BEH, MISC) and maps types to
BIDS `channels.tsv` values.

## Module Documentation

::: biosigio.core.modality
    options:
      show_root_heading: true
      show_source: true
      members: true
