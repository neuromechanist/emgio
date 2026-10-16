# One-minute BrainVision exemplar

`one_minute.vhdr`, `one_minute.vmrk` and `one_minute.eeg` are a one-minute, 8 channel BrainVision recording for the reader tests.
It is built from the repository's own EEG test fixture (`examples/bids/eeg`), so no data was fetched for it.
The EDF+C and BDF+C exemplars in `../edf` and `../bdf` carry the same recording.

## Source

The fixture is a derivative of this public dataset:

- Ramne, M., Damercheli, S., Apelgren, F., Pettersson, I., and Lendaro, E.
  High Density Resting State EEG of Phantom Limb Pain and Controls.
  OpenNeuro, version 1.1.1.
  doi:10.18112/openneuro.ds006921.v1.1.1
- NEMAR copy: doi:10.82901/nemar.on006921

License: CC0 1.0 (public domain dedication).
The authors are credited here as a courtesy; CC0 does not require it.

The fixture itself is one 60 second excerpt of an eyes-open resting-state run, anti-alias downsampled from 2400 Hz to 250 Hz, with three synthetic event markers added (see its `dataset_description.json`).

## Derivation from the fixture

- Duration: all 60 seconds.
- Channels: the first 8, in the fixture's order (FP1, AF7, FPZ, FP2, AF8, F9, F7, F5).
- Sampling rate: 250 Hz (`SamplingInterval=4000` microseconds), unchanged.
- Samples: the fixture's 32-bit floats in microvolts, copied bit for bit and written as multiplexed (interleaved) little-endian `IEEE_FLOAT_32` with a resolution of 1 microvolt.
  Nothing is quantized, resampled, filtered or rescaled.
- Markers: three `Stimulus` markers at the fixture's three event times (0.5, 20 and 40 seconds), named by the fixture's trial types.

## What was removed

- The other 56 channels of the fixture, its electrode positions, and every sidecar file.
- The original recording path and the dataset's participant information; the three files carry no names, identifiers, birth dates or free-text notes about a person, and the header has no `[Comment]` section.
