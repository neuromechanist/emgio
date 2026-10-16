# One-minute BDF+C exemplar

`one_minute.bdf` is a one-minute, 8 channel BioSemi Data Format Plus (BDF+) recording for the reader tests.
It is built from the repository's own EEG test fixture (`examples/bids/eeg`), so no data was fetched for it.
The BrainVision and EDF+C exemplars in `../brainvision` and `../edf` carry the same recording.

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

- Duration: all 60 seconds, written as 60 one-second data records.
- Channels: the first 8, in the fixture's order (FP1, AF7, FPZ, FP2, AF8, F9, F7, F5), in microvolts.
- Sampling rate: 250 Hz, unchanged.
- Format: BDF+C (continuous), with the annotations signal that BDF+ requires.
- Quantization: BDF stores 24-bit integers.
  Each channel's physical range is its own minimum and maximum, widened to whole microvolts and one microvolt beyond, mapped onto the digital range -8388608 to 8388607.
  A sample therefore differs from the fixture's value by less than one digital step, which is under 0.001 microvolts.
- Annotations: the fixture's three events at 0.5, 20 and 40 seconds, with zero duration.
- Header: the start date is 1985-01-01 00:00:00, the date EDF and BDF use for anonymized files, because the fixture has no recording date; the patient and recording fields are left unspecified.
- Written with pyedflib.

## What was removed

- The other 56 channels of the fixture, its electrode positions, and every sidecar file.
- The original recording path and the dataset's participant information.
