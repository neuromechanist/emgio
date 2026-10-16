# One-minute FIF exemplar

`one_minute_raw.fif` is a real, small Neuromag/Elekta magnetoencephalography (MEG) recording in the FIF format.
It is kept under test so the FIF reader is exercised on a multi-sensor-type channel set and on stimulus-channel events.
The tests also write a split chain from it with MNE-Python inside `tmp_path`, to cover split recordings without committing the parts.

## Source

The recording is cut from the public dataset "MNE-somato-data-bids (anonymized)", a somatosensory MEG recording.

- OpenNeuro source DOI: `10.18112/openneuro.ds003104.v1.0.1`.
- NEMAR DOI: `10.82901/nemar.on003104`.
- License: CC0 1.0 Universal public domain dedication, so there is no restriction on reuse.
- Dataset authors: Lauri Parkkonen, Stefan Appelhoff, Alexandre Gramfort, Mainak Jas, and Richard Höchenberger.

## What was cut

- Time: the first 60 seconds of the continuous recording (`raw.crop(tmin=0, tmax=60)`), 18019 samples.
- Sampling rate: unchanged, 300.3074951171875 Hz.
  Nothing was resampled or filtered.
- Channels: 15 of the original 316, with the original names and order.
  - 4 magnetometers and 8 planar gradiometers (`MEG 0111` to `MEG 0143`, the first four sensor triplets).
  - 1 electrooculogram channel (`EOG 061`).
  - 2 stimulus channels (`STI 001`, and the composite trigger channel `STI 014`), which carry 7 trigger events in the cut.
- Sample values: identical to the source for the kept channels and the kept time span (checked with an exact array comparison after reading both with MNE-Python).
  The recording keeps its original start offset (`first_samp`), so event sample numbers are offset from the start of the cut.
- Format: one file, saved with MNE-Python 1.13.2 as `raw.save("one_minute_raw.fif")`, which does not split it.

## What was removed or cleared

- All subject information, and the experimenter and project name fields.
- All channels outside the 15 listed above.
- The head-position-indicator measurement arrays, the MaxFilter processing history, the cross-talk and fine-calibration tables, and the acquisition parameter and stimulus dumps.
  These are bound to the full 316-channel layout (one of them holds a patient-name field), and the head-position-indicator arrays alone would exceed the size budget.

## What was kept

- The measurement date.
  The source dataset anonymized it with a time shift, so it is not an identity.
- The digitized fiducial and head-position-coil points and the device-to-head transform.
- The event annotations that fall inside the first 60 seconds.
