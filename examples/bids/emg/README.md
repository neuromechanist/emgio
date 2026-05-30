# Grison et al 2025: HDsEMG recordings (emgio test fixture, truncated)

BIDS-formatted version of a HDsEMG dataset corresponding to *[Grison et al. 2025](https://doi.org/10.1113/JP287913)*.

> **emgio test fixture note.** This is a heavily truncated copy of one subject / one
> run from the original dataset (OpenNeuro/NEMAR `nm000165`, License: CC0). To keep the
> fixture small (< 5 MB), only the first 5 EMG channels (Ch001-Ch005, grid1) and 2 MISC
> torque channels (Ch130 performed-path, Ch131 acquired-V) were retained, out of the
> original 131 channels. The full 30 s recording at the native 10240 Hz sampling rate is
> preserved with no resampling. The `space-grid2` and `space-lowerLeg` coordinate systems
> and their electrodes were dropped because no channels from them were kept. The source EDF
> contained no embedded annotations, so `*_events.tsv` was populated with the 3 real
> trapezoidal-contraction protocol phases (ramp_up / plateau / ramp_down at 10% MVC) for
> testing; onsets/durations follow the protocol described below.

### Overview
One healthy subject performed 10 submaximal (10 to 70 percent MVC) isometric
ankle dorsiflexions. EMG signals were recorded from the right tibialis anterior
using two arrays of 64 surface electrodes (4 mm interelectrode distance, 13x5 configuration)
for a total of 128 electrodes.

### Protocol description
The participant performed one, two, or three trapezoidal contractions (with repetitions being
specified by the run labels) at 10, 15, 20, 25, 30, 35, 40, 50, 60, and 70 percent MVC
with 120 s of rest in between, consisting of linear ramps up and down performed at
5 percent per second and a plateau maintained for 20 s up to 30 percent MVC, 15 s for 35 percent
and 40 percent MVC, and 10 s from 50 percent to 70 percent MVC. The order of the
contractions was randomized.

### Set-up description
The participant sat on a chair with the hips flexed at 30 degrees, 0 degrees being
the hip neutral position, and their knees fully extended. The foot of
the dominant leg (right) was fixed onto the pedal of a commercial dynamometer (OT Bioelettronica)
positioned at 30 degrees in the plantarflexion direction. Force signals
were recorded with a load cell (CCT Transducer s.a.s.) connected in-series to the pedal
using the same acquisition system as for the HD-EMG recordings.

### Coordinate systems
All electrode coordinates (reported in mm) are reported in their respective
grid coordinate system (*space-grid1* and *space-grid2*).
Their relative positions as well as the positions of the reference
and ground electrodes are reported in a separate coordinate system (*space-lowerLeg*)
reported in percent of the lower leg length.

### Labeled motor unit spike trains
Labeled motor unit spike trains were derived from concurrently recorded invasive EMG
and curated by an experienced investigator (only available for *_run-01* of each trial).

### Conversion
The dataset has been converted semi-automatically using the [*MUniverse*](https://github.com/dfarinagroup/muniverse/tree/main) software.
See *dataset_description.json* for further details.
