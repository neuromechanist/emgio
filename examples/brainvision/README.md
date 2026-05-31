# BrainVision test fixture

`sub-01_task-rest_eeg.{vhdr,vmrk,eeg}` is a small BrainVision recording used to
test the BrainVision importer (issue #54).

**Provenance (CC0):** derived from the repository's CC0 EEG fixture
(`examples/bids/eeg/sub-01/...eeg.set`, OpenNeuro): the first 8 channels and
first 5 seconds (250 Hz) were written to the BrainVision triplet with `pybv`,
plus three synthetic stimulus markers at 1.0 s, 2.5 s, and 4.0 s. `pybv` is only
needed to *write* this fixture; the importer reads BrainVision via MNE.

Regenerate (requires `pybv` + the `meg` extra):

```python
import numpy as np, pybv
from emgio import EMG
emg = EMG.from_file("examples/bids/eeg/sub-01/eeg/sub-01_task-eyesopen_eeg.set", importer="eeglab")
rate = 250
chans = list(emg.signals.columns)[:8]
data = emg.signals[chans].to_numpy()[: 5 * rate].T.astype(float)
events = np.array([[rate, 1, 1], [int(2.5 * rate), 2, 1], [4 * rate, 1, 1]], dtype=int)
pybv.write_brainvision(data=data, sfreq=rate, ch_names=chans, fname_base="sub-01_task-rest_eeg",
                       folder_out="examples/brainvision", events=events, unit="uV", overwrite=True)
```
