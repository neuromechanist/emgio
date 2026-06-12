# CTF / VSM MEG test fixture

`catch-alp-good-f.ds` is a small (~6.5 MB) CTF/VSM MEG recording used to test
biosigIO's MEG importer CTF path (`read_raw_ctf`, exercised via
`Recording.from_file`) and the streaming Zarr export that the NEMAR serving
pipeline routes CTF `.ds` directories through. 244 channels (151 magnetometers
+ 29 reference magnetometers + 60 simultaneous EEG + 1 trigger + 3 misc),
1250 Hz, 4 s, 2 trigger events.

CTF is a *directory* format: the `.ds` folder holds a `.res4` header, a `.meg4`
binary (big-endian int32, channel-major), and sidecars. Unlike the single-file
MEG formats (FIF, KIT), it carries simultaneous EEG alongside MEG, so it also
exercises the importer's MEG/EEG modality split.

## Provenance

Vendored verbatim from the MNE testing data repository
(`mne-tools/mne-testing-data`, `CTF/catch-alp-good-f.ds`), which redistributes
real CTF sample recordings for software testing. The bytes are unmodified:

| file                      | md5                                |
| ------------------------- | ---------------------------------- |
| `catch-alp-good-f.meg4`   | `7e6c1b81805d3e83ce1ad85ec48f7470` |
| `catch-alp-good-f.res4`   | `1e73939af8c8d7d0a7e4b27d774fc300` |

Source: https://github.com/mne-tools/mne-testing-data (`CTF/` directory).
Upstream does not attach an explicit license to the `CTF/` files; they are
redistributed here solely as a test fixture. CTF test data is otherwise
download-only via `mne.datasets.testing` (too large to ship inside MNE-Python),
which is why the file is committed here for offline CI.

## What this covers

- `read_raw_ctf` reading + sensor-type and MEG/EEG modality split:
  `biosigio/tests/test_meg_importer.py`
- streaming Zarr export of a `.ds` directory (the driver's large-file path,
  including the 1250 -> 250 Hz MEG resample): `biosigio/tests/test_zarr_stream.py`
