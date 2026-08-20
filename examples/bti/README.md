# 4D Neuroimaging / BTi MEG test fixture

`sub-01_task-test_meg/` is a small (~480 KB total) 4D Neuroimaging/BTi MEG
recording used to test biosigIO's MEG importer BTi path (`read_raw_bti`,
exercised via `Recording.from_file` and its content-based directory
detection). 280 channels (248 magnetometers + 23 reference magnetometers + 7
misc + 2 stim), ~1017.25 Hz, 305 samples (~0.3 s).

The directory is named without an extension and holds the three files a real
BIDS BTi recording ships (`sub-<label>[_ses-<label>]_task-<label>
[_run-<index>]_meg/` containing the PDF file, `config`, and `hs_file`):

```
sub-01_task-test_meg/
  c,rfDC     # processed-data (PDF) file
  config     # acquisition config
  hs_file    # digitized head-shape points
```

## Provenance

Vendored from MNE-Python's own test suite
(`mne/io/bti/tests/data/test_pdf_linux`, `test_config_linux`, `test_hs_linux`),
which is distributed under the BSD-3-Clause license. See
https://github.com/mne-tools/mne-python. These files are not part of the large,
download-only `mne-testing-data` repository (unlike the CTF fixture in
`examples/ctf/`) -- they are small binaries committed directly to mne-python's
own git history for its unit tests, which is why they can be vendored here
instead of gated behind a multi-GB download. Renamed to the conventional BTi/
BIDS basenames (`c,rfDC`/`config`/`hs_file`); the bytes are unmodified.

| vendored file | original path (mne-python) | md5 |
| --- | --- | --- |
| `c,rfDC` | `mne/io/bti/tests/data/test_pdf_linux` | `c9fb92ff484070e24bbf083ad623c0ea` |
| `config` | `mne/io/bti/tests/data/test_config_linux` | `d43d1ee1e18606fdd21a76b3b5c53614` |
| `hs_file` | `mne/io/bti/tests/data/test_hs_linux` | `6f6b4b6136adad38e3197ccf28a992e8` |

## What this covers

- content-based BTi directory detection (`_find_bti_pdf`) and the negative case
  (a directory holding only `.datalad/config` is not misdetected): both are
  synthetic-layout tests, not this fixture
- `read_raw_bti` reading through `Recording.from_file` (channel types, units,
  sample count): `biosigio/tests/test_bti_importer.py`
- streaming Zarr export of a BTi directory: `biosigio/tests/test_zarr_stream.py`
