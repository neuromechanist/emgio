# KIT / Yokogawa MEG test fixture

`sub-01_task-test_meg.sqd` is a small (~100 KB) KIT/Yokogawa MEG recording used to
test biosigIO's MEG importer KIT path (`read_raw_kit`, exercised via
`Recording.from_file`). 193 channels, 1000 Hz, 0.1 s.

## Provenance

Vendored from MNE-Python's test suite
(`mne/io/kit/tests/data/test_umd-raw.sqd`), which is distributed under the
BSD-3-Clause license. See https://github.com/mne-tools/mne-python. Renamed to a
BIDS-style filename; the bytes are unmodified.

KIT systems use both `.sqd` and `.con` extensions for the same format; biosigIO's
importer dispatches both (plus `.kdf`) to `read_raw_kit`. Reading is covered here;
the `.con`/`.kdf` extension dispatch is covered in `test_meg_importer.py`.

CTF `.ds` reading has its own fixture under `examples/ctf/` (see that README).
