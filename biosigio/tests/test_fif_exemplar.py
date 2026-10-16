"""FIF reader tests on a real one-minute exemplar and a split chain made from it.

``tests/data/fif/one_minute_raw.fif`` is a 60 s, 15-channel cut of a public CC0
Neuromag MEG recording (see ``tests/data/fif/NOTICE.md`` for the source, the
DOIs, the license and exactly what was cut). The split chain is NOT committed:
the tests write it into ``tmp_path`` from the exemplar with MNE-Python's own
``Raw.save(..., split_size=...)``, so the parts are what MNE writes and the
oracle for every comparison is MNE's read of the single exemplar file.

Everything goes through the public entry point, ``Recording.from_file``.
NO MOCKS: every scenario is the real exemplar, the chain MNE wrote from it, or
a byte-level copy of that chain broken on purpose.

Skips cleanly if the optional ``meg`` extra (mne) is not installed.
"""

import pathlib
import shutil
from collections import Counter

import numpy as np
import pytest

from biosigio import Recording
from biosigio.exceptions import CorruptFileError, FileReadError

mne = pytest.importorskip("mne", reason="FIF import requires the optional 'meg' extra (mne)")

EXEMPLAR = pathlib.Path(__file__).resolve().parent / "data" / "fif" / "one_minute_raw.fif"

pytestmark = pytest.mark.skipif(not EXEMPLAR.exists(), reason="FIF exemplar missing")

N_CHANNELS = 15
N_SAMPLES = 18019  # 60 s at the original 300.3 Hz
N_EVENTS = 7

# MNE splits a file once a part passes split_size minus a cushion of about 1 MiB,
# so this yields exactly two parts for the exemplar (1.5 MB gives three, 2.1 MB
# none), with a wide margin on both sides. The chain helper asserts the count.
SPLIT_SIZE = "1.8MB"
CHAIN_STEM = "chain_raw.fif"  # MNE names the second part chain_raw-1.fif
CHAIN_PART_2 = "chain_raw-1.fif"


def _write_chain(dest: pathlib.Path) -> pathlib.Path:
    """Write the exemplar into ``dest`` as an MNE split chain; return part 1's path.

    Fails loudly unless MNE wrote exactly two parts, so every scenario below is
    about a two-part chain and not about whatever the split size happened to give.
    """
    raw = mne.io.read_raw_fif(EXEMPLAR, preload=False, verbose="ERROR")
    raw.save(dest / CHAIN_STEM, split_size=SPLIT_SIZE, overwrite=True, verbose="ERROR")
    assert sorted(p.name for p in dest.glob("*.fif")) == [CHAIN_PART_2, CHAIN_STEM]
    return dest / CHAIN_STEM


def _copy_chain(src: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    """Copy both parts of a written chain into ``dest``; return part 1's path."""
    shutil.copy(src / CHAIN_STEM, dest)
    shutil.copy(src / CHAIN_PART_2, dest)
    return dest / CHAIN_STEM


@pytest.fixture(scope="module")
def oracle():
    """MNE-Python's own read of the single exemplar file, the reference for everything."""
    return mne.io.read_raw_fif(EXEMPLAR, preload=True, verbose="ERROR")


@pytest.fixture(scope="module")
def chain_dir(tmp_path_factory):
    """A directory holding the intact two-part chain MNE wrote from the exemplar."""
    path = tmp_path_factory.mktemp("fif_chain")
    _write_chain(path)
    return path


@pytest.fixture(scope="module", params=["single", "chain"])
def rec(request, chain_dir):
    """The exemplar read as a single file and as the two-part chain made from it."""
    path = EXEMPLAR if request.param == "single" else chain_dir / CHAIN_STEM
    return Recording.from_file(str(path))


def test_chain_has_two_parts_and_the_second_holds_samples(chain_dir, tmp_path):
    """MNE follows part 1 into part 2, and part 2 holds samples part 1 does not.

    Guards the scenarios below: were the second part empty or absent from the
    chain, a missing-part test would not be testing a split recording.
    """
    chained = mne.io.read_raw_fif(chain_dir / CHAIN_STEM, verbose="ERROR")
    assert [p.name for p in chained.filenames] == [CHAIN_STEM, CHAIN_PART_2]
    assert chained.n_times == N_SAMPLES

    shutil.copy(chain_dir / CHAIN_STEM, tmp_path)
    part_1_only = mne.io.read_raw_fif(
        tmp_path / CHAIN_STEM, on_split_missing="ignore", verbose="ERROR"
    )
    assert 0 < part_1_only.n_times < N_SAMPLES


def test_reader_matches_mne(rec, oracle):
    """Channel names, rate, length and every sample equal MNE's read of the exemplar.

    Runs on the single file and on the split chain: the chain read must give
    back the same recording, which also proves both parts were read.
    """
    sfreq = float(oracle.info["sfreq"])

    assert list(rec.channels) == oracle.ch_names
    assert list(rec.signals.columns) == oracle.ch_names
    assert {c["sample_frequency"] for c in rec.channels.values()} == {sfreq}
    assert rec.get_sampling_frequency() == sfreq

    assert rec.get_n_samples() == oracle.n_times
    assert rec.get_n_samples() == N_SAMPLES
    assert rec.get_n_channels() == N_CHANNELS
    assert rec.get_duration() == pytest.approx(60.0, abs=0.01)

    # Exact equality, not a tolerance: the reader must not alter a single value.
    np.testing.assert_array_equal(rec.signals.to_numpy().T, oracle.get_data())


def test_sensor_types_and_units_stay_distinct(rec):
    """Magnetometers, planar gradiometers, EOG and triggers keep their own type and unit."""
    types = Counter(c["channel_type"] for c in rec.channels.values())
    assert types == {"MEGMAG": 4, "MEGGRADPLANAR": 8, "EOG": 1, "TRIG": 2}

    units = {c["channel_type"]: c["physical_dimension"] for c in rec.channels.values()}
    assert units["MEGMAG"] == "T"
    assert units["MEGGRADPLANAR"] == "T/m"
    assert units["EOG"] == "V"


def test_missing_split_part_raises_corrupt_file_error(chain_dir, tmp_path):
    """Deleting the second part makes the read fail with a typed error naming it."""
    part_1 = _copy_chain(chain_dir, tmp_path)
    # Control: the intact copy reads, so the failure below is the deleted part alone.
    assert Recording.from_file(str(part_1)).get_n_samples() == N_SAMPLES

    (tmp_path / CHAIN_PART_2).unlink()

    with pytest.raises(CorruptFileError) as excinfo:
        Recording.from_file(str(part_1))
    assert excinfo.value.code == CorruptFileError.code
    assert CHAIN_PART_2 in str(excinfo.value)


@pytest.mark.parametrize("truncated_part", [CHAIN_STEM, CHAIN_PART_2])
def test_truncated_part_raises_file_read_error(chain_dir, tmp_path, truncated_part):
    """Half of either part is a typed read error, but the generic one, not CorruptFileError.

    Pins the behavior observed today. ``classify_read_error`` maps a truncated FIF
    to ``CorruptFileError`` only for phrasings it knows; the one numpy raises when
    the file ends inside a data tag ("buffer size must be a multiple of element
    size") is not among them, so it falls through to the generic ``FileReadError``
    with code ``file_read_error``. If the classifier learns this phrasing, the type
    becomes ``CorruptFileError`` (a subclass of ``FileReadError``) and this
    assertion is the one to update.
    """
    part_1 = _copy_chain(chain_dir, tmp_path)
    # Control: the intact copy reads, so the failure below is the truncation alone.
    assert Recording.from_file(str(part_1)).get_n_samples() == N_SAMPLES

    victim = tmp_path / truncated_part
    data = victim.read_bytes()
    victim.write_bytes(data[: len(data) // 2])

    with pytest.raises(FileReadError) as excinfo:
        Recording.from_file(str(part_1))
    assert type(excinfo.value) is FileReadError
    assert excinfo.value.code == FileReadError.code
    # The error names the file that was opened (part 1), whichever part is cut.
    assert CHAIN_STEM in str(excinfo.value)


def test_trigger_events_match_mne_find_events(rec, oracle):
    """Stim-channel triggers become events at the same times and codes as MNE finds."""
    events = mne.find_events(oracle, verbose="ERROR")
    sfreq = float(oracle.info["sfreq"])

    # The cut keeps its original start offset, so the onset arithmetic is exercised.
    assert oracle.first_samp > 0
    assert len(events) == N_EVENTS
    assert rec.events is not None
    assert len(rec.events) == N_EVENTS

    expected_onsets = (events[:, 0] - oracle.first_samp) / sfreq
    np.testing.assert_array_equal(rec.events["onset"].to_numpy(), expected_onsets)
    assert rec.events["description"].tolist() == [str(code) for code in events[:, 2]]
    assert (rec.events["duration"] == 0.0).all()
    # Every event falls inside the 60 s cut.
    assert rec.events["onset"].between(0.0, 60.0).all()
