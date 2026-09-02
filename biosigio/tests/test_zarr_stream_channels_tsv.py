"""The streaming exporter applies the BIDS channels.tsv too (issue #127).

``Recording.from_file`` has applied the sibling ``_channels.tsv`` since #57, and
since #122/#125 adopting a declared unit *converts* the samples into it. The
streaming exporter read straight through the importers and never saw the sidecar,
so one dataset whose recordings straddle a size threshold served its small runs in
the sidecar's unit and its large runs in the importer's native unit -- each store
self-consistent, the pair wrong by 10^6, with per-channel ``type`` differing too.

The property under test is therefore not "streaming applies a sidecar" but
**streamed == in-memory**: the same recording and the same sidecar must produce
the same units, types, scale/offset and values whichever path built the store.
That is what the parity tests below pin, with ``force_modality`` set and unset,
on a real BrainVision iEEG triple and on the committed 305-channel MEG FIF.

NO MOCKS: every fixture is a real recording on disk, read back through the real
importers -- BrainVision triples in the pybv 0.7.6 shape (shared with
``test_bids_channels_units.py``) and the committed ``examples/bids/meg`` FIF,
copied with a sidecar whose ``units`` column is rewritten to ``fT``.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("zarr", reason="the Zarr serving format requires the 'zarr' extra")
pytest.importorskip("mne", reason="BrainVision/FIF import requires the 'meg' extra (mne)")

import zarr  # noqa: E402

from biosigio import Recording, stream_to_zarr  # noqa: E402

from .test_bids_channels_units import write_brainvision, write_channels_tsv  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[2]
MEG_FIF = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_meg.fif"
MEG_CHANNELS_TSV = _REPO / "examples/bids/meg/sub-01/meg/sub-01_task-mouse_channels.tsv"

SFREQ = 250.0
N_SAMPLES = 1000
# The caps the NEMAR converter passes to every export (nemar-cli#1068).
MODALITY_RATES = {"EEG": 250, "MEG": 250, "IEEG": 1000, "EMG": 1000}


def signal_microvolts(n_channels: int, n_samples: int = N_SAMPLES) -> np.ndarray:
    """``(n_samples, n_channels)`` of distinct µV-scale waveforms with an offset.

    Per-channel amplitudes and a non-zero mean make a mislabelled or unscaled
    channel visible in ``scale``/``offset``, not only in the samples.
    """
    t = np.arange(n_samples) / SFREQ
    return np.column_stack(
        [(20.0 + 7 * c) * np.sin(2 * np.pi * (3 + c) * t) + c for c in range(n_channels)]
    )


def channel_facts(store_path: str) -> dict[str, dict]:
    """Per-channel unit/type/modality/scale/offset from a store, keyed by label.

    Flattened across channel groups on purpose: which group a channel lands in is
    part of what the sidecar can change (a ``type`` re-derives the modality), so a
    per-group comparison would hide exactly the disagreement being tested.
    """
    root = zarr.open_group(store_path, mode="r")
    facts: dict[str, dict] = {}
    for gname in root.attrs["channel_groups"]:
        for channel in root[gname].attrs["channels"]:
            fact = {
                "group": gname,
                "channel_type": channel["channel_type"],
                "modality": channel["modality"],
                "unit": channel["unit"],
                "scale": channel["scale"],
                "offset": channel["offset"],
                "bids_unit": channel.get("bids_unit"),
            }
            facts[channel["label"]] = fact
    return facts


def dequantized(store_path: str) -> dict[str, np.ndarray]:
    """Physical-unit level-0 samples per channel label."""
    root = zarr.open_group(store_path, mode="r")
    values: dict[str, np.ndarray] = {}
    for gname in root.attrs["channel_groups"]:
        group = root[gname]
        base = np.asarray(group["0"][:])
        for channel in group.attrs["channels"]:
            row = base[channel["row_index"]]
            values[channel["label"]] = row * channel["scale"] + channel["offset"]
    return values


def physical_range(fact: dict) -> tuple[float, float]:
    """The (min, max) physical values a channel's ``scale``/``offset`` encode.

    ``scale`` and ``offset`` are just those two endpoints in other coordinates
    (``scale = (pmax - pmin) / 65535``, ``offset = pmin + 32768 * scale``), so
    comparing endpoints compares scale and offset -- but without the cancellation
    that makes ``scale`` a bad thing to compare directly. A channel with a large
    DC offset and a narrow span (an MEG reference sensor, say) has
    ``pmax - pmin`` many orders below ``|pmax|``, so a last-bit float32
    difference in the endpoints is a much larger *relative* difference in their
    subtraction. The endpoints are what the two paths actually derived.
    """
    return fact["offset"] - 32768 * fact["scale"], fact["offset"] + 32767 * fact["scale"]


def assert_stores_agree(streamed: str, in_memory: str) -> None:
    """The #127 contract: two paths, one recording, one sidecar, one answer.

    Unit, type, modality and group must match exactly -- those are labels, and a
    label is either right or wrong. The quantization parameters are compared to
    float32 precision rather than bit-for-bit: the streaming path stages the
    recording in a channel-major float32 memmap (that is what bounds its memory),
    so the physical extremes it quantizes against differ in the last float32 bit
    from the in-memory float64 path. That predates this change, and is also why
    the dequantized values are compared within a few int16 steps.
    """
    streamed_facts = channel_facts(streamed)
    in_memory_facts = channel_facts(in_memory)
    assert set(streamed_facts) == set(in_memory_facts)
    for label, expected in in_memory_facts.items():
        actual = streamed_facts[label]
        for key in ("group", "channel_type", "modality", "unit", "bids_unit"):
            assert actual[key] == expected[key], f"{label}.{key}"
        # Absolute floor: a millionth of the channel's own physical span, for a
        # channel whose min or max is exactly zero and has no relative scale.
        floor = 1e-6 * 65535 * abs(expected["scale"])
        for actual_edge, expected_edge in zip(
            physical_range(actual), physical_range(expected), strict=True
        ):
            assert actual_edge == pytest.approx(expected_edge, rel=1e-6, abs=floor), (
                f"{label} physical range"
            )

    streamed_values = dequantized(streamed)
    for label, expected_values in dequantized(in_memory).items():
        actual_values = streamed_values[label]
        assert actual_values.shape == expected_values.shape
        span = float(expected_values.max() - expected_values.min()) or 1.0
        # One int16 step is span/65535; the float32 memmap costs a few of them.
        assert np.max(np.abs(actual_values - expected_values)) <= 6 * span / 65535

    streamed_root = dict(zarr.open_group(streamed, mode="r").attrs)
    in_memory_root = dict(zarr.open_group(in_memory, mode="r").attrs)
    assert streamed_root.get("channels_tsv_units") == in_memory_root.get("channels_tsv_units")


def export_both_ways(
    recording: pathlib.Path, tmp_path: pathlib.Path, *, force_modality: str | None
) -> tuple[str, str]:
    """Export one recording through both paths; return ``(streamed, in_memory)``.

    ``force_modality`` is the NEMAR driver's suffix-driven grouping, applied to
    the in-memory path the way the driver applies it (overwrite every channel's
    modality after import) so the two stores are genuinely comparable.
    """
    streamed = str(tmp_path / "streamed.zarr")
    in_memory = str(tmp_path / "in_memory.zarr")
    stream_to_zarr(
        str(recording),
        streamed,
        force_modality=force_modality,
        modality_rates=MODALITY_RATES,
        dtype="int16",
    )
    rec = Recording.from_file(str(recording))
    if force_modality is not None:
        for label in rec.channels:
            rec.channels[label]["modality"] = force_modality
    rec.to_zarr(in_memory, dtype="int16", modality_rates=MODALITY_RATES)
    return streamed, in_memory


@pytest.fixture
def ieeg_in_volts(tmp_path):
    """A real BrainVision iEEG triple in SI volts with a ``uV`` SEEG sidecar.

    The nm000183 shape: pybv wrote µV, MNE reads SI volts, and the sidecar asks
    for the file's own microvolts back -- a 10^6 conversion, plus a type
    (``SEEG``) the BrainVision header cannot express.
    """
    stem = "sub-000_task-clips_run-001"
    names = ("LMacro_01", "LMacro_02")
    microvolts = signal_microvolts(len(names))
    vhdr = write_brainvision(
        tmp_path, stem, microvolts * 1e-6, unit="V", resolution=1.0, sfreq=SFREQ, names=names
    )
    write_channels_tsv(tmp_path, stem, [(name, "SEEG", "uV") for name in names])
    return vhdr, microvolts


@pytest.mark.parametrize("force_modality", [None, "IEEG"])
def test_streamed_and_in_memory_stores_agree_on_units_and_types(
    ieeg_in_volts, tmp_path, force_modality
):
    """#127's headline: the two export paths must not disagree about a recording."""
    vhdr, microvolts = ieeg_in_volts
    streamed, in_memory = export_both_ways(vhdr, tmp_path, force_modality=force_modality)

    assert_stores_agree(streamed, in_memory)

    facts = channel_facts(streamed)
    assert {f["unit"] for f in facts.values()} == {"uV"}
    assert {f["channel_type"] for f in facts.values()} == {"SEEG"}
    assert {f["modality"] for f in facts.values()} == {force_modality or "IEEG"}

    # Not merely relabelled: the values are the file's own microvolts.
    values = dequantized(streamed)
    for column, label in enumerate(("LMacro_01", "LMacro_02")):
        expected = microvolts[:, column]
        step = float(expected.max() - expected.min()) / 65535
        assert np.max(np.abs(values[label] - expected)) <= 6 * step


def test_streaming_without_the_sidecar_still_serves_the_importers_volts(ieeg_in_volts, tmp_path):
    """The pre-#127 behaviour, kept reachable and pinned as the contrast.

    This is what every streamed store looked like: MNE's SI volts under the
    importer's own label. It is also the mutation check for the test above --
    the two differ by exactly the sidecar.
    """
    vhdr, microvolts = ieeg_in_volts
    store = str(tmp_path / "off.zarr")
    stream_to_zarr(str(vhdr), store, force_modality="IEEG", bids_channels="off")

    facts = channel_facts(store)
    assert {f["unit"] for f in facts.values()} == {"V"}
    assert {f["channel_type"] for f in facts.values()} == {"EEG"}  # MNE's guess, not SEEG
    assert dict(zarr.open_group(store, mode="r").attrs).get("channels_tsv_units") is None

    values = dequantized(store)["LMacro_01"]
    expected = microvolts[:, 0] * 1e-6
    step = float(expected.max() - expected.min()) / 65535
    assert np.max(np.abs(values - expected)) <= 6 * step


def test_bids_channels_none_disables_the_sidecar_like_off(ieeg_in_volts, tmp_path):
    """None is the same "do not look" as ``"off"``; neither is a path."""
    vhdr, _ = ieeg_in_volts
    off = str(tmp_path / "off.zarr")
    none = str(tmp_path / "none.zarr")
    stream_to_zarr(str(vhdr), off, force_modality="IEEG", bids_channels="off")
    stream_to_zarr(str(vhdr), none, force_modality="IEEG", bids_channels=None)

    assert channel_facts(none) == channel_facts(off)


def test_an_explicit_sidecar_path_overrides_the_sibling_one(ieeg_in_volts, tmp_path):
    """A named path is used as given, from wherever it lives."""
    vhdr, _ = ieeg_in_volts
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    named = write_channels_tsv(
        elsewhere, "whatever_the_name", [("LMacro_01", "ECOG", "mV"), ("LMacro_02", "ECOG", "mV")]
    )

    store = str(tmp_path / "named.zarr")
    stream_to_zarr(str(vhdr), store, force_modality="IEEG", bids_channels=str(named))

    facts = channel_facts(store)
    assert {f["unit"] for f in facts.values()} == {"mV"}  # not the sibling's uV
    assert {f["channel_type"] for f in facts.values()} == {"ECOG"}


def test_a_dataframe_sidecar_matches_the_file_it_was_read_from(ieeg_in_volts, tmp_path):
    """An in-memory table is the same input as the TSV on disk."""
    vhdr, _ = ieeg_in_volts
    sidecar = tmp_path / "sub-000_task-clips_run-001_channels.tsv"
    frame = pd.read_csv(sidecar, sep="\t", dtype=str, keep_default_na=False)

    from_path = str(tmp_path / "from_path.zarr")
    from_frame = str(tmp_path / "from_frame.zarr")
    stream_to_zarr(str(vhdr), from_path, force_modality="IEEG", bids_channels=str(sidecar))
    stream_to_zarr(str(vhdr), from_frame, force_modality="IEEG", bids_channels=frame)

    assert channel_facts(from_frame) == channel_facts(from_path)
    assert {f["unit"] for f in channel_facts(from_frame).values()} == {"uV"}


def test_auto_resolves_the_sidecar_inside_a_bids_tree(tmp_path):
    """``"auto"`` finds the sibling sidecar in a real BIDS layout, not just a flat dir."""
    session = tmp_path / "sub-07" / "ses-01" / "ieeg"
    session.mkdir(parents=True)
    stem = "sub-07_ses-01_task-rest_run-03"
    microvolts = signal_microvolts(1)
    vhdr = write_brainvision(
        session, stem, microvolts * 1e-6, unit="V", resolution=1.0, sfreq=SFREQ, names=("A12",)
    )
    # BIDS names the sidecar from the data file's entities minus the suffix.
    write_channels_tsv(session, "sub-07_ses-01_task-rest_run-03", [("A12", "SEEG", "uV")])

    store = str(tmp_path / "auto.zarr")
    stream_to_zarr(str(vhdr), store, force_modality="IEEG")

    fact = channel_facts(store)["A12"]
    assert fact["unit"] == "uV"
    assert fact["channel_type"] == "SEEG"


def test_force_modality_wins_over_the_sidecars_type_for_grouping(ieeg_in_volts, tmp_path):
    """The documented precedence: the suffix groups, the sidecar types.

    ``SEEG`` derives IEEG, so a driver that forced EEG would otherwise find its
    recording split out from under it by the sidecar.
    """
    vhdr, _ = ieeg_in_volts
    store = str(tmp_path / "forced.zarr")
    stream_to_zarr(str(vhdr), store, force_modality="EEG", modality_rates=MODALITY_RATES)

    assert dict(zarr.open_group(store, mode="r").attrs)["channel_groups"] == ["eeg_250hz"]
    fact = channel_facts(store)["LMacro_01"]
    assert fact["modality"] == "EEG"  # forced
    assert fact["channel_type"] == "SEEG"  # still the sidecar's
    assert fact["unit"] == "uV"


def test_discrete_trigger_channels_survive_the_streaming_path(tmp_path):
    """Codes are not a measured quantity: never rescaled, whatever the sidecar says.

    The sidecar is what makes this channel a ``TRIG`` at all (MNE reads a
    BrainVision channel as EEG), so the type has to be adopted *before* the unit
    decision or the exemption never fires -- the same within-row order the
    in-memory path uses.
    """
    stem = "sub-01_task-rest"
    codes = np.array([0.0, 5.0, 3.0, 7.0, 0.0, 5.0, 7.0, 3.0])
    vhdr = write_brainvision(
        tmp_path, stem, codes, unit="V", resolution=1.0, sfreq=SFREQ, names=("STI014",)
    )
    write_channels_tsv(tmp_path, stem, [("STI014", "TRIG", "mV")])

    streamed, in_memory = export_both_ways(vhdr, tmp_path, force_modality=None)
    assert_stores_agree(streamed, in_memory)

    fact = channel_facts(streamed)["STI014"]
    assert fact["channel_type"] == "TRIG"
    assert fact["unit"] == "V"  # the importer's, not the sidecar's mV
    assert fact["bids_unit"] == "mV"  # the declaration is recorded, not asserted
    assert np.allclose(dequantized(streamed)["STI014"], codes, atol=1e-3)

    report = dict(zarr.open_group(streamed, mode="r").attrs)["channels_tsv_units"]
    assert report == {
        "converted": 0,
        "relabelled": 0,
        "kept_importer_unit": 1,
        "units_column_present": True,
    }


def test_an_unconvertible_sidecar_unit_leaves_the_streamed_values_alone(ieeg_in_volts, tmp_path):
    """Different quantities are recorded, never silently rescaled by a guess."""
    vhdr, microvolts = ieeg_in_volts
    write_channels_tsv(
        tmp_path,
        "sub-000_task-clips_run-001",
        [("LMacro_01", "SEEG", "T"), ("LMacro_02", "SEEG", "T")],
    )

    streamed, in_memory = export_both_ways(vhdr, tmp_path, force_modality="IEEG")
    assert_stores_agree(streamed, in_memory)

    fact = channel_facts(streamed)["LMacro_01"]
    assert fact["unit"] == "V"
    assert fact["bids_unit"] == "T"
    expected = microvolts[:, 0] * 1e-6
    step = float(expected.max() - expected.min()) / 65535
    assert np.max(np.abs(dequantized(streamed)["LMacro_01"] - expected)) <= 6 * step


@pytest.mark.skipif(not MEG_FIF.exists(), reason="MEG FIF fixture missing")
def test_meg_femtotesla_sidecar_agrees_across_both_export_paths(tmp_path):
    """305 real MEG channels, a sidecar in fT: a 10^15 conversion, both paths.

    Copies the committed fixture and rewrites only its ``units`` column, so the
    channel names, types and samples are the real recording's and the one thing
    that changed is the claim under test. The two ``TRIG`` rows stay in ``V``,
    which is also what MNE reports for them, so the file exercises a converted
    majority and an untouched discrete minority at once.
    """
    stem = "sub-01_task-mouse"
    shutil.copy(MEG_FIF, tmp_path / f"{stem}_meg.fif")
    sidecar = pd.read_csv(MEG_CHANNELS_TSV, sep="\t", dtype=str, keep_default_na=False)
    assert set(sidecar["units"]) == {"T", "V"}
    sidecar["units"] = sidecar["units"].replace({"T": "fT"})
    sidecar.to_csv(tmp_path / f"{stem}_channels.tsv", sep="\t", index=False)

    streamed, in_memory = export_both_ways(
        tmp_path / f"{stem}_meg.fif", tmp_path, force_modality="MEG"
    )
    assert_stores_agree(streamed, in_memory)

    facts = channel_facts(streamed)
    assert len(facts) == 305
    magnetometers = [f for f in facts.values() if f["channel_type"].startswith("MEG")]
    triggers = [f for f in facts.values() if f["channel_type"] == "TRIG"]
    assert len(magnetometers) == 303
    assert {f["unit"] for f in magnetometers} == {"fT"}
    assert len(triggers) == 2
    assert {f["unit"] for f in triggers} == {"V"}  # already agreed; nothing to convert

    report = dict(zarr.open_group(streamed, mode="r").attrs)["channels_tsv_units"]
    assert report["converted"] == 303
    assert report["kept_importer_unit"] == 0

    # A femtotesla reading of a MEG sensor is an order 1e2-1e4 number; the same
    # channel in SI tesla is 1e-13. Without the conversion the store would carry
    # the second under the label of the first.
    magnitudes = [
        float(np.std(values))
        for label, values in dequantized(streamed).items()
        if facts[label]["channel_type"].startswith("MEG")
    ]
    assert min(magnitudes) > 1.0
