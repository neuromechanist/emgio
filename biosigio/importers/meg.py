"""MEG importer backed by MNE-Python (.fif, CTF .ds, KIT/Yokogawa .con/.sqd/.kdf,
4D Neuroimaging/BTi).

MNE is an optional dependency (it is heavy), so it is imported lazily with a
clear install hint rather than at module import. MEG recordings mix several
sensor types in one file (magnetometers, gradiometers, reference sensors)
alongside stim/EEG/EOG/ECG channels; the shared :mod:`_mne_common` mapping keeps
each type distinct (they are NOT collapsed into a single "MEG" type), and
stim-channel triggers are read into ``Recording.events``.

Formats and their MNE reader:
    ``.fif``                 -> ``read_raw_fif`` (Neuromag / Elekta / MEGIN)
    ``.ds`` (a directory)    -> ``read_raw_ctf`` (CTF / VSM)
    ``.con`` / ``.sqd`` / ``.kdf`` -> ``read_raw_kit`` (KIT / Yokogawa / RICOH)
    4D/BTi directory (no extension) -> ``read_raw_bti`` (see below)

4D/BTi is genuinely a MEG system (a whole-head magnetometer array), so it lives
here alongside CTF/KIT rather than in its own module -- unlike MEF3 (iEEG), which
is a different modality entirely and gets its own module (see
:mod:`biosigio.importers.mef3`). The one wrinkle: BIDS names the BTi recording
directory with NO extension (``sub-<label>[_ses-<label>]_task-<label>[_run-<index>]_meg/``,
containing the processed-data file -- conventionally ``c,rfDC`` -- plus ``config``
and, usually, ``hs_file``), so it cannot be recognized by extension the way
``.ds``/``.con``/``.sqd``/``.kdf`` are. Detection is therefore content-based
(:func:`_find_bti_pdf`): a directory qualifies only if it directly contains both
a ``c,rf*``-prefixed file (the actual PDF data) and a sibling ``config`` file --
checked non-recursively, so an unrelated nested ``config`` (every datalad-tracked
dataset has one at ``.datalad/config``) can never false-positive.

When a directory holds MORE THAN ONE ``c,rf*`` file (e.g. the conventional
unfiltered ``c,rfDC`` alongside a hardware-filtered copy such as
``c,rfDC,fn50,o``), an exact ``c,rfDC`` is always preferred -- a serving copy
should default to the least-processed signal -- with ``c,rf*`` candidates in
``sorted()`` order as a deterministic fallback otherwise. ``os.listdir`` order
is filesystem/OS-dependent and MUST NEVER decide this (verified to matter in
practice: a real directory holding both files returned the filtered copy FIRST
from ``os.listdir`` on at least one filesystem). Whenever the choice falls back
to a non-canonical name, OR more than one candidate exists at all (even if
``c,rfDC`` itself was chosen), a ``logging.warning`` names the file chosen and
the ones skipped, and the importer records which file was actually read in
``Recording.metadata["bti_pdf_file"]`` so a caller can tell what was read without
digging through logs.

Detection (:func:`_find_bti_pdf`) and reader-kwargs resolution
(:func:`_resolve_bti_reader_kwargs`) are used by three call sites --
``Recording._infer_importer`` (auto-detection), :meth:`MEGImporter._read_raw`
(in-memory reads), and :func:`biosigio.exporters.zarr_stream._open_stream_source`
(streaming reads) -- so the precedence rule and the ``config``/``hs_file``
sidecar-resolution rule each live in exactly one place rather than three.
"""

import logging
import os

import numpy as np
import pandas as pd

from ..core.emg import Recording
from ..exceptions import UnsupportedFormatError, classify_read_error, is_resource_exhaustion
from ._mne_common import raw_to_recording, require_mne
from .base import BaseImporter


def _find_bti_pdf(dirpath: str) -> str | None:
    """Return the path to a 4D/BTi processed-data file directly inside ``dirpath``,
    or None if this doesn't look like a BTi recording directory.

    Looks for top-level entries whose name starts with ``c,rf`` (the conventional
    PDF basename is ``c,rfDC``; filtered copies such as ``c,rfDC,fn50,o`` also
    match) alongside a sibling ``config`` file. Both checks are non-recursive and
    look only at ``dirpath`` itself -- never a subdirectory -- so a dataset's
    ``.datalad/config`` (present in almost every datalad-tracked repo) can never
    be mistaken for the BTi sidecar.

    Precedence when more than one ``c,rf*`` candidate exists: see the module
    docstring. A genuine :class:`PermissionError` on ``dirpath`` itself is
    re-raised rather than swallowed (a directory that cannot even be listed is
    not "not BTi", it is a permissions problem the caller needs to see); any
    other :class:`OSError` (missing path, not a directory, ...) is treated as
    "not BTi" and returns None, matching every other extension/content check in
    this codebase.
    """
    try:
        entries = os.listdir(dirpath)
    except PermissionError:
        raise
    except OSError:
        return None
    if "config" not in entries or not os.path.isfile(os.path.join(dirpath, "config")):
        return None
    candidates = sorted(
        name
        for name in entries
        if name.startswith("c,rf") and os.path.isfile(os.path.join(dirpath, name))
    )
    if not candidates:
        return None
    fell_back = "c,rfDC" not in candidates
    chosen = candidates[0] if fell_back else "c,rfDC"
    ambiguous = len(candidates) > 1
    # Warn on EITHER condition independently: falling back to a non-canonical
    # name is worth knowing about even with a single candidate (the data may be
    # filtered), and multiple candidates are worth knowing about even when the
    # canonical 'c,rfDC' was the one chosen (a filtered copy was sitting right
    # next to it).
    if fell_back or ambiguous:
        detail = (
            f"multiple processed-data files ({', '.join(candidates)})"
            if ambiguous
            else (f"a non-canonical processed-data file ({chosen!r})")
        )
        logging.warning(
            "4D/BTi directory %s has %s; reading %r%s.",
            dirpath,
            detail,
            chosen,
            " (no unfiltered 'c,rfDC' present; falling back to the first candidate in sorted order)"
            if fell_back
            else "",
        )
    return os.path.join(dirpath, chosen)


def _resolve_bti_reader_kwargs(dirpath: str) -> dict:
    """Resolve ``dirpath`` into ``mne.io.read_raw_bti`` kwargs, or raise
    :class:`UnsupportedFormatError` if it doesn't look like a 4D/BTi recording.

    This is the ONE place that decides "is this extension-less directory a
    valid BTi layout" and resolves its ``config``/``hs_file`` sidecars; every
    call site (auto-detection, in-memory read, streaming read -- see the module
    docstring) raises through this same function, so there is exactly one error
    message and one precedence rule to keep in sync, not three.

    ``config_fname``/``head_shape_fname`` are returned as the plain BIDS
    basenames (``"config"``/``"hs_file"``); MNE resolves a non-absolute name
    against ``dirname(pdf_fname)`` internally, which is exactly this directory,
    so no path-joining is needed for them. ``hs_file`` is optional in BIDS (not
    every BTi recording ships digitized head-shape points), so it resolves to
    None rather than a guessed filename when absent; MNE raises on a missing
    file it was never told to skip.
    """
    pdf_fname = _find_bti_pdf(dirpath)
    if pdf_fname is None:
        raise UnsupportedFormatError(
            f"{dirpath!r} has no file extension and does not look like a 4D/BTi "
            "recording directory (expected a 'c,rf*' processed-data file "
            "alongside a 'config' file). MEG import supports .fif, CTF .ds, KIT "
            ".con/.sqd/.kdf, and 4D/BTi directories."
        )
    hs_fname = os.path.join(dirpath, "hs_file")
    return {
        "pdf_fname": pdf_fname,
        "config_fname": "config",
        "head_shape_fname": "hs_file" if os.path.isfile(hs_fname) else None,
    }


#: CTF head-coil names, mapped to the coil each one identifies. ``lpa``/``rpa``
#: are an alternate vocabulary for the ear coils, not a casing difference, so
#: they have to be listed rather than derived.
_CTF_COIL_NAMES = {
    "nasion": "CTFV_COIL_NAS",
    "left ear": "CTFV_COIL_LPA",
    "right ear": "CTFV_COIL_RPA",
    "lpa": "CTFV_COIL_LPA",
    "rpa": "CTFV_COIL_RPA",
}


def _patch_ctf_coil_aliases() -> None:
    """Teach MNE's ``.hc`` reader the ``Nasion``/``LPA``/``RPA`` coil spelling.

    MNE classifies each head-coil point by substring-matching the ``.hc``
    descriptor line against a literal, case-sensitive table that only knows
    ``nasion``/``left ear``/``right ear``. Datasets spelling the coils
    ``Nasion``/``LPA``/``RPA`` therefore parse every point as "unknown", and the
    read fails with *"Some of the mandatory HPI device-coordinate info was not
    there."* even though the coordinates are present and correct. CTF's own
    reference reader (``readHc`` in ``readCTFds.m``) takes the coil name
    positionally and does not check it against a vocabulary at all.

    Fixed upstream in mne-tools/mne-python#14191, which both lower-cases the
    match and adds the aliases. Until a release carries that, add the alternate
    spellings to the table MNE already consults. Case-insensitivity cannot be
    injected the same way -- MNE tests ``key in descriptor`` against the raw
    line -- so the plausible casings are enumerated instead. Appending leaves
    the canonical names first, so a conventional ``.hc`` matches as before.

    No-ops when the installed MNE carries the fix, or if the private table it
    patches has moved -- neither is worth failing a read over, since MNE then
    either handles the file or raises its own error (biosigio#117).
    """
    try:
        from mne.io.ctf import hc
        from mne.io.ctf.constants import CTF
    except ImportError:  # pragma: no cover - MNE layout changed
        return
    kind_dict = getattr(hc, "_kind_dict", None)
    if not isinstance(kind_dict, dict) or "lpa" in kind_dict:
        return  # MNE already understands these, or there is nothing to patch
    for name, constant in _CTF_COIL_NAMES.items():
        kind = getattr(CTF, constant, None)
        if kind is None:  # pragma: no cover - MNE constants renamed
            continue
        for spelling in (name, name.title(), name.upper()):
            kind_dict.setdefault(spelling, kind)


class MEGImporter(BaseImporter):
    """Importer for MEG recordings via MNE-Python (.fif, CTF .ds, KIT .con/.sqd/.kdf,
    4D/BTi)."""

    def _read_bti(self, mne, dirpath: str):
        """Read a 4D/BTi recording found in ``dirpath``.

        Returns ``(raw, extra_metadata)``; ``extra_metadata["bti_pdf_file"]``
        records exactly which processed-data file was read (see
        :func:`_resolve_bti_reader_kwargs` for the precedence rule when more
        than one candidate exists), so a caller can tell what was read without
        digging through logs.
        """
        kwargs = _resolve_bti_reader_kwargs(dirpath)
        raw = mne.io.read_raw_bti(preload=True, verbose="ERROR", **kwargs)
        return raw, {"bti_pdf_file": kwargs["pdf_fname"]}

    def _read_raw(self, mne, filepath: str):
        """Read the recording into an MNE Raw object, dispatching by extension
        (or, for 4D/BTi, by directory content -- see :func:`_find_bti_pdf`).

        Returns ``(raw, extra_metadata)``; ``extra_metadata`` is empty for every
        format except 4D/BTi (see :meth:`_read_bti`).

        ``.ds`` is a CTF directory; ``.con``/``.sqd``/``.kdf`` are KIT/Yokogawa
        single files (markers/headshape sidecars are optional and not needed for
        the signal copy); an extension-less directory is checked for a BTi PDF;
        everything else is treated as Neuromag ``.fif``.
        """
        stripped = filepath.rstrip("/\\")
        ext = os.path.splitext(stripped)[1].lower()
        if ext == ".ds":
            _patch_ctf_coil_aliases()
            return mne.io.read_raw_ctf(filepath, preload=True, verbose="ERROR"), {}
        if ext in (".con", ".sqd", ".kdf"):
            return mne.io.read_raw_kit(filepath, preload=True, verbose="ERROR"), {}
        if ext == "" and os.path.isdir(stripped):
            return self._read_bti(mne, stripped)
        return mne.io.read_raw_fif(filepath, preload=True, verbose="ERROR"), {}

    def _read_events(self, mne, raw, sfreq: float) -> pd.DataFrame:
        """Read stim-channel triggers into an events DataFrame (onsets in seconds)."""
        try:
            events = mne.find_events(raw, verbose="ERROR")
        except (ValueError, RuntimeError) as exc:
            # "No stim channels found" is the common, benign case (many MEG
            # recordings carry no triggers, or events come from a BIDS events.tsv
            # applied separately) -- stay quiet for that. Any OTHER failure
            # (malformed/ambiguous triggers) would silently drop real events, so
            # surface it rather than swallowing it.
            if "stim" not in str(exc).lower():
                logging.warning("MEG find_events failed; events left empty: %s", exc)
            events = np.empty((0, 3), dtype=int)
        first_samp = int(raw.first_samp)
        rows = [
            ((int(sample) - first_samp) / sfreq, 0.0, str(int(code)))
            for sample, _prev, code in events
        ]
        evt = pd.DataFrame(rows, columns=["onset", "duration", "description"])
        if not evt.empty:
            evt = evt.sort_values("onset").reset_index(drop=True)
            evt["onset"] = evt["onset"].astype("float64")
            evt["duration"] = evt["duration"].astype("float64")
        return evt

    def load(self, filepath: str) -> Recording:
        """Load a MEG recording into a Recording object.

        Args:
            filepath: Path to a ``.fif`` file, a CTF ``.ds`` directory, a KIT/
                Yokogawa ``.con``/``.sqd``/``.kdf`` file, or a 4D/BTi recording
                directory (no extension; detected by content, see
                :func:`_find_bti_pdf`).

        Returns:
            Recording: channels carry their MEG/EEG/stim type and physical unit; stim
            triggers are read into ``Recording.events``.
        """
        mne = require_mne()
        try:
            raw, extra_metadata = self._read_raw(mne, filepath)
        except Exception as e:
            # Resource exhaustion (MemoryError, thread/allocation-exhaustion
            # OSError/RuntimeError) is a host condition, not a file problem --
            # propagate unchanged rather than reclassifying it as a permanent
            # read failure (see biosigio.exceptions.is_resource_exhaustion).
            if is_resource_exhaustion(e):
                raise
            # Classify into a typed biosigIO error (not_continuous for an
            # evoked/epoched derivative, corrupt_or_truncated for a bad/incomplete
            # file, ...) so callers can surface a specific reason. The Pyright
            # import-resolution noise here is the editor lacking the venv; the
            # module exists.
            raise classify_read_error(e, filepath) from e

        rec = raw_to_recording(raw)
        rec.set_metadata("source_file", filepath)
        for key, value in extra_metadata.items():
            rec.set_metadata(key, value)

        events = self._read_events(mne, raw, float(raw.info["sfreq"]))
        if not events.empty:
            rec.events = events

        return rec
