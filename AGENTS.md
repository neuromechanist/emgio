# biosigIO Instructions

Shared instructions for every coding agent (Claude Code, Codex, Copilot, and others). `CLAUDE.md` imports this file, so cross-agent rules live here and only Claude-specific guidance goes in `CLAUDE.md`.

## Project Overview

**Purpose:** a Python package that imports, exports, and manipulates electrophysiology recordings (EMG, EEG, iEEG, MEG) through one interface, with metadata preserved across formats.

**Tech stack:** Python 3.11+, numpy, pandas, scipy, matplotlib, pyedflib, wfdb, pyxdf, plus optional extras for MNE, Neo, Parquet/Arrow, and Zarr. Setuptools build backend, Ruff for lint and format, Ty for types, pytest for tests, MkDocs for documentation.

**Supported formats**
- **Import:** EEGLAB `.set`, Delsys Trigno, OTB, EDF/BDF(+), WFDB, CSV, XDF, MEG (`.fif`, CTF `.ds`, via MNE), BrainVision (`.vhdr`, via MNE), proprietary formats through python-neo (Intan, Blackrock, Spike2, Plexon, Micromed, Neuralynx), and the biosigIO serialization formats (Parquet, Arrow/Feather, Zarr).
- **Export:** EDF/BDF(+) with automatic format selection; Parquet and Arrow/Feather (lossless columnar, `arrow` extra); Zarr (cloud-native serving store, `zarr` extra).

**Architecture map**
```
biosigio/
├── core/          # Recording, the central data class
├── importers/     # Format-specific import logic
├── exporters/     # Format-specific export logic
├── analysis/      # Signal analysis routines
├── visualization/ # Plotting and signal display
├── utils/         # Shared helpers
└── tests/         # Real-data tests only, no mocks
```

**Key classes:** `Recording` (`biosigio/core`) is the main data class. Importers (`biosigio/importers`) and exporters (`biosigio/exporters`) hold the format-specific logic.

## Commands

```bash
uv sync --extra dev                              # Editable install with dev dependencies
uv run pytest                                    # Tests with the default options
uv run pytest -o addopts=""                      # Tests without coverage (faster)
uv run pytest --cov=biosigio                     # Tests with coverage
uv run ruff check --fix . && uv run ruff format .   # Lint and format locally
uvx ruff@0.15.4 check . && uvx ruff@0.15.4 format --check .   # CI gate (ruff is pinned in CI)
uv run ty check biosigio                         # Type check (CI)
uvx codespell --skip='.git*,.venv,*.svg,*.ipynb,*.dat,*.edf,*.bdf,tests/data,examples' --ignore-words-list='OT' --check-hidden   # Spelling (CI)
uv sync --extra docs && uv run mkdocs serve      # Documentation site
```

## Development Workflow
1. **Check context:** read `.context/plan.md` for current tasks and `.context/ideas.md` for design decisions.
2. **Research if needed:** record findings in `.context/research.md`.
3. **Branch:** `gh issue develop <issue-number>` or `feature/short-description`. When a PR is stacked on another PR's branch, keep that base branch until the stacked PR is retargeted, because deleting it closes the PR.
4. **Code:** follow the patterns in the existing importers and exporters (`.rules/python.md`).
5. **Test:** use real recordings only (see Test Data and Examples below and `.rules/testing.md`).
6. **Document failures:** log them in `.context/scratch_history.md` immediately.
7. **Commit:** atomic, subject under 50 characters, no emojis, no AI attribution.
8. **PR:** reference the context and the issue. Run `/review-pr` after creating it, and address every finding (no technical debt carried forward).

## Core Principles

### No mocks: test reality only
- Use real recordings from `examples/` and real format conversions. No mocks, stubs, or fake data in tests.
- Verify round-trip integrity (import, export, import) and metadata preservation.
- Details: `.rules/testing.md`.

### Signal integrity
- Preserve metadata during conversions. Metadata loss is data loss.
- Automatic EDF/BDF format selection based on dynamic range.
- Maintain annotations and events across formats.
- Channel information must be preserved.
- Consider sampling rates and bit depths, and document format limitations.

### Git
- Atomic commits with focused changes; messages under 50 characters; no emojis; no AI attribution.
- Feature branches for multi-step work. Details: `.rules/git.md`.

### Documentation
- Examples over explanations. Document every supported format and its limitations. Include usage examples in docstrings. Details: `.rules/documentation.md`.

### Never
- Never use mocks, stubs, or fake data in tests.
- Never use `pip`, `conda`, or `virtualenv`; use uv for all Python work.
- Never commit secrets, `.env` files, or credentials.
- Never leave empty catch blocks or silent failures.
- Never add backward-compatibility shims; replace directly.
- Never add a TODO without a linked issue.
- Never use emojis in commits, PRs, or code.

### Think like a signal processing engineer
- Signal integrity is paramount. Test with real physiological data.
- Extract repeated patterns (three or more uses) into rules. See `.rules/self_improve.md`.

## Test Data and Examples

The committed example recordings live in `examples/` at the repository root. Tests read them directly; they are the shared test-data directory.

**Provenance is required for every example.** Each example directory has a `README.md`, a `NOTICE.md`, or a BIDS `dataset_description.json` that records:
- the source dataset title;
- the source DOI (for example, the OpenNeuro DOI) and, when NEMAR hosts the copy, the NEMAR DOI (`10.82901/nemar.<id>`);
- the license, as the source records it;
- how the example was derived: excerpt length, downsampling, channel subset, truncation, or conversion tool;
- any synthetic content, labeled as synthetic.

**Owner decisions**
- DOIs are included for attribution. A NEMAR DOI carries the dataset identifier, so identifiers appear only inside DOIs.
- Documentation never records subject information, checksums, or byte counts of recordings.

**Adding an example**
- Look first. An existing example usually covers the scenario; add a new one only when none does.
- Keep new examples small (under 4 MB each).
- Prefer a CC0 or CC-BY source, with its provenance file in the same directory.
- For split chains, truncations, and late-onset cuts, derive the variant inside the test (in `tmp_path`) with the real writer or a byte-exact cut. Do not commit derived variants.
- The owner asked for one-minute exemplars of BrainVision and FIF recordings. See Known Gaps.

**Real data that cannot be committed**
- A recording too large to commit, or under a license that conflicts with this BSD-3-Clause repository, is fetched on demand through `biosigio/tests/real_data.py` (see Testing Requirements).
- The anonymous late-onset EDF+C recording behind biosigio #145 (CC BY-SA 4.0) is never committed. Its test is opt-in through `BIOSIGIO_EDF_TIMEKEEPING_FILE`.

**Inventory (verified 2026-10-16)**

| Path | Format | Facts | Source and license | DOI |
|---|---|---|---|---|
| `examples/bids/eeg/` | EEGLAB `.set` with BIDS sidecars | 60 s at 250 Hz (RecordingDuration 60.0) | OpenNeuro ds006921, CC0 | 10.18112/openneuro.ds006921.v1.1.1; NEMAR 10.82901/nemar.on006921 |
| `examples/bids/meg/` | FIF `sub-01_task-mouse_meg.fif` | 30 s excerpt (per the dataset description) | OpenNeuro ds002908, CC0 | 10.18112/openneuro.ds002908.v1.0.0; NEMAR 10.82901/nemar.on002908 |
| `examples/bids/ieeg/` | EDF+C `sub-01_ses-postimp_task-stim_run-08_ieeg.edf` | 1 s (one record), 65 signals; a truncated fixture | OpenNeuro ds004819, CC0 | 10.18112/openneuro.ds004819.v1.0.0; NEMAR 10.82901/nemar.on004819 |
| `examples/bids/emg/` | EDF (plain, no annotations) | 30 s, 7 signals; MUniverse conversion of NEMAR nm000165 | nm000165; license recorded as "CC0 BY 4.0" in the dataset description and "CC0" in the README | 10.1113/JP287913 (paper); 10.7910/DVN/ID1WNQ (dataset); NEMAR 10.82901/nemar.nm000165 |
| `examples/brainvision/` | BrainVision `.vhdr`, `.vmrk`, `.eeg` | 5 s, 8 channels, 250 Hz, IEEE_FLOAT_32; three synthetic markers (labeled in the README) | derived from `examples/bids/eeg/` (CC0) | as for `examples/bids/eeg/` |
| `examples/verification_output.edf`, `examples/verification_output.bdf` | EDF+C, BDF+C | 20 s, 3 signals; written by `examples/verification_example.py` from `one_sessantaquattro_truncated.otb+` | not recorded | not recorded |
| `examples/kit/` | KIT `.sqd` | 193 channels, 1000 Hz, 0.1 s; bytes unmodified | MNE-Python test suite, BSD-3-Clause | not recorded |
| `examples/bti/` | BTi (`c,rfDC`, `config`, `hs_file`) | bytes unmodified | MNE-Python test suite, BSD-3-Clause | not recorded |
| `examples/ctf/` | CTF `.ds` | bytes unmodified | MNE testing data; no explicit upstream license | not recorded |

**Known gaps** (close these before any new use of the affected files)
- The BrainVision example (5 s) and the FIF example (30 s) are shorter than the one-minute target.
- Source and license are not recorded for `examples/100.dat` (with `100.hea` and `100.atr`), `one_sessantaquattro_truncated.otb+`, `two_mouvi_truncated.otb+`, `multi_stream_test.xdf`, `test.xdf`, `truncated_emg_sample_zenodo7668251.txt`, `truncated_trigno_sample.csv`, and `wristbandEMG_truncated.set`. The verification outputs depend on the `.otb+` file, so they inherit this gap.
- The verification outputs have no provenance file of their own.
- The MNE-vendored (KIT, BTi) and CTF examples have no DOI recorded. The CTF files have no explicit upstream license, which is an owner decision.
- The EMG example's license is recorded two ways (CC0 and CC0 BY 4.0). Reconcile it before citing.
- No committed real late-onset EDF+C recording exists. Late-onset coverage uses derived cuts and the opt-in test.

## Testing Requirements
- Test each importer with real recordings.
- Verify round-trip conversions (import, export, import).
- Check metadata preservation.
- Validate signal integrity.
- A test that needs a recording too large to commit fetches it through `biosigio/tests/real_data.py`. It skips unless `BIOSIGIO_REAL_DATA=1` is set, and it caches outside the repo (`~/.cache/biosigio/real_data`, overridable with `BIOSIGIO_REAL_DATA_CACHE`). Add a `fetch_real_recording` call there, never a second download or cache helper, and pass the file's `sha256` so a corrupted cache fails loudly.

## Release and Landing
- **Version bump:** every behavior change carries a patch version bump, committed in the PR or as the release commit. Tests-only and docs-only PRs do not bump.
- **Bump commit** (subject `chore: bump version to X.Y.Z`, the repository's convention): `biosigio/version.py` (both `__version__` and `__version_info__`), `pyproject.toml`, and `CITATION.cff` (`version` and `date-released`).
- **Changelog:** [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The release commit (subject `Release X.Y.Z in the changelog`) renames `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD`, adds a short release note, and replaces the `[Unreleased]` footer link with `[X.Y.Z]`. The date is the UTC day the GitHub Release is published.
- **Landing:** merge with a merge commit, as #150 and #152 were. Do not delete a branch that another open PR is stacked on; retarget that PR first.
- **Publishing:** `gh release create vX.Y.Z --target <full 40-character merge SHA>`; a short SHA is rejected. The release event publishes to PyPI. The tag push publishes to TestPyPI with `skip-existing`. The workflow fails when the tag does not match `biosigio/version.py`. A TestPyPI 503 during the trusted-publishing exchange is not a release failure; rerun the failed job.
- **Verify:** `uv pip install --refresh "biosigio==X.Y.Z"` in a clean environment. Right after publishing, uv's index can lag for minutes, so an early "no version" error is not a failed publish.

## Rules and Context Files
- **Rules directory:** `.rules/testing.md` (the no-mock policy), `.rules/python.md` (style, Ruff, type hints), `.rules/git.md` (commits and branches), `.rules/documentation.md` (MkDocs), `.rules/ci_cd.md` (GitHub Actions), `.rules/self_improve.md` (learning from projects).
- **Context files:** `.context/plan.md` (current tasks and phases), `.context/ideas.md` (design concepts), `.context/research.md` (technical explorations), `.context/scratch_history.md` (failed attempts and lessons), `.context/decisions/` (architecture decision records; copy `0000-template.md` to start).

## Documentation Site
- https://neuromechanist.github.io/biosigio/
- Update `docs/` when adding features, and include API documentation.

---
Remember: you are handling biomedical signals. Accuracy and data integrity are critical. Check `.rules/` for detailed guidance on any topic.
