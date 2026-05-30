# EMGIO Instructions

## Project Context
**Purpose:** A Python package for electromyography (EMG) data import/export and manipulation, with a unified interface across various EMG systems.
**Tech Stack:** Python 3.11+, numpy, pandas, scipy, matplotlib, pyedflib, wfdb, pyxdf
**Architecture:** Modular importers/exporters with automatic format detection and metadata preservation. Setuptools build backend; ruff for lint/format; pytest for tests.

## Architecture Map
```
emgio/
├── core/          # EMG main class — central data handling
├── importers/     # Format-specific import logic (Trigno, EDF/BDF, WFDB, EEGLAB, OTB, CSV)
├── exporters/     # Format-specific export (EDF/BDF(+) with auto format selection)
├── analysis/      # Signal analysis routines
├── visualization/ # Plotting and signal display
├── utils/         # Shared helpers
└── tests/         # Real tests only — real EMG data, no mocks
```

## Environment Setup
```bash
uv sync --extra dev          # Install package (editable) + dev dependencies
uv run pytest                # Run tests
uv run pytest --cov=emgio    # Run tests with coverage
uv sync --extra docs         # Install docs dependencies
uv run mkdocs serve          # Build/serve documentation
```

## Development Workflow
1. **Check context:** Review .context/plan.md for current tasks
2. **Understand deeply:** Check .context/ideas.md for design decisions
3. **Research if needed:** Update .context/research.md with findings
4. **Branch:** `gh issue develop <issue-number>` (or `feature/short-description`)
5. **Code:** Follow patterns in existing importers/exporters (see .rules/python.md)
6. **Test:** Real EMG data only — see tests/ and .rules/testing.md
7. **Document failures:** Log in .context/scratch_history.md immediately
8. **Commit:** Atomic, <50 chars, no emojis, no AI attribution
9. **PR:** Reference context and issue
10. **Code review:** Run `/review-pr` after creating PR; address all findings (no technical debt carried forward)

## [CRITICAL] Core Principles - Never Compromise

### [FUNDAMENTAL] NO MOCKS - Test Reality Only
- Use real EMG data files from `examples/` (the shared test-data directory)
- Test with actual format conversions
- Verify round-trip integrity (import → export → import)
**Details:** .rules/testing.md

### Signal Integrity
- Preserve metadata during conversions — metadata loss is data loss
- Automatic EDF/BDF format selection based on dynamic range
- Maintain annotations and events across formats
- Channel information must be preserved
- Consider sampling rates and bit depths; document format limitations

### Commits & Git
- Atomic commits, focused changes
- Messages <50 chars, no emojis, no AI attribution
- Feature branches for multi-step work
**Details:** .rules/git.md

### Documentation
- Examples > explanations
- Document all supported formats and their limitations
- Include usage examples in docstrings
**Details:** .rules/documentation.md

## [NEVER DO THIS]
- Never use mocks, stubs, or fake data in tests
- Never use `pip`, `conda`, or `virtualenv`; use UV for all Python work
- Never commit secrets, .env files, or credentials
- Never leave empty catch blocks or silent failures
- Never add backward-compatibility shims; replace directly
- Never add TODO without a linked issue
- Never use emojis in commits, PRs, or code

## Think Like a Signal Processing Engineer
- Signal integrity is paramount
- Metadata loss is data loss
- Test with real physiological data
- Extract repeated patterns (3+ uses) into rules
**See:** .rules/self_improve.md for the learning process

## [REFERENCE] Rules Directory

### Core Standards
- `.rules/testing.md` - Complete NO MOCK policy
- `.rules/self_improve.md` - Learning from projects
- `.rules/documentation.md` - MkDocs setup

### Language & Tools
- `.rules/python.md` - Style, linting (ruff), type hints
- `.rules/ci_cd.md` - GitHub Actions setup
- `.rules/git.md` - Commit and branching conventions

## Context Files
- `.context/plan.md` - Current tasks and phases
- `.context/ideas.md` - Design concepts
- `.context/research.md` - Technical explorations
- `.context/scratch_history.md` - Failed attempts and lessons
- `.context/decisions/` - Architecture Decision Records (copy `0000-template.md` to start)

## Quick Commands
```bash
uv run pytest --cov=emgio                          # Run tests with coverage
uv run ruff check --fix . && uv run ruff format .  # Lint + format
uv run mkdocs serve                                # Build docs
```

## Project-Specific Guidelines

### Supported Formats
- **Import:** EEGLAB .set, Delsys Trigno, OTB, EDF/BDF(+), WFDB, CSV
- **Export:** EDF/BDF(+) with automatic format selection

### Key Classes
- `EMG` (emgio/core) — main class for data handling
- Importers (emgio/importers) — format-specific import logic
- Exporters (emgio/exporters) — format-specific export logic

### Testing Requirements
- Test each importer with real data files
- Verify round-trip conversions (import → export → import)
- Check metadata preservation
- Validate signal integrity

### Documentation Site
- https://neuromechanist.github.io/emgio/
- Update docs/ when adding features; include API documentation

---
Remember: You're handling biomedical signals. Accuracy and data integrity are critical.
Check .rules/ for detailed guidance on any topic.
