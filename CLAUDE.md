@AGENTS.md

## Claude Code Specific Instructions

The shared project instructions live in `AGENTS.md`; this file imports them for Claude Code with `@AGENTS.md`. Keep cross-agent project rules in `AGENTS.md` so Codex, Copilot, Claude Code, and other AGENTS.md-aware tools stay aligned. Append only Claude-specific guidance below.

### Workflow skills
- `/review-pr` — run after creating a PR; address all findings (no technical debt carried forward), skipping only genuine false positives with an explanation.
- `/plan` — enter plan mode to design a detailed implementation plan before non-trivial work.
- `/epic-dev` — epic/sprint workflow with git worktrees and phased delivery for multi-phase features.

### Conventions for Claude Code
- Run `date` at session start.
- Use UV for all Python work (`uv run pytest`, `uv run ruff ...`); never pip/conda.
- No emojis and no AI attribution in commits, PRs, or code.
