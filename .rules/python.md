# Python Development Standards

## Version & Environment
- **Python 3.11+** minimum (use latest stable)
- **Environment & Package Management:** UV only (`uv sync`, `uv run`, `uv add`). Never pip, conda, or virtualenv.
- **Dependencies:** declared in `pyproject.toml`

## Code Style
- **Formatter:** `uv run ruff format`
- **Linter:** `uv run ruff check` with aggressive fixes
- **Line Length:** 100 characters (see `pyproject.toml`)
- **Imports:** Sorted with `isort` (via ruff)

## Type Hints
- **Required for:** All public functions and methods
- **Tool:** `ty` for type checking (`uv run ty`)
- **Example:**
```python
def process_data(items: list[dict[str, Any]]) -> pd.DataFrame:
    """Process raw data into DataFrame."""
    ...
```

## Project Structure
```
project/
├── src/project/       # Source code
│   ├── __init__.py
│   └── module.py
├── tests/            # Real tests only
├── pyproject.toml    # Project config
└── .gitignore
```

## Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
if [ -n "$files" ]; then
    ruff check --fix --unsafe-fixes $files
    ruff format $files
    git add $files
fi
```

## Common Patterns
- **Context Managers:** For resource management
- **Dataclasses:** For data structures
- **Pathlib:** For file operations (not os.path)
- **F-strings:** For string formatting

## Error Handling
```python
# Be specific with exceptions
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise  # Re-raise or handle appropriately
```

## Documentation
- **Docstrings:** Google or NumPy style
- **Module docs:** At file top
- **Type hints:** Self-documenting code

---
*Follow PEP 8 with ruff enforcement. Real tests only.*