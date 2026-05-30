# CI/CD Workflow Standards

## Purpose: Automated Quality Gates
**Why CI/CD?** Catch issues before users do.
**Think:** Every pipeline failure is a production bug prevented.
**Goal:** Fast feedback, high confidence, zero surprises.

## Essential Workflows

### 1. Testing (`tests.yml`)
**Triggers:** `on: [push, pull_request]` to main branches  
**Jobs (in order):**
- **Lint:** `uv run ruff check` (fails fast)
- **Test:** Real tests only, matrix for versions
- **Build:** Verify compilation if applicable
- **Coverage:** Optional reporting to Codecov

### 2. Documentation (`docs.yml`)
**Triggers:** `on: push: branches: [main]`  
**Jobs:** Build with MkDocs → Deploy to GitHub Pages

### 3. Publish (`publish.yml`)
**Triggers:** Tag creation or manual  
**Jobs:** Build → Create release → Publish packages

## Minimal Python Example (UV)
```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix: { python-version: ['3.12', '3.13', '3.14'] }
    steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v4
    - name: Set up Python ${{ matrix.python-version }}
      run: uv python install ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        uv venv --python ${{ matrix.python-version }}
        uv pip install -e ".[dev]"
    - name: Lint
      run: uv run ruff check .
    - name: Test
      run: uv run pytest --cov=emgio --cov-report=xml
```

## Key Practices (Think About Pipeline Flow)
- **Pin versions:** `actions/checkout@v4`, `astral-sh/setup-uv@v4` (reproducibility)
- **UV everywhere:** Use `astral-sh/setup-uv` + `uv` for installs; never pip/conda
- **Cache deps:** Speed matters for developer happiness
- **Fail fast:** Lint→Test→Build→Deploy (catch cheap failures first)
- **Matrix testing:** Test all supported versions
- **Secrets:** Never commit credentials
- **Conditional:** Deploy only from protected branches

## Pipeline Philosophy
**Fast feedback:** Developers should know in <5 min
**Clear failures:** Error messages should guide fixes
**No surprises:** If it passes CI, it works in production

**Ask yourself:**
- Will this catch real issues?
- Is the feedback loop fast enough?
- Are we testing what actually matters?
