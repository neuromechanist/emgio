# Installation

There are several ways to install EMGIO depending on your needs.

## From PyPI (Recommended)

The easiest way to install EMGIO is from PyPI:

```bash
pip install emgio
```

## From GitHub Repository

To install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/neuromechanist/emgio.git
```

Or clone and install:

```bash
git clone https://github.com/neuromechanist/emgio.git
cd emgio
pip install .
```

## Development Installation

For development purposes, install EMGIO in editable mode with dev dependencies:

```bash
git clone https://github.com/neuromechanist/emgio.git
cd emgio
pip install -e ".[dev]"
```

This allows you to modify the source code and see the changes without reinstalling.

## Optional Dependencies

EMGIO provides optional dependency groups:

```bash
# Development tools (pytest, ruff, coverage)
pip install emgio[dev]

# Documentation tools (mkdocs, mkdocstrings)
pip install emgio[docs]

# All optional dependencies
pip install emgio[all]
```

## Dependencies

EMGIO has the following core dependencies:

| Dependency    | Purpose                | Minimum Version |
|---------------|------------------------|-----------------|
| numpy         | Array operations       | >=1.20.0        |
| pandas        | Data manipulation      | >=1.3.0         |
| scipy         | Signal processing      | >=1.7.0         |
| matplotlib    | Visualization          | >=3.4.0         |
| pyedflib      | EDF/BDF file handling  | >=0.1.30        |
| wfdb          | WFDB format support    | >=4.0.0         |

These dependencies are automatically installed when you install EMGIO.

## Python Version

EMGIO requires Python 3.11 or later. We test on Python 3.11, 3.12, 3.13, and 3.14.

## Testing

To run the tests:

```bash
pip install -e ".[dev]"
pytest
```

## Verifying Installation

Verify that EMGIO is correctly installed:

```python
import emgio
print(emgio.__version__)
```

You should see the version number (e.g., `0.2.0`) without any errors.
