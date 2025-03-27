# Contributing to EMGIO

We welcome contributions to EMGIO! This guide will help you get started with the development process.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/neuromechanist/emgio.git
cd emgio
```

2. Install for development:
```bash
pip install -e .
```

3. Install test dependencies:
```bash
pip install -r test-requirements.txt
```

## Code Structure

The EMGIO codebase is organized as follows:

```
emgio/
├── emgio/              # Main package directory
│   ├── __init__.py     # Package initialization
│   ├── core/           # Core functionality
│   │   ├── emg.py      # Main EMG class
│   ├── importers/      # Data import modules
│   │   ├── base.py     # Base importer class
│   │   ├── trigno.py   # Trigno importer
│   │   ├── eeglab.py   # EEGLAB importer
│   │   ├── otb.py      # OTB importer
│   │   ├── edf.py      # EDF importer
│   ├── exporters/      # Data export modules
│   │   ├── edf.py      # EDF/BDF exporter
│   ├── utils/          # Utility functions
│   ├── analysis/       # Analysis methods
│   ├── tests/          # Test suite
├── examples/           # Example scripts
├── docs/               # Documentation
```

## Testing

### Running Tests

EMGIO uses pytest for testing. To run the full test suite:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=emgio
```

### Writing Tests

When adding new features, please include appropriate tests. Tests should be placed in the `emgio/tests` directory, with a naming convention of `test_*.py`.

Example test:

```python
def test_channel_selection():
    # Create a test EMG object
    ...
    
    # Run the function being tested
    ...
    
    # Assert the expected outcome
    assert ...
```

## Documentation

### Building Documentation

Documentation is built using MkDocs with the Material theme:

1. Install documentation dependencies:
```bash
pip install -r docs/requirements.txt
```

2. Build and serve the documentation locally:
```bash
mkdocs serve
```

3. Open your browser at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### Writing Documentation

- API documentation is generated automatically from docstrings
- Use NumPy-style docstrings for all functions and classes
- Add examples to the examples directory and reference them in the documentation
- Update the appropriate section of the user guide when adding new features

## Pull Request Process

1. Fork the repository on GitHub
2. Create a feature branch (`git checkout -b feature/my-new-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest`)
5. Make sure the documentation builds without errors (`mkdocs build`)
6. Commit your changes (`git commit -am 'Add new feature'`)
7. Push to your branch (`git push origin feature/my-new-feature`)
8. Create a new Pull Request on GitHub

## Coding Style

EMGIO follows PEP 8 guidelines. We recommend using tools like flake8 or pylint to check your code style before submitting a pull request.

Some specific style guidelines:
- Use 4 spaces for indentation (not tabs)
- Maximum line length of 100 characters
- Use docstrings for all public methods and classes
- Use meaningful variable and function names

## Adding New Importers

To add support for a new EMG format:

1. Create a new file in `emgio/importers/` (e.g., `new_format.py`)
2. Implement a class that inherits from `BaseImporter`
3. Implement the required methods
4. Add the new importer to `__init__.py`
5. Add tests in `tests/test_importers.py`
6. Document the new format in the user guide

## License

By contributing to EMGIO, you agree that your contributions will be licensed under the BSD 3-Clause License. 