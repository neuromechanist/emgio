# Installation

There are several ways to install EMGIO depending on your needs.

## From GitHub Repository

The recommended way to install the latest version of EMGIO is directly from the GitHub repository:

```bash
git clone https://github.com/neuromechanist/emgio.git
cd emgio
pip install .
```

## Development Installation

For development purposes, you can install EMGIO in editable mode:

```bash
git clone https://github.com/neuromechanist/emgio.git
cd emgio
pip install -e .
```

This allows you to modify the source code and see the changes without reinstalling.

## Dependencies

EMGIO has the following dependencies:

| Dependency    | Purpose                | Minimum Version |
|---------------|------------------------|----------------|
| numpy         | Array operations       | -              |
| pandas        | Data manipulation      | -              |
| scipy         | Signal processing      | -              |
| matplotlib    | Visualization          | -              |
| pyedflib      | EDF/BDF file handling  | -              |

These dependencies will be automatically installed when you install EMGIO.

## Testing

To run the tests, you need to install the test dependencies:

```bash
pip install -r test-requirements.txt
```

Then you can run the tests with:

```bash
pytest
```

## Verifying Installation

You can verify that EMGIO is correctly installed by running:

```python
import emgio
print(emgio.__version__)
```

If you don't see any error messages, EMGIO is installed correctly. 