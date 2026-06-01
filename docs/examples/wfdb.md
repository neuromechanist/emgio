# WFDB Example

This example demonstrates how to load a WFDB record, specifically record `100` from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) available on PhysioNet. This record includes signal data (`.dat`), header information (`.hea`), and annotations (`.atr`).

## Loading Data and Annotations

The `WFDBImporter` automatically detects and loads annotations if the corresponding `.atr` file is found alongside the `.hea` and `.dat` files.

```python title="examples/wfdb_example.py"
--8<-- "examples/wfdb_example.py"
```

## Key Steps Explained

1.  **Import `Recording`:** The core class is imported.
2.  **Define Path:** The path to the WFDB header file (`.hea`) is specified.
3.  **Load Data:** `Recording.from_file()` is called with the header file path. biosigIO infers the importer (`'wfdb'`) based on the `.hea` extension (this could also be explicitly set using `importer='wfdb'`). The importer reads the header, data, and automatically looks for `100.atr` to load annotations.
4.  **Access Metadata:** Basic metadata like `record_name` and `sampling_frequency` are accessed using `rec.get_metadata()`.
5.  **Access Channels:** Information about the loaded channels (`MLII`, `V5`) is iterated through `rec.channels`.
6.  **Access Annotations:** The script checks if the `rec.events` DataFrame is populated. If the `.atr` file was loaded successfully, it prints the first and last few annotations.
7.  **Plotting:** A short segment of the first channel is plotted using `rec.plot_signals()`.
8.  **Exporting:** The loaded data (including annotations) is exported to both EDF+ and BDF+ formats using `rec.to_edf()`. The presence of annotations is noted in the output.

## Running the Example

1.  Ensure you have the necessary files (`100.hea`, `100.dat`, `100.atr`) in the `examples/` directory.
2.  Run the script from the project root directory: `uv run python examples/wfdb_example.py`

You should see output detailing the loaded metadata, channels, annotations, and the paths to the exported EDF and BDF files. A plot window showing the first 10 seconds of the 'MLII' channel will also appear. 