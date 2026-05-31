# Jupyter Notebook Examples

biosigIO can be used effectively in Jupyter Notebooks for interactive data analysis and visualization. This page provides an overview of working with biosigIO in Jupyter environments.

## Available Example Notebooks

The biosigIO repository includes several example Jupyter notebooks that demonstrate different aspects of working with EMG data:

1. **Trigno Example Notebook**: Working with Delsys Trigno data
   - Location: `examples/trigno_example.ipynb`

2. **MATLAB Example Notebook**: Working with MATLAB-exported data
   - Location: `examples/matlab_example.ipynb`

## Running the Example Notebooks

To run the example notebooks:

1. Install biosigIO in development mode:
   ```bash
   uv sync --extra dev
   ```

2. Start Jupyter (no separate install needed):
   ```bash
   uv run --with jupyter jupyter notebook
   ```

3. Navigate to the `examples` directory and open one of the notebooks.

## Key Features for Jupyter Notebooks

When working with biosigIO in Jupyter notebooks, you can take advantage of these features:

### Interactive Plotting

```python
from biosigio import Recording
import matplotlib.pyplot as plt
%matplotlib inline  # For displaying plots in the notebook

# Load EMG data
emg = Recording.from_file('data.csv', importer='trigno')

# Plot signals with customized appearance
emg.plot_signals(
    channels=['EMG1', 'EMG2'],
    time_range=(0, 5),
    title='EMG Signals',
    figsize=(12, 6),
    grid=True
)
```

### Data Exploration

Jupyter notebooks are excellent for interactively exploring your EMG data:

```python
# Load data
emg = Recording.from_file('data.otb+', importer='otb')

# Display channel information
channel_types = emg.get_channel_types()
for ch_type in channel_types:
    channels = emg.get_channels_by_type(ch_type)
    print(f"{ch_type} channels: {len(channels)}")
    
    # Show details of the first channel
    if channels:
        ch_name = channels[0]
        ch_info = emg.channels[ch_name]
        print(f"  Sample channel: {ch_name}")
        for key, value in ch_info.items():
            print(f"    {key}: {value}")
```

### Creating Interactive Widgets

You can use ipywidgets to create interactive controls for exploring your data:

```python
import ipywidgets as widgets
from IPython.display import display

# Load data
emg = Recording.from_file('data.set', importer='eeglab')

# Get all channels
all_channels = list(emg.channels.keys())

# Create widgets
channel_dropdown = widgets.Dropdown(
    options=all_channels,
    description='Channel:',
    disabled=False,
)

time_slider = widgets.FloatRangeSlider(
    value=[0, 5],
    min=0,
    max=emg.get_duration(),
    step=1,
    description='Time (s):',
    disabled=False,
    continuous_update=False,
    readout=True,
    readout_format='.1f',
)

# Define update function
def update_plot(channel, time_range):
    plt.figure(figsize=(12, 6))
    emg.plot_signals(
        channels=[channel],
        time_range=time_range,
        title=f'Channel: {channel}, Time: {time_range[0]}-{time_range[1]}s',
        grid=True
    )
    plt.show()

# Create interactive output
out = widgets.interactive_output(
    update_plot, 
    {'channel': channel_dropdown, 'time_range': time_slider}
)

# Display widgets and output
display(widgets.HBox([channel_dropdown, time_slider]), out)
```

### Exporting Figures

You can easily export high-quality figures for publications:

```python
# Create a figure
plt.figure(figsize=(10, 6), dpi=300)  # High DPI for publication quality
emg.plot_signals(
    channels=emg.get_channels_by_type('EMG')[:4],  # First 4 EMG channels
    time_range=(10, 15),
    title='EMG Signals During Movement',
    grid=True
)
plt.tight_layout()

# Save the figure in various formats
plt.savefig('emg_plot.png', dpi=300)  # PNG for presentations
plt.savefig('emg_plot.pdf')  # PDF for publications
plt.savefig('emg_plot.svg')  # SVG for editing
```

## Tips for Jupyter Notebooks

1. **Memory management**: When working with large files, consider using:
   ```python
   # Load only metadata first
   importer = EDFImporter('large_file.edf')
   signals, channels, metadata = importer.load(load_data=False)
   
   # Then load specific time segments as needed
   importer = EDFImporter('large_file.edf')
   signals, channels, metadata = importer.load(start_time=10, end_time=20)
   ```

2. **Progress tracking**: For long operations, use `tqdm` for progress bars:
   ```python
   from tqdm.notebook import tqdm
   
   channel_stats = {}
   for channel in tqdm(emg.channels.keys()):
       # Perform analysis on each channel
       signal = emg.signals[channel].values
       channel_stats[channel] = {
           'mean': signal.mean(),
           'std': signal.std(),
           'max': signal.max(),
           'min': signal.min()
       }
   ```

3. **Table display**: Format channel information as a dataframe for better visualization:
   ```python
   import pandas as pd
   
   # Convert channel info to DataFrame
   channel_df = pd.DataFrame()
   for ch_name, ch_info in emg.channels.items():
       ch_data = pd.Series(ch_info)
       ch_data.name = ch_name
       channel_df = channel_df.append(ch_data)
   
   # Display as an interactive table
   channel_df
   ```

These examples show how to leverage Jupyter notebooks for interactive exploration and analysis of EMG data with biosigIO. 