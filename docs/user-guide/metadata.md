# Metadata Handling

EMGIO provides comprehensive metadata management capabilities, allowing you to work with recording session information, subject details, and other contextual data. Proper metadata handling is particularly important when working with research data that needs to be shared or archived.

## Accessing Metadata

When you load data into EMGIO, any available metadata from the source file is automatically imported:

```python
# Load data
emg = EMG.from_file('data.set', importer='eeglab')

# Access all metadata
all_metadata = emg.metadata
print(all_metadata)

# Access specific metadata field
subject = emg.get_metadata('subject')
print(f"Subject: {subject}")

# Check if a metadata field exists
if emg.has_metadata('condition'):
    condition = emg.get_metadata('condition')
    print(f"Condition: {condition}")
```

## Setting Metadata

You can add or modify metadata fields:

```python
# Set a single metadata field
emg.set_metadata('subject', 'S001')

# Set multiple metadata fields at once
emg.set_metadata_dict({
    'subject': 'S001',
    'condition': 'resting',
    'experimenter': 'John Doe',
    'recording_date': '2023-01-15'
})
```

## Common Metadata Fields

While EMGIO is flexible about what metadata you can store, some common fields include:

| Field | Description | Example |
|-------|-------------|---------|
| `subject` | Subject identifier | `"S001"` |
| `session` | Session identifier | `"1"` |
| `condition` | Experimental condition | `"rest"` |
| `recording_date` | Date of recording | `"2023-01-15"` |
| `device` | Recording device | `"Trigno Wireless"` |
| `srate` | Sampling rate in Hz | `2000` |

## Metadata in Exported Files

When exporting to EDF/BDF, EMGIO automatically includes metadata in the file header and generates a sidecar channels.tsv file with channel-specific metadata following BIDS conventions:

```python
# Export to EDF with metadata
emg.to_edf('output')

# This will create:
# - output.edf or output.bdf (depending on format selection)
# - output.channels.tsv (channel metadata in tab-separated format)
```

The channels.tsv file will include information like:

```
name    type    units   sampling_frequency    ...
EMG1    EMG     µV      2000                  ...
EMG2    EMG     µV      2000                  ...
ACC1    ACC     g       2000                  ...
```

## Copying Metadata Between EMG Objects

When working with multiple EMG objects, you can copy metadata between them:

```python
# Create a subset with only EMG channels
emg_only = emg.select_channels(channel_type='EMG')

# Copy all metadata from original to subset
emg_only.metadata = emg.metadata.copy()

# Or selectively copy metadata
emg_only.set_metadata('subject', emg.get_metadata('subject'))
```

## Best Practices for Metadata

1. **Consistency**: Establish conventions for metadata fields and stick to them
2. **Completeness**: Include all relevant information about the recording context
3. **Standardization**: Use standard units and nomenclature
4. **Validation**: Verify metadata accuracy before export
5. **Documentation**: Document your metadata structure for collaborators

By maintaining good metadata practices, you ensure that your EMG data remains interpretable and useful for future analysis. 