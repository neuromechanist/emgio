import matplotlib.pyplot as plt
from emgio.core.emg import EMG


def main():
    # Load OTB data
    print("Loading OTB data...")
    # example two: one_sessantaquattro_truncated.otb+, example two: two_mouvi_truncated.otb+
    emg = EMG.from_file('examples/one_sessantaquattro_truncated.otb+', importer='otb')

    # Print metadata
    print("\nDevice Information:")
    print("-" * 50)
    print(f"Device: {emg.get_metadata('device')}")
    print(f"Resolution: {emg.get_metadata('signal_resolution')} bits")

    # Print channel type summary
    print("\nChannel Type Summary:")
    print("-" * 50)
    channel_types = {}
    for ch_info in emg.channels.values():
        ch_type = ch_info['channel_type']
        if ch_type not in channel_types:
            channel_types[ch_type] = 1
        else:
            channel_types[ch_type] += 1

    for ch_type, count in channel_types.items():
        print(f"{ch_type}: {count} channels")

    # Create a new EMG object with only EMG channels
    emg_data = emg.select_channels(channel_type='EMG')
    if emg_data.signals is not None and not emg_data.signals.empty:
        print("\nPlotting EMG channels...")
        emg_data.plot_signals(
            title='EMG Channels',
            grid=True,
            channels=list(emg.channels.keys())[33:-1]  # optionally plot a subset of channels
        )
        plt.show()
    else:
        print("\nNo EMG channels found in the data")

    # Export EMG channels to EDF
    output_path = 'examples/otb_emg'
    emg_data.to_edf(output_path)  # Use the EMG-only data for export

    print("\nExport complete!")


if __name__ == '__main__':
    main()
