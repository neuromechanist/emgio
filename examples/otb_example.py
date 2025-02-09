import os
import matplotlib.pyplot as plt
from emgio.core.emg import EMG


def main():
    # Load OTB data
    print("Loading OTB data...")
    emg = EMG.from_file('examples/two_mouvi_truncated.otb+', importer='otb')

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
        ch_type = ch_info['type']
        if ch_type not in channel_types:
            channel_types[ch_type] = 1
        else:
            channel_types[ch_type] += 1

    for ch_type, count in channel_types.items():
        print(f"{ch_type}: {count} channels")

    # # Plot EMG channels
    # emg_data = emg.select_channels(channel_type='EMG')
    # if emg_data.signals is not None and not emg_data.signals.empty:
    #     print("\nPlotting EMG channels...")
    #     emg_data.plot_signals(
    #         title='EMG Channels',
    #         style='line',
    #         grid=True
    #     )
    #     plt.show()
    # else:
    #     print("\nNo EMG channels found in the data")

    output_path = 'examples/otb_emg'
    emg.to_edf(output_path)

    print("\nExport complete!")

if __name__ == '__main__':
    main()
