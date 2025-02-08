import os
import matplotlib.pyplot as plt
from emgio.core.emg import EMG


def main():
    # Load OTB data
    print("Loading OTB data...")
    emg = EMG.from_file('examples/t002_60_2s_sessL.otb+', importer='otb')

    # Print metadata
    print("\nDevice Information:")
    print("-" * 50)
    print(f"Device: {emg.get_metadata('device')}")
    print(f"AD Bits: {emg.get_metadata('ad_bits')}")

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

    # Plot EMG channels
    emg_data = emg.select_channels(channel_type='EMG')
    if emg_data.signals is not None and not emg_data.signals.empty:
        print("\nPlotting EMG channels...")
        emg_data.plot_signals(
            title='EMG Channels',
            style='line',
            grid=True
        )
        plt.show()
    else:
        print("\nNo EMG channels found in the data")

    # Try different precision thresholds
    thresholds = [0.1, 0.01, 0.001, 0.0001]
    print("\nTesting different precision thresholds:")
    print("-" * 50)
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}%")
        output_path = f'examples/otb_emg_{threshold:.4f}.edf'
        emg.to_edf(output_path, precision_threshold=threshold)
        
        # Get file info
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            ext = os.path.splitext(output_path)[1]
            print(f"Format: {ext[1:].upper()}")
            print(f"File size: {size_mb:.2f} MB")
        else:
            print(f"File not created: {output_path}")


if __name__ == '__main__':
    main()
