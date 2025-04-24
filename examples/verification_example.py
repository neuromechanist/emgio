import os
import logging
import sys
from emgio.core.emg import EMG

# Add the project root to the Python path
# This allows importing emgio even if it's not installed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
# Use a file that exists in the examples directory
input_file = "truncated_trigno_sample.csv"
output_file_edf = "verification_output.edf"
output_file_bdf = "verification_output.bdf"
# Try setting verify to True and False
verify_export = True
# Set a tolerance for comparison (e.g., 1e-6 for strict, 1e-4 for looser)
verification_tolerance = 1e-6

# --- Script ---
# Construct full paths
input_path = os.path.join(os.path.dirname(__file__), input_file)
output_path_edf = os.path.join(os.path.dirname(__file__), output_file_edf)
output_path_bdf = os.path.join(os.path.dirname(__file__), output_file_bdf)

logging.info(f"Loading EMG data from: {input_path}")
try:
    # Load the EMG data (assuming it's a CSV identifiable by the importer)
    # You might need to specify the importer='trigno' if auto-detection fails
    emg_data = EMG.from_file(input_path, importer='trigno')
    logging.info("EMG data loaded successfully.")
    logging.info(f"Channels found: {list(emg_data.signals.columns)}")
    logging.info(f"Duration: {emg_data.signals.index[-1]:.2f} seconds")
    emg_channels = [ch for ch, info in emg_data.channels.items() if info['channel_type'] == 'EMG']
    emg_only = emg_data.select_channels(emg_channels)  # Creates a new EMG object with only EMG channels


except FileNotFoundError:
    logging.error(f"Input file not found: {input_path}")
    sys.exit(1)
except Exception as e:
    logging.error(f"Error loading EMG data: {e}")
    sys.exit(1)

# --- Export to EDF with Verification ---
logging.info(f"\\n--- Exporting to EDF ({output_file_edf}) ---")
try:
    verification_results_edf = emg_only.to_edf(
        output_path_edf,
        format='edf',  # Force EDF for this example
        verify=verify_export,
        verify_tolerance=verification_tolerance
    )

    logging.info(f"Successfully exported to {output_path_edf}")
    if verify_export and verification_results_edf:
        # Basic print of results, you could process this dict further
        # print(verification_results_edf)
        pass # Results are already printed by the logger in to_edf

except Exception as e:
    logging.error(f"Error exporting to EDF: {e}")

# --- Export to BDF with Verification ---
# BDF is often better for high-precision data like Trigno
logging.info(f"\\n--- Exporting to BDF ({output_file_bdf}) ---")
try:
    verification_results_bdf = emg_only.to_edf(
        output_path_bdf,
        format='bdf',  # Force BDF for this example
        verify=verify_export,
        verify_tolerance=verification_tolerance
    )
    logging.info(f"Successfully exported to {output_path_bdf}")
    if verify_export and verification_results_bdf:
        # print(verification_results_bdf)
        pass # Results are already printed by the logger in to_edf

except Exception as e:
    logging.error(f"Error exporting to BDF: {e}")

# --- Cleanup (Optional) ---
# Uncomment to automatically delete generated files
# if os.path.exists(output_path_edf):
#     os.remove(output_path_edf)
#     logging.info(f"Removed {output_path_edf}")
# if os.path.exists(output_path_bdf):
#     os.remove(output_path_bdf)
#     logging.info(f"Removed {output_path_bdf}")
# # Also remove the .tsv files if they exist
# if os.path.exists(output_path_edf.replace('.edf', '.tsv')):
#     os.remove(output_path_edf.replace('.edf', '.tsv'))
# if os.path.exists(output_path_bdf.replace('.bdf', '.tsv')):
#     os.remove(output_path_bdf.replace('.bdf', '.tsv'))


logging.info("\\nExample finished.") 