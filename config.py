import os
from pathlib import Path

DATASET_DIR = Path(r"D:\chb-mit-scalp-eeg-database-1.0.0\chb-mit-scalp-eeg-database-1.0.0")
OUTPUT_DIR = Path(r"D:\chb-mit-scalp-eeg-database-1.0.0\pomas\output")

SAMPLE_RATE = 256
WINDOW_SECONDS = 10
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS
STRIDE_SECONDS = 5
STRIDE = SAMPLE_RATE * STRIDE_SECONDS

# The common 19-channel bipolar montage used by the initial PoMAS experiments.
# Recordings that do not provide every channel are excluded rather than silently
# substituted with a different montage.
SELECTED_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7",
]
NUM_CHANNELS = 19

PREICTAL_MINUTES = 30
PREICTAL_SECONDS = PREICTAL_MINUTES * 60
# Windows in this recovery period are excluded from training/evaluation.  This
# avoids treating post-seizure EEG as ordinary interictal background activity.
POSTICTAL_MINUTES = 30
POSTICTAL_SECONDS = POSTICTAL_MINUTES * 60

BAND_FILTER_LOW = 0.5
BAND_FILTER_HIGH = 50.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
