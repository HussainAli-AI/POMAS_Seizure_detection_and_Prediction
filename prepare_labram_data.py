"""Create a separate LaBraM-compatible CHB-MIT archive without changing PoMAS data."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import DATASET_DIR, OUTPUT_DIR, SELECTED_CHANNELS
from data.hdf5_dataset import HDF5WindowWriter
from data.summary_parser import parse_all_summaries
from data.dataset import label_window
from preprocessing.preprocess import extract_windows, preprocess_file

LABRAM_FS = 200
WINDOW_SECONDS = 10
WINDOW_SAMPLES = LABRAM_FS * WINDOW_SECONDS
STRIDE_SAMPLES = LABRAM_FS * 5


def prepare_subject(subject: str, overwrite: bool) -> None:
    output_dir = OUTPUT_DIR / "labram_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject}_labram_200hz.h5"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}. Use --overwrite only to replace this derived LaBraM file.")

    summaries = parse_all_summaries(DATASET_DIR)
    if subject not in summaries:
        raise ValueError(f"Unknown subject: {subject}")
    subject_seconds = 0.0
    written = 0
    subject_dir = DATASET_DIR / subject
    with HDF5WindowWriter(output_path, len(SELECTED_CHANNELS), WINDOW_SAMPLES) as writer:
        writer.set_metadata(subject_id=subject, channel_names=SELECTED_CHANNELS, sample_rate=LABRAM_FS)
        writer.file.attrs["unit"] = "uV"
        writer.file.attrs["bandpass_hz"] = "0.1-75"
        writer.file.attrs["notch_hz"] = 50.0
        writer.file.attrs["normalization"] = "none"
        for file_info in tqdm(summaries[subject]["files"], desc=subject):
            edf_path = subject_dir / file_info["name"]
            if not edf_path.exists():
                continue
            try:
                # MNE returns volts; LaBraM documents microvolt input and no
                # z-score normalization in its preprocessing recipe.
                data = preprocess_file(
                    edf_path, pick_channels=SELECTED_CHANNELS, target_fs=LABRAM_FS,
                    band_low=0.1, band_high=75.0, notch_freq=50.0, apply_zscore=False,
                ) * 1e6
            except Exception as exc:
                print(f"Skipping {file_info['name']}: {exc}")
                continue
            windows = extract_windows(data, WINDOW_SAMPLES, STRIDE_SAMPLES)
            file_starts = np.arange(len(windows)) * STRIDE_SAMPLES / LABRAM_FS
            labels = np.asarray([
                label_window(file_info.get("seizures", []), start, start + WINDOW_SECONDS)
                for start in file_starts
            ])
            keep = labels >= 0
            if keep.any():
                writer.append(windows[keep], labels[keep], file_info["name"], subject_seconds + file_starts[keep])
                written += int(keep.sum())
            subject_seconds += data.shape[1] / LABRAM_FS
    print(f"Saved {written} LaBraM-compatible windows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace only an existing derived LaBraM archive.")
    args = parser.parse_args()
    for subject_name in args.subjects:
        prepare_subject(subject_name, args.overwrite)
