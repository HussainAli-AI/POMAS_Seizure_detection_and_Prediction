"""PoMAS main entry point - run the full preprocessing pipeline."""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from config import DATASET_DIR, OUTPUT_DIR, SELECTED_CHANNELS, SAMPLE_RATE, WINDOW_SIZE, STRIDE, PREICTAL_SECONDS
from data.summary_parser import parse_all_summaries
from preprocessing.preprocess import preprocess_file, extract_windows
from data.dataset import CHBMITDataset, create_dataloader, label_window
from data.hdf5_dataset import HDF5WindowWriter


def build_dataset(
    subjects: Optional[List[str]] = None,
    output_file: Optional[str] = None,
) -> CHBMITDataset:
    """Build and optionally save the preprocessed dataset."""
    ds = CHBMITDataset(
        subjects=subjects,
        preload=True,
    )

    if output_file:
        output_path = OUTPUT_DIR / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            data=np.array([w["data"] for w in ds.windows]),
            labels=np.array([w["label"] for w in ds.windows]),
            patient_ids=np.array([w["patient_id"] for w in ds.windows]),
            timestamps=np.array([w["timestamp"] for w in ds.windows]),
            recording_ids=np.array([w["file"] for w in ds.windows]),
            channel_names=np.array(SELECTED_CHANNELS),
            sample_rate=np.array(SAMPLE_RATE),
        )
        print(f"Saved dataset to {output_path}")

    return ds


def run_preprocessing(
    subjects: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    apply_ica: bool = False,
    apply_emd: bool = False,
):
    """Run the full preprocessing pipeline for specified subjects.

    Saves preprocessed windows as a chunked HDF5 archive per subject.
    """
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = parse_all_summaries(DATASET_DIR)
    if subjects:
        summaries = {k: v for k, v in summaries.items() if k in subjects}

    for subj_name, info in tqdm(summaries.items(), desc="Processing subjects"):
        subj_dir = DATASET_DIR / subj_name
        save_path = output_dir / f"{subj_name}_preprocessed.h5"
        windows_written = 0
        subject_seconds = 0.0

        with HDF5WindowWriter(save_path, len(SELECTED_CHANNELS), WINDOW_SIZE) as writer:
            writer.set_metadata(subject_id=subj_name, channel_names=SELECTED_CHANNELS, sample_rate=SAMPLE_RATE)
            for file_info in tqdm(info["files"], desc=f"  {subj_name}", leave=False):
                edf_path = subj_dir / file_info["name"]
                if not edf_path.exists():
                    continue

                try:
                    data = preprocess_file(
                        edf_path,
                        pick_channels=SELECTED_CHANNELS,
                        apply_ica=apply_ica,
                        apply_emd=apply_emd,
                    )
                except Exception as e:
                    print(f"  Error processing {file_info['name']}: {e}")
                    continue

                windows = extract_windows(data, WINDOW_SIZE, STRIDE)
                n_windows = windows.shape[0]
                seizure_info = file_info.get("seizures", [])

                # Seizure annotations are relative to the current EDF file,
                # whereas stored timestamps use a monotonic subject replay
                # clock across EDF files.
                file_starts = np.arange(n_windows) * STRIDE / SAMPLE_RATE
                timestamps = subject_seconds + file_starts
                labels = np.array([
                    label_window(seizure_info, start, start + WINDOW_SIZE / SAMPLE_RATE)
                    for start in file_starts
                ])
                keep = labels >= 0
                if keep.any():
                    writer.append(windows[keep], labels[keep], file_info["name"], timestamps[keep])
                    windows_written += int(keep.sum())
                subject_seconds += data.shape[1] / SAMPLE_RATE

        if windows_written:
            print(f"  Saved {windows_written} windows to {save_path}")
        else:
            save_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoMAS Preprocessing Pipeline")
    parser.add_argument("--subjects", nargs="+", help="Subjects to process (e.g., chb01 chb02)")
    parser.add_argument("--ica", action="store_true", help="Apply ICA artifact removal")
    parser.add_argument("--emd", action="store_true", help="Apply EMD filtering")
    parser.add_argument("--output", type=str, default=None, help="Output .npz file name")
    parser.add_argument("--build-dataset", action="store_true", help="Build dataset and save")
    parser.add_argument("--info", action="store_true", help="Show dataset info and exit")

    args = parser.parse_args()

    if args.info:
        summaries = parse_all_summaries(DATASET_DIR)
        for subj_name, info in sorted(summaries.items()):
            n_files = len(info["files"])
            n_seizure = sum(1 for f in info["files"] if f["seizures"])
            total_seizures = sum(len(f["seizures"]) for f in info["files"])
            print(f"{subj_name}: {n_files} files, {n_seizure} with seizures, {total_seizures} total seizures")
        print(f"\nTotal subjects: {len(summaries)}")

    elif args.build_dataset:
        ds = build_dataset(subjects=args.subjects, output_file=args.output)
        print(f"Dataset built: {len(ds)} windows")
        labels = {}
        for w in ds.windows:
            l = w["label"]
            labels[l] = labels.get(l, 0) + 1
        print(f"Label distribution: {labels}")

    else:
        run_preprocessing(
            subjects=args.subjects,
            apply_ica=args.ica,
            apply_emd=args.emd,
        )
