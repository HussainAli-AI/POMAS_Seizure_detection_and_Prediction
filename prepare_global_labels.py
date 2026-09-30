"""Build leakage-safe PoMAS archives with seizure labels spanning EDF boundaries.

Outputs are written to a new directory. Existing preprocessing archives are
never replaced unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import DATASET_DIR, OUTPUT_DIR, SAMPLE_RATE, SELECTED_CHANNELS, STRIDE, WINDOW_SIZE
from data.dataset import label_window
from data.hdf5_dataset import HDF5WindowWriter
from data.summary_parser import parse_all_summaries
from preprocessing.preprocess import extract_windows, preprocess_file


def clock_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def global_file_offsets(files: list[dict]) -> dict[str, float]:
    """Unwrap summary clock times across midnight into one subject timeline."""
    offsets: dict[str, float] = {}
    previous = None
    day_offset = 0
    for file_info in files:
        raw = clock_seconds(file_info["start_time"])
        start = raw + day_offset
        while previous is not None and start < previous:
            day_offset += 24 * 3600
            start = raw + day_offset
        offsets[file_info["name"]] = float(start)
        previous = start
    return offsets


def prepare_subject(subject: str, overwrite: bool = False) -> Path:
    summaries = parse_all_summaries(DATASET_DIR)
    if subject not in summaries:
        raise ValueError(f"Unknown subject: {subject}")
    info = summaries[subject]
    offsets = global_file_offsets(info["files"])
    base_time = min(offsets.values())
    seizures = [
        (offsets[file_info["name"]] + start, offsets[file_info["name"]] + end)
        for file_info in info["files"]
        for start, end in file_info.get("seizures", [])
        if start is not None and end is not None
    ]

    output_dir = OUTPUT_DIR / "preprocessed_global_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject}_preprocessed_global_v2.h5"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite to replace only this v2 file")

    written = 0
    with HDF5WindowWriter(output_path, len(SELECTED_CHANNELS), WINDOW_SIZE) as writer:
        writer.set_metadata(subject_id=subject, channel_names=SELECTED_CHANNELS, sample_rate=SAMPLE_RATE)
        writer.file.attrs["label_timeline"] = "subject_global_summary_clock_v2"
        writer.file.attrs["preictal_minutes"] = 30
        writer.file.attrs["postictal_exclusion_minutes"] = 30
        for file_info in tqdm(info["files"], desc=subject):
            edf_path = DATASET_DIR / subject / file_info["name"]
            if not edf_path.exists():
                continue
            try:
                data = preprocess_file(edf_path, pick_channels=SELECTED_CHANNELS)
            except Exception as exc:
                print(f"Skipping {file_info['name']}: {exc}")
                continue
            windows = extract_windows(data, WINDOW_SIZE, STRIDE)
            local_starts = np.arange(len(windows)) * STRIDE / SAMPLE_RATE
            absolute_starts = offsets[file_info["name"]] + local_starts
            labels = np.asarray([
                label_window(seizures, start, start + WINDOW_SIZE / SAMPLE_RATE)
                for start in absolute_starts
            ])
            keep = labels >= 0
            if keep.any():
                replay_times = absolute_starts[keep] - base_time
                writer.append(windows[keep], labels[keep], file_info["name"], replay_times)
                written += int(keep.sum())
    print(f"Saved {written} globally labeled windows to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for subject_name in args.subjects:
        prepare_subject(subject_name, args.overwrite)
