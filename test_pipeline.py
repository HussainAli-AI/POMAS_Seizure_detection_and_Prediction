"""Quick test of Phase 0 pipeline on one subject."""

from pathlib import Path
from data.summary_parser import parse_summary, parse_all_summaries
from data.chb_mit_parser import load_edf, parse_seizures_file
from data.dataset import CHBMITDataset, create_dataloader
from data.split import loso_split, patient_dependent_split
from preprocessing.preprocess import preprocess_file, extract_windows
from config import DATASET_DIR, SELECTED_CHANNELS


def test_summary_parser():
    print("=== Test Summary Parser ===")
    subj_dir = DATASET_DIR / "chb01"
    summary = parse_summary(subj_dir / "chb01-summary.txt")
    print(f"Sampling rate: {summary['sampling_rate']}")
    print(f"Number of channels: {len(summary['channels'])}")
    print(f"Number of files: {len(summary['files'])}")
    seizure_files = [f["name"] for f in summary["files"] if f["seizures"]]
    print(f"Files with seizures: {seizure_files}")
    for f in summary["files"]:
        if f["seizures"]:
            print(f"  {f['name']}: {f['seizures']}")
    return True


def test_load_edf():
    print("\n=== Test Load EDF ===")
    edf_path = DATASET_DIR / "chb01" / "chb01_01.edf"
    raw = load_edf(edf_path)
    available = [ch for ch in SELECTED_CHANNELS if ch in raw.ch_names]
    raw.pick(available)
    data, times = raw.get_data(return_times=True)
    print(f"Picked {len(available)}/{len(SELECTED_CHANNELS)} channels")
    print(f"Data shape: {data.shape}")
    print(f"Duration: {times[-1]:.1f}s")
    return True


def test_preprocess():
    print("\n=== Test Preprocessing ===")
    edf_path = DATASET_DIR / "chb01" / "chb01_01.edf"
    data = preprocess_file(edf_path, pick_channels=SELECTED_CHANNELS)
    print(f"Preprocessed shape: {data.shape}")
    windows = extract_windows(data)
    print(f"Windows shape: {windows.shape}")
    return True


def test_dataset():
    print("\n=== Test Dataset (chb01 only) ===")
    ds = CHBMITDataset(subjects=["chb01"], preload=True)
    print(f"Total windows: {len(ds)}")
    labels = {}
    for w in ds.windows:
        l = w["label"]
        labels[l] = labels.get(l, 0) + 1
    print(f"Label distribution: {labels}")
    if len(ds) > 0:
        x, y, pid, ts = ds[0]
        print(f"Sample: x shape={x.shape}, y={y}, patient={pid}, timestamp={ts}")
    return True


def test_split():
    print("\n=== Test Splits ===")
    ds = CHBMITDataset(subjects=["chb01", "chb02"], preload=True)
    loso = loso_split(ds)
    for test_subj, split_data in loso.items():
        print(f"LOSO test={test_subj}: train={len(split_data['train'])}, test={len(split_data['test'])}")
    pd_splits = patient_dependent_split(ds)
    for subj, split_data in pd_splits.items():
        print(f"PD test={subj}: train={len(split_data['train'])}, test={len(split_data['test'])}")
    return True


if __name__ == "__main__":
    tests = [
        test_summary_parser,
        test_load_edf,
        test_preprocess,
        test_dataset,
    ]
    all_pass = True
    for t in tests:
        try:
            result = t()
            if result:
                print(f"  PASSED\n")
            else:
                print(f"  FAILED\n")
                all_pass = False
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}\n")
            all_pass = False

    print("=" * 40)
    print(f"All tests passed: {all_pass}")
