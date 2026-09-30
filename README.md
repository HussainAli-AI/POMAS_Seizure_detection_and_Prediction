# PoMAS research implementation

PoMAS is an offline EEG replay prototype for epileptic seizure early warning.
It is a research tool, not a clinical device and must not be used to make
medical decisions.

## First milestone

The first milestone is a reproducible CHB-MIT preprocessing dataset.  Every
accepted recording must contain the fixed 19-channel bipolar montage in
`config.py`.  Recordings with missing channels are excluded and reported by
the preprocessing run; channels are never replaced with unrelated signals.

Labels are `0=interictal`, `1=preictal`, and `2=ictal`.  The 30 minutes after
a seizure are excluded, so postictal EEG is not used as interictal background.

## Environment

Use a supported Python version for PyTorch (recommend Python 3.11) and install
the dependencies:

```powershell
cd D:\chb-mit-scalp-eeg-database-1.0.0\pomas
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## First run

From the `pomas` directory, process one subject before attempting the full
dataset:

```powershell
python main.py --subjects chb01
```

The resulting file is written under `output\preprocessed`.  Inspect the
window count and class distribution before training any model.

## Evaluation rule

Perform the subject/time split before augmentation, feature scaling, or model
selection.  Train augmentation only on the training partition.  Keep recording
ID and window timestamp with every prediction so false alarms per hour and
warning latency can be calculated during offline replay.
