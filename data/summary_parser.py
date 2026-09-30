from pathlib import Path
from typing import List, Dict, Tuple


def parse_summary(filepath: Path) -> Dict:
    """Parse a chbXX-summary.txt file.

    Returns dict with keys:
      - sampling_rate: int
      - channels: list of str
      - files: list of dict with name, start_time, end_time, seizures (list of (start_sec, end_sec))
    """
    text = filepath.read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    result = {"sampling_rate": None, "channels": [], "files": []}
    in_channels = False
    current_file = None

    for line in lines:
        if line.startswith("Data Sampling Rate:"):
            result["sampling_rate"] = int(line.split(":")[1].strip().split()[0])
        elif line.startswith("Channels in EDF Files:"):
            in_channels = True
            continue
        elif line.startswith("File Name:"):
            in_channels = False
            if current_file:
                result["files"].append(current_file)
            current_file = {"name": line.split(":")[1].strip(), "seizures": []}
        elif in_channels and line.startswith("Channel"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                result["channels"].append(parts[1].strip())
        elif current_file is not None:
            if line.startswith("File Start Time:"):
                current_file["start_time"] = line.split(":", 1)[1].strip()
            elif line.startswith("File End Time:"):
                current_file["end_time"] = line.split(":", 1)[1].strip()
            elif "Start Time:" in line and ("Seizure" in line):
                val = int(line.split(":")[-1].strip().split()[0])
                current_file["seizures"].append([val])
            elif "End Time:" in line and ("Seizure" in line):
                val = int(line.split(":")[-1].strip().split()[0])
                if current_file["seizures"] and len(current_file["seizures"][-1]) == 1:
                    current_file["seizures"][-1].append(val)
                else:
                    current_file["seizures"].append([None, val])
            elif line.startswith("Number of Seizures in File:"):
                pass

    if current_file:
        result["files"].append(current_file)

    for f in result["files"]:
        f["seizures"] = [tuple(s) if len(s) == 2 else (s[0], None) for s in f["seizures"]]

    return result


def parse_all_summaries(dataset_dir: Path) -> Dict[str, Dict]:
    """Parse all chbXX-summary.txt files in dataset directory."""
    subjects = {}
    for subj_dir in sorted(dataset_dir.iterdir()):
        if subj_dir.is_dir() and subj_dir.name.startswith("chb"):
            summary_file = subj_dir / f"{subj_dir.name}-summary.txt"
            if summary_file.exists():
                subjects[subj_dir.name] = parse_summary(summary_file)
    return subjects
