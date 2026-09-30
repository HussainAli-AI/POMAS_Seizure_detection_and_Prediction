BIPOLAR_19_MONTAGE = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8",
]

# Standard 19-channel subset used in CHB-MIT literature
STANDARD_19 = BIPOLAR_19_MONTAGE[:19]


def channel_to_bipolar_pair(ch_name: str) -> tuple:
    """Convert a bipolar channel name to its two electrode components."""
    parts = ch_name.split("-")
    if len(parts) == 2:
        return tuple(parts)
    return (ch_name,)
