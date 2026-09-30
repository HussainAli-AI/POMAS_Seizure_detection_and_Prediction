"""Official-checkpoint loader for the LaBraM fine-tuning backbone."""
from pathlib import Path
import sys
import torch

# Index 18 is P7-T7, the reversed duplicate of T7-P7 at index 2.  LaBraM's
# checkpoint has 18 learned electrode positions, so this view preserves the
# unique bipolar pairs without rewriting any HDF5 archive.
LABRAM_CHANNEL_INDICES = tuple(range(18))
# The checkpoint stores one leading CLS position plus 18 electrode positions.
LABRAM_POSITION_INDICES = tuple(range(19))


def build_labram_base(checkpoint_path: Path, num_classes: int = 3):
    source_root = Path(__file__).resolve().parents[2] / "external" / "LaBraM"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import modeling_finetune
    model = modeling_finetune.labram_base_patch200_200(
        num_classes=num_classes, use_mean_pooling=True, use_rel_pos_bias=False,
        use_abs_pos_emb=True, init_values=0.1, qkv_bias=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model"]
    # Official pretraining checkpoints store backbone weights under `student.`.
    converted = {key.removeprefix("student."): value for key, value in source.items() if key.startswith("student.")}
    target = model.state_dict()
    compatible = {key: value for key, value in converted.items() if key in target and value.shape == target[key].shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    loaded_backbone = len(compatible)
    if loaded_backbone < 150:
        raise RuntimeError(f"Only {loaded_backbone} tensors matched; refusing incomplete LaBraM initialization")
    return model, {"loaded_tensors": loaded_backbone, "missing": missing, "unexpected": unexpected}


def labram_input_from_pomas(windows: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Convert (batch, 19, 2000) PoMAS windows to LaBraM patch input."""
    x = windows[:, LABRAM_CHANNEL_INDICES, :]
    if x.shape[-1] != 2000:
        raise ValueError(f"LaBraM expects 2000 samples, got {x.shape[-1]}")
    return x.reshape(x.shape[0], 18, 10, 200), list(LABRAM_POSITION_INDICES)
