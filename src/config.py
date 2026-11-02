from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "data" / "packed"
OUT = ROOT / "outputs"
RUNS = OUT / "runs"

SEEDS: tuple[int, ...] = (42, 123, 456)


@dataclass(frozen=True)
class Protocol:
    domain: str = "full"            # (noisy_full_100k, gt_full)
    noise: str = "100k"
    mask_tag: str = "ts"            # TotalSegmentator, evaluation + valid-patch rule only
    epochs: int = 40
    iters_per_epoch: int = 150
    batch_size: int = 16
    train_patch: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-4
    band_weights: tuple[float, float, float] = (0.5, 1.0, 2.0)   # low, mid, high
    cutoffs: tuple[float, float] = (0.15, 0.35)                  # cycles/pixel
    lbw_patch: int = 64
    lbw_stride: int = 32
    tissue_patch: int = 32          # per-tissue G_loc window (see amendment 5)
    min_valid: float = 0.9
    min_tissue: float = 0.7


PROTOCOL = Protocol()

