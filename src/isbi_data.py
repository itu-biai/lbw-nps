"""Split checks and a slice reader over the packed arrays.

The data are packed into one `.npy` per split and key (the same arrays the
training loader in `data.py` reads), so evaluation sees the same slices in the
same order and split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SPLITS = ("train", "val", "test")


def patient_id(stem: str) -> str:
    """`000003_01_01_016.png.npz` -> `000003`."""
    return stem.split("_")[0]


def split_files(packed_root: Path, split: str) -> list[str]:
    """Return the ordered file stems of one split."""
    return json.loads((packed_root / f"{split}_files.json").read_text())


def assert_no_patient_leakage(packed_root: Path) -> dict[str, int]:
    """Check that no patient appears in two splits; return patients per split."""
    patients = {s: {patient_id(f) for f in split_files(packed_root, s)} for s in SPLITS}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = patients[first] & patients[second]
        assert not shared, f"patients in both {first} and {second}: {sorted(shared)[:5]}"
    return {s: len(ids) for s, ids in patients.items()}


class PackedSplit:
    """Read-only view of one split: image pair, tissue masks, file stems."""

    def __init__(
        self, packed_root: Path, split: str, input_key: str, target_key: str, mask_tag: str = "ts"
    ) -> None:
        self.root, self.split = Path(packed_root), split
        self.noisy = np.load(self.root / f"{split}_{input_key}.npy", mmap_mode="r")
        self.gt = np.load(self.root / f"{split}_{target_key}.npy", mmap_mode="r")
        self.masks = np.load(self.root / f"{split}_masks_{mask_tag}.npy", mmap_mode="r")
        self.files = split_files(self.root, split)
        assert len(self.files) == self.gt.shape[0] == self.noisy.shape[0] == self.masks.shape[0]

    def __len__(self) -> int:
        return len(self.files)

    def tensors(
        self, i: int, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(noisy, gt, masks)` for slice `i` as `[1,1,H,W]`, `[1,1,H,W]`, `[1,R,H,W]`."""
        noisy = torch.from_numpy(np.asarray(self.noisy[i], dtype=np.float32))[None, None].to(device)
        gt = torch.from_numpy(np.asarray(self.gt[i], dtype=np.float32))[None, None].to(device)
        masks = torch.from_numpy(np.asarray(self.masks[i], dtype=np.float32))[None].to(device)
        return noisy, gt, masks

    def ids(self, i: int) -> tuple[str, str]:
        """Return `(slice_id, patient_id)` for slice `i`."""
        stem = self.files[i].replace(".png.npz", "")
        return stem, patient_id(stem)
