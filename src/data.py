"""Datasets over the packed memmaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedSlices:
    """Lazy memmap holder shared by the patch and the full slice datasets."""

    def __init__(
        self, root: Path, split: str, input_key: str, target_key: str, mask_tag: str
    ) -> None:
        self.root, self.split = Path(root), split
        self.files = json.loads((self.root / f"{split}_files.json").read_text())
        self.noisy = np.load(self.root / f"{split}_{input_key}.npy", mmap_mode="r")
        self.gt = np.load(self.root / f"{split}_{target_key}.npy", mmap_mode="r")
        self.masks = np.load(self.root / f"{split}_masks_{mask_tag}.npy", mmap_mode="r")
        self.rois = np.load(self.root / f"{split}_rois_{mask_tag}.npy", mmap_mode="r")
        self.nrois = np.load(self.root / f"{split}_nrois_{mask_tag}.npy", mmap_mode="r")
        self.n_classes = int(self.masks.shape[1])
        # Soft mask tags store a continuous partition as uint8 with 255 = 1.0;
        # binary tags need no scaling. The optional `maskinfo_<tag>.json`
        # sidecar holds that scale and is absent for binary tags.
        info = self.root / f"maskinfo_{mask_tag}.json"
        self.mask_scale = json.loads(info.read_text())["scale"] if info.exists() else 1.0

    def __len__(self) -> int:
        return len(self.files)


class PatchDataset(Dataset):
    """Random crops, half of them anchored on a tissue ROI.

    Lung covers 4.5% of a frame and bone 2.1%, so uniform random crops would put
    a bone dominated region in a batch roughly once every fifty patches and the
    tissue terms of the loss would spend most of training on the batch level
    fallback. Anchoring half the crops on a cached tissue ROI fixes the exposure
    without touching the loss. Every variant in the ablation uses this same
    sampler, including the L1 only baseline, so the comparison stays fair.
    """

    def __init__(
        self,
        slices: PackedSlices,
        patch_size: int = 128,
        length: int = 2400,
        anchored_prob: float = 0.5,
        roi_size: int = 32,
        seed: int = 0,
        augment: bool = True,
    ) -> None:
        self.s = slices
        self.patch = patch_size
        self.length = length
        self.anchored_prob = anchored_prob
        self.roi_size = roi_size
        self.seed = seed
        self.augment = augment
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so that crops differ between epochs but stay reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def _crop_origin(self, rng: np.random.Generator, idx: int) -> tuple[int, int]:
        p, h, w = self.patch, 512, 512
        if rng.random() < self.anchored_prob:
            counts = np.asarray(self.s.nrois[idx])
            avail = np.flatnonzero(counts > 0)
            if len(avail):
                c = int(rng.choice(avail))
                j = int(rng.integers(counts[c]))
                ry, rx = (int(v) for v in self.s.rois[idx, c, j])
                y0 = int(rng.integers(max(0, ry + self.roi_size - p), min(ry, h - p) + 1))
                x0 = int(rng.integers(max(0, rx + self.roi_size - p), min(rx, w - p) + 1))
                return y0, x0
        return int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, self.epoch, i))
        idx = int(rng.integers(len(self.s)))
        y0, x0 = self._crop_origin(rng, idx)
        p = self.patch
        sl = (slice(y0, y0 + p), slice(x0, x0 + p))
        noisy = np.array(self.s.noisy[idx][sl], dtype=np.float32)
        gt = np.array(self.s.gt[idx][sl], dtype=np.float32)
        masks = np.array(self.s.masks[idx][(slice(None), *sl)], dtype=np.float32)
        if self.s.mask_scale != 1.0:
            masks /= self.s.mask_scale
        if self.augment:
            k = int(rng.integers(4))
            if k:
                noisy, gt = np.rot90(noisy, k), np.rot90(gt, k)
                masks = np.rot90(masks, k, axes=(1, 2))
            if rng.random() < 0.5:
                noisy, gt, masks = noisy[:, ::-1], gt[:, ::-1], masks[:, :, ::-1]
        return {
            "noisy": torch.from_numpy(np.ascontiguousarray(noisy))[None],
            "gt": torch.from_numpy(np.ascontiguousarray(gt))[None],
            "masks": torch.from_numpy(np.ascontiguousarray(masks)),
            "index": torch.tensor(idx),
        }


class SliceDataset(Dataset):
    """Full 512x512 slices for validation and test."""

    def __init__(self, slices: PackedSlices, limit: int | None = None) -> None:
        self.s = slices
        self.n = len(slices) if limit is None else min(limit, len(slices))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "noisy": torch.from_numpy(np.array(self.s.noisy[i], dtype=np.float32))[None],
            "gt": torch.from_numpy(np.array(self.s.gt[i], dtype=np.float32))[None],
            "masks": torch.from_numpy(np.array(self.s.masks[i], dtype=np.uint8)),
            "rois": torch.from_numpy(np.array(self.s.rois[i], dtype=np.int64)),
            "nrois": torch.from_numpy(np.array(self.s.nrois[i], dtype=np.int64)),
            "index": torch.tensor(i),
        }
