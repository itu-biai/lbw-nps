#!/usr/bin/env python3
"""The metrics every ISBI row is scored with - one implementation, all rows.

`G_loc` is deliberately **not** the training loss: non-overlapping windows and
uniform band weights, while the loss uses overlapping windows and the variant B
band profile. Training on exactly what you report is how a frequency loss
flatters itself, and the plan asks for this gap on purpose.

`R_H` comes from `work/tcbwnps/code/nps.py`, the implementation the lab's other
NPS numbers already use (see `config.PROTOCOL_AMENDMENTS`).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


from spectra import BAND_EDGES, EPS, extract_patches, patch_band_energies, patch_fraction  # noqa: E402
from nps import (  # noqa: E402
    DEFAULT_BAND_CUTOFFS,
    RoiSpec,
    band_ratios,
    find_class_rois,
    nps_freq_axis,
    nps_from_rois,
)

CLASS_NAMES: tuple[str, ...] = ("lung", "soft", "bone")
CLIP_MARGIN = 1e-3
GLOBAL_ROI = RoiSpec(size=64, stride=32, min_coverage=0.95, max_rois=16)


def local_nps_gaps(prediction: torch.Tensor, target: torch.Tensor, size: int = 64) -> torch.Tensor:
    """Per-window |mean log-energy difference| on NON-overlapping windows."""
    predicted = patch_band_energies(extract_patches(prediction, size, size), BAND_EDGES)
    reference = patch_band_energies(extract_patches(target, size, size), BAND_EDGES)
    return (torch.log(predicted + EPS) - torch.log(reference + EPS)).abs().mean(dim=-1)[0]


def valid_map(target: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
    unclipped = ((target > CLIP_MARGIN) & (target < 1.0 - CLIP_MARGIN)).float()
    return body.float() * unclipped


def _r_high(pred: np.ndarray, gt: np.ndarray, body: np.ndarray) -> float:
    """Global high-band NPS ratio over body ROIs (1.0 = matched texture)."""
    rois = find_class_rois(body.astype(np.uint8), GLOBAL_ROI, image=gt)
    if rois.shape[0] == 0:
        return float("nan")
    np_pred = nps_from_rois(pred, rois, GLOBAL_ROI.size)
    np_gt = nps_from_rois(gt, rois, GLOBAL_ROI.size)
    if np_pred is None or np_gt is None:
        return float("nan")
    return float(band_ratios(nps_freq_axis(GLOBAL_ROI.size), np_pred, np_gt,
                             DEFAULT_BAND_CUTOFFS)[2])


WINDOWS: tuple[int, ...] = (64, 32)


def slice_metrics(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                  tissue: torch.Tensor, lpips_fn=None, windows: tuple[int, ...] = WINDOWS,
                  min_valid: float = 0.9, min_tissue: float = 0.7) -> dict:
    """One row of the per-slice CSV. Tensors are `[1,1,H,W]` / `[1,3,H,W]`.

    Every spectral quantity is produced at **both** window sizes, body and per
    tissue, because the two answer different questions and neither is a safe
    default on its own:

    * 64 x 64 is the plan's window and gives the better-conditioned spectrum
      (more frequency bins per band), but at that size the test split holds 4
      windows that are 70% bone, so the bone column is unusable;
    * 32 x 32 reaches every tissue (129 bone windows, 1261 lung) at the cost of
      a coarser spectral estimate.

    Columns are named `gloc_w64`, `gloc_lung_w32` and so on. The unsuffixed
    names kept at the end are aliases for the earlier default (body at 64,
    tissue at 32) so nothing downstream breaks.
    """
    pred_np = prediction[0, 0].detach().cpu().numpy().astype(np.float64)
    gt_np = target[0, 0].detach().cpu().numpy().astype(np.float64)
    body = (tissue[0].detach().cpu().numpy().max(axis=0) > 0).astype(np.uint8)

    # data_range = 1.0, as the plan specifies and as `evaluate.psnr` already
    # uses, so these numbers sit on the same scale as every #230 number.
    row = {
        "psnr": float(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)),
        "ssim": float(structural_similarity(gt_np, pred_np, data_range=1.0)),
        "lpips": float(lpips_fn(prediction, target)) if lpips_fn else math.nan,
        "r_h": _r_high(pred_np, gt_np, body),
    }
    row["r_h_error"] = abs(1.0 - row["r_h"]) if np.isfinite(row["r_h"]) else math.nan

    for w in windows:
        gaps = local_nps_gaps(prediction, target, w)
        keep = patch_fraction(valid, w, w)[0, 0] >= min_valid
        frac = patch_fraction(tissue.float(), w, w)[0]
        row[f"gloc_w{w}"] = float(gaps[keep].mean()) if bool(keep.any()) else math.nan
        row[f"patches_w{w}"] = int(keep.sum())
        for name, f in zip(CLASS_NAMES, frac):
            sel = keep & (f >= min_tissue)
            row[f"gloc_{name}_w{w}"] = float(gaps[sel].mean()) if bool(sel.any()) else math.nan
            row[f"patches_{name}_w{w}"] = int(sel.sum())

    # backward-compatible aliases: body at 64, tissue at 32
    row["local_nps_gap"] = row.get("gloc_w64", math.nan)
    row["patches_valid"] = row.get("patches_w64", 0)
    for name in CLASS_NAMES:
        row[f"local_nps_gap_{name}"] = row.get(f"gloc_{name}_w32", math.nan)
        row[f"patches_{name}"] = row.get(f"patches_{name}_w32", 0)
    return row


class LpipsAlex:
    """AlexNet LPIPS, matched to the #230 evaluator (clamp to [0,1], no rescale)."""

    def __init__(self, device: str = "cuda") -> None:
        import lpips as lpips_lib

        self.net = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        # Clamp, do not rescale: identical to `evaluate.Evaluator._lpips`, so the
        # LPIPS column is comparable with everything the lab has already reported.
        a = prediction.clamp(0, 1).to(self.device).repeat(1, 3, 1, 1) * 2 - 1
        b = target.clamp(0, 1).to(self.device).repeat(1, 3, 1, 1) * 2 - 1
        return float(self.net(a, b).item())
