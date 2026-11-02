#!/usr/bin/env python3
"""Step 1.4 sanity checks, plus the Step 1.1 unit tests for the shared spectra."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbw_nps import LocalBandNPSLoss  # noqa: E402
from metrics import local_nps_gaps  # noqa: E402
from spectra import BAND_EDGES, extract_patches, patch_band_energies  # noqa: E402


def test_zero_at_equality_and_gradient() -> None:
    target = torch.rand(2, 1, 128, 128)
    valid = torch.ones_like(target)
    loss_fn = LocalBandNPSLoss(patch_size=32, stride=16)
    assert loss_fn(target.clone(), target, valid).item() < 1e-6
    noisy = (target + 0.05 * torch.randn_like(target)).clamp(0, 1).requires_grad_(True)
    value = loss_fn(noisy, target, valid)
    value.backward()
    assert torch.isfinite(value) and noisy.grad is not None and noisy.grad.abs().sum() > 0


def test_invalid_patches_are_ignored() -> None:
    target = torch.rand(1, 1, 128, 128)
    prediction = target.clone()
    prediction[..., :64, :] += 0.2 * torch.randn(1, 1, 64, 128)
    valid = torch.zeros_like(target)
    valid[..., 64:, :] = 1.0
    assert LocalBandNPSLoss(patch_size=32, stride=32)(prediction, target, valid).item() < 1e-6


def test_identity_metrics() -> None:
    """Identity input: G_loc is 0 (PSNR/SSIM are skimage's and trivially perfect)."""
    target = torch.rand(1, 1, 128, 128)
    assert float(local_nps_gaps(target.clone(), target, 64).mean()) < 1e-6


def test_white_noise_bands_are_equal() -> None:
    """White noise has a flat spectrum, so the three band means must agree."""
    torch.manual_seed(0)
    patches = extract_patches(torch.randn(64, 1, 64, 64), 64, 64)
    e = patch_band_energies(patches, BAND_EDGES).mean(dim=(0, 1))
    r = (e / e.mean()).tolist()
    assert max(abs(v - 1.0) for v in r) < 0.1, r


def test_global_case_is_one_patch() -> None:
    """P = image size, stride = P: the loss reduces to a single global spectrum."""
    torch.manual_seed(0)
    gt = torch.rand(1, 1, 64, 64)
    pred = gt + 0.03 * torch.randn_like(gt)
    valid = torch.ones_like(gt)
    local = LocalBandNPSLoss(patch_size=64, stride=64, min_valid=0.0)
    ep = patch_band_energies(extract_patches(pred, 64, 64), BAND_EDGES)
    et = patch_band_energies(extract_patches(gt, 64, 64), BAND_EDGES)
    manual = ((torch.log(ep + 1e-8) - torch.log(et + 1e-8)).pow(2)).sum()
    assert abs(local(pred, gt, valid).item() - manual.item()) < 1e-5


def test_band_weights_are_unit_mean() -> None:
    loss_fn = LocalBandNPSLoss(band_weights=(0.5, 1.0, 2.0))
    assert abs(float(loss_fn.band_weights.mean()) - 1.0) < 1e-6


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"  FAIL  {name}: {exc}")
    print("tum testler gecti" if not fails else f"{fails} test basarisiz")
    sys.exit(1 if fails else 0)
