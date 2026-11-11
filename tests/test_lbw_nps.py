"""Unit tests for the LBW-NPS loss, the shared band spectra and the local NPS metric."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from isbi_data import assert_no_patient_leakage
from lbw_nps import LocalBandNPSLoss
from metrics import local_nps_gaps
from spectra import BAND_EDGES, extract_patches, patch_band_energies


def test_zero_at_equality_and_gradient() -> None:
    """The loss is zero for identical images and has a finite, non-zero gradient otherwise."""
    target = torch.rand(2, 1, 128, 128)
    valid = torch.ones_like(target)
    loss_fn = LocalBandNPSLoss(patch_size=32, stride=16)
    assert loss_fn(target.clone(), target, valid).item() < 1e-6
    noisy = (target + 0.05 * torch.randn_like(target)).clamp(0, 1).requires_grad_(True)
    value = loss_fn(noisy, target, valid)
    value.backward()
    assert torch.isfinite(value) and noisy.grad is not None and noisy.grad.abs().sum() > 0


def test_invalid_patches_are_ignored() -> None:
    """Differences inside patches outside the valid mask do not contribute."""
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
    """Band weights are normalized to unit mean."""
    loss_fn = LocalBandNPSLoss(band_weights=(0.5, 1.0, 2.0))
    assert abs(float(loss_fn.band_weights.mean()) - 1.0) < 1e-6


def test_patient_leakage_is_detected(tmp_path: Path) -> None:
    """Disjoint splits pass and a patient shared by two splits is rejected."""
    splits = {
        "train": ["000001_01_01_010.png.npz", "000002_01_01_020.png.npz"],
        "val": ["000003_01_01_030.png.npz"],
        "test": ["000004_01_01_040.png.npz"],
    }
    for split, files in splits.items():
        (tmp_path / f"{split}_files.json").write_text(json.dumps(files))
    assert assert_no_patient_leakage(tmp_path) == {"train": 2, "val": 1, "test": 1}
    (tmp_path / "test_files.json").write_text(json.dumps(["000001_01_01_050.png.npz"]))
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(tmp_path)
