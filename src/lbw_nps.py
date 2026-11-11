"""Local band-weighted image power matching for CT denoising.

The loss averages squared log-power discrepancies over valid local windows.
The training adapter uses the reference body mask and intensity criteria.
"""

from __future__ import annotations

import torch
from torch import nn

from spectra import (
    BAND_EDGES,
    EPS,
    extract_patches,
    patch_band_energies,
    patch_fraction,
)


class LocalBandNPSLoss(nn.Module):
    """LBW-NPS: band-weighted squared log-power discrepancy over valid local windows.

    Args:
        patch_size: Side length P of the square analysis windows.
        stride: Window stride (P/2 in all reported experiments).
        band_weights: Weight per frequency band (low, mid, high).
        band_edges: Band edges as fractions of Nyquist.
        min_valid: Minimum fraction of valid pixels for a window to contribute.
        normalise_weights: Rescale the band weights to unit mean.
    """

    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 32,
        band_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
        band_edges: tuple[float, ...] = BAND_EDGES,
        min_valid: float = 0.9,
        normalise_weights: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.band_edges = band_edges
        self.min_valid = min_valid
        w = torch.tensor(band_weights, dtype=torch.float32)
        # Unit mean, so the total NPS mass is the same whatever the band profile
        # is and one lambda stays comparable across arms. The pooled baselines
        # normalise the same way; without it a band sweep is also a lambda sweep.
        if normalise_weights:
            w = w / w.mean()
        self.register_buffer("band_weights", w)

    def _energies(self, image: torch.Tensor) -> torch.Tensor:
        return patch_band_energies(
            extract_patches(image, self.patch_size, self.stride), self.band_edges
        )

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Return the mean loss over valid windows.

        Args:
            prediction: Denoised image, `[B, 1, H, W]`.
            target: Reference image, `[B, 1, H, W]`.
            valid: Binary validity map, `[B, 1, H, W]`.
        """
        predicted = self._energies(prediction.float())
        reference = self._energies(target.float()).detach()
        keep = (patch_fraction(valid, self.patch_size, self.stride)[:, 0] >= self.min_valid).float()
        per_patch = (
            self.band_weights * (torch.log(predicted + EPS) - torch.log(reference + EPS)).pow(2)
        ).sum(dim=-1)
        return (per_patch * keep).sum() / keep.sum().clamp(min=1.0)


class LBWNPSTrainingLoss(LocalBandNPSLoss):
    """Adapter so `train.py` can call this with its common signature.

    The trainer hands every auxiliary loss `(pred, gt, masks, noisy=...)`. The
    valid map is derived here rather than in the trainer, so the training loss
    and the evaluation metric agree on what a usable window is:

    * body: any of the tissue masks, which is how every other arm defines it;
    * unclipped: `0 < x < 1` on the target. In the `full` domain nothing is
      window-clipped so this term admits everything, but it is kept because the
      rule has to hold for the `clipped` domain too.
    """

    CLIP_MARGIN = 1e-3

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor | None = None,
        noisy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the reference validity map from the tissue masks and return the loss.

        `noisy` is accepted for signature compatibility with the other losses and is unused.
        """
        assert masks is not None, "LBW-NPS needs the tissue masks to build the body map"
        body = masks.amax(dim=1, keepdim=True).float()
        unclipped = ((target > self.CLIP_MARGIN) & (target < 1.0 - self.CLIP_MARGIN)).float()
        return super().forward(prediction, target, body * unclipped)
