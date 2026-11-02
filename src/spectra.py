#!/usr/bin/env python3
"""Patch spectra: the one definition shared by the LBW-NPS loss and G_loc.

The plan (Step 1.1) requires the loss and the metric to use the same band
definition, so both import this module and nothing re-implements it.

`BAND_EDGES` is expressed as a fraction of Nyquist. The plan leaves it as a
TODO to be matched to "#230 variant B"; variant B's `cutoffs` are (0.15, 0.35)
in cycles/pixel and Nyquist is 0.5 cycles/pixel, so the fractions are
(0.30, 0.70). That is where the numbers below come from - they are not a new
choice.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BAND_EDGES: tuple[float, ...] = (0.0, 0.30, 0.70, 1.0)  # = #230 variant B (0.15, 0.35) cyc/px
BAND_NAMES: tuple[str, str, str] = ("low", "mid", "high")
EPS = 1e-8


def hann_2d(size: int, device: torch.device) -> torch.Tensor:
    window = torch.hann_window(size, periodic=False, device=device)
    return torch.outer(window, window)


def band_index_map(size: int, edges: tuple[float, ...], device: torch.device) -> torch.Tensor:
    """Band id per FFT bin, -1 for DC and for anything past the last edge.

    The corners of the square FFT grid reach sqrt(2) x Nyquist. They are
    dropped rather than folded into the top band, so the high band means the
    same thing it means in a radial NPS curve.
    """
    frequencies = torch.fft.fftfreq(size, device=device)
    fy, fx = torch.meshgrid(frequencies, frequencies, indexing="ij")
    radius = torch.sqrt(fx ** 2 + fy ** 2) / 0.5
    index = torch.bucketize(radius, torch.tensor(edges[1:-1], device=device), right=True)
    index[(radius == 0) | (radius > edges[-1])] = -1
    return index


def extract_patches(image: torch.Tensor, size: int, stride: int) -> torch.Tensor:
    """`[B,1,H,W]` -> `[B,L,size,size]`, L sliding windows in raster order."""
    patches = F.unfold(image, kernel_size=size, stride=stride)
    return patches.transpose(1, 2).reshape(image.shape[0], -1, size, size)


def patch_band_energies(patches: torch.Tensor, edges: tuple[float, ...] = BAND_EDGES) -> torch.Tensor:
    """`[B,L,P,P]` -> `[B,L,n_bands]` mean power per band.

    Mean removal then a Hann taper, normalised by sum(h^2) so the result is a
    power spectral density rather than a window-dependent number.
    """
    size = patches.shape[-1]
    window = hann_2d(size, patches.device)
    centered = (patches - patches.mean(dim=(-2, -1), keepdim=True)) * window
    power = torch.fft.fft2(centered).abs().pow(2) / window.pow(2).sum()
    index = band_index_map(size, edges, patches.device)
    return torch.stack([power[..., index == band].mean(dim=-1)
                        for band in range(len(edges) - 1)], dim=-1)


def patch_fraction(mask: torch.Tensor, size: int, stride: int) -> torch.Tensor:
    """`[B,C,H,W]` -> `[B,C,L]`, the fraction of each window covered per channel."""
    channels = mask.shape[1]
    columns = F.unfold(mask, kernel_size=size, stride=stride)
    return columns.reshape(mask.shape[0], channels, size * size, -1).mean(dim=2)
