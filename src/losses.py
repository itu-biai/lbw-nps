#!/usr/bin/env python3
"""Loss functions for the TC-BW-NPS study.

Four cells of the Week 5 decomposition share one implementation and differ only
in the mask set and the weight matrix:

| cell | tissue conditioning | band weighting |
| ---- | ------------------- | -------------- |
| A    | no (one body mask)  | no (w = 1)     |
| B    | no                  | yes            |
| C    | yes (3 masks)       | no             |
| D    | yes                 | yes            |

`A0` is the literature form of the global NPS loss (uniform over every radial
bin, CT-Mamba / Zhang MLMI) and is kept separate so the paper can report both
the faithful prior-work baseline and a clean 2x2 factorial.

Region localisation has two implementations, compared in the paper:

- `mode="roi"`   ROIs sampled inside each tissue class, Hann windowed. This is
                 how NPS is defined (TG 233) and it keeps the measurement inside
                 a homogeneous region.
- `mode="masked"` the whole image multiplied by the binary mask, as sketched in
                 `paper_plan.md` Week 4. Cheap, but the mask boundary is a step
                 edge whose spectral leakage lands in exactly the high band the
                 loss is supposed to control.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nps import DEFAULT_BAND_CUTOFFS, TorchNPS

EPS = 1e-8


def _log_sq_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared difference of log NPS. NPS spans decades, so log is the scale."""
    return (torch.log(pred + EPS) - torch.log(target.detach() + EPS)).pow(2)


def _rel_sq_diff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Linear control for the log form: squared *relative* difference.

    The log form is a modelling choice, not a given, and it has never been
    tested here. A plain squared difference would be dominated by the low band
    (which carries decades more power), so the honest linear comparator
    normalises by the target - same scale invariance, no log.
    """
    t = target.detach()
    return ((pred - t) / (t.abs() + EPS)).pow(2)


def extract_patches(image: torch.Tensor, coords: torch.Tensor, size: int) -> torch.Tensor:
    """`[B,1,H,W]` and `[B,R,K,2]` top-left coords -> `[B,R,K,size,size]`."""
    b, r, k, _ = coords.shape
    dev = image.device
    ar = torch.arange(size, device=dev)
    gy = coords[..., 0, None, None] + ar.view(1, 1, 1, size, 1)
    gx = coords[..., 1, None, None] + ar.view(1, 1, 1, 1, size)
    h, w = image.shape[-2:]
    gy = gy.clamp(0, h - 1)
    gx = gx.clamp(0, w - 1)
    bi = torch.arange(b, device=dev).view(b, 1, 1, 1, 1).expand(b, r, k, size, size)
    return image[:, 0][bi, gy, gx]



def rois_from_masks(
    masks: torch.Tensor,
    roi_size: int,
    stride: int,
    min_coverage: float,
    k: int,
    erode_px: int = 0,
    soft_weight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best covered ROI positions per region, straight off the GPU.

    `masks` is `[B, R, H, W]`. A strided average pool is the coverage of every
    candidate window, so the search costs one pooling op instead of a python
    loop over positions. Returns `[B, R, k, 2]` top-left coords and a
    `[B, R, k]` validity mask; a region with no window above `min_coverage`
    comes back all-invalid and the loss then routes it to the batch fallback.

    With `soft_weight`, the second return is the window's coverage rather than a
    boolean, so an admitted ROI contributes in proportion to how much of it
    really is that tissue. Combined with a soft mask (`pack_soft_masks.py`) the
    coverage is a mean tissue *probability*, and the whole region term becomes
    continuous in the segmentation rather than stepping across a one-pixel
    boundary. `_roi_forward` consumes `valid` as a float weight throughout, so
    nothing downstream has to change.
    """
    b, r, h, w = masks.shape
    m = masks.reshape(b * r, 1, h, w).float()
    if erode_px > 0:
        k = 2 * erode_px + 1
        m = -F.max_pool2d(-m, k, stride=1, padding=erode_px)
    cov = F.avg_pool2d(m, roi_size, stride=stride)
    g = cov.shape[-1]
    flat = cov.reshape(b, r, -1)
    k = min(k, flat.shape[-1])
    top, idx = flat.topk(k, dim=-1)
    y = (idx // g) * stride
    x = (idx % g) * stride
    admitted = top >= min_coverage
    return torch.stack([y, x], dim=-1), (top * admitted) if soft_weight else admitted


class _NPSLossBase(nn.Module):
    def __init__(
        self,
        roi_size: int = 64,
        cutoffs: tuple[float, float] = DEFAULT_BAND_CUTOFFS,
        reduction: str = "band",
        field: str = "image",
        detrend: int = 0,
        log_domain: bool = True,
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.cutoffs = cutoffs
        self.reduction = reduction
        self.field = field
        self.log_domain = log_domain
        self.nps = TorchNPS(roi_size, detrend=detrend)

    def _fields(self, pred, target, noisy):
        """What the spectrum is taken of.

        `image` measures the spectrum of the prediction and the reference inside
        a homogeneous ROI, the phantom convention and what `viz_toolbox` does.

        `residual` follows CT-Mamba (Li et al., CMIG 2025, Sec. 2.4): form
        `noise_GT = x - y` and `noise_pred = x - y_hat` and take the spectra of
        those. The anatomy cancels in the subtraction, so the ROI no longer has
        to be anatomically flat - only inside the tissue class. The caveat is
        that `x - y_hat` is the noise the model *removed*, which equals the
        noise it *left* only if the model is unbiased on the signal, so the two
        estimators are compared rather than one assumed better.
        """
        if self.field == "residual":
            assert noisy is not None, "residual field needs the low dose input"
            return noisy - pred, noisy - target
        return pred, target

    def _curves(self, rois: torch.Tensor) -> torch.Tensor:
        """`[..., s, s]` -> `[..., n_bins]`."""
        shape = rois.shape[:-2]
        flat = rois.reshape(-1, self.roi_size, self.roi_size)
        return self.nps(flat).reshape(*shape, self.nps.n_bins)

    def _reduce(self, curve_p: torch.Tensor, curve_t: torch.Tensor) -> torch.Tensor:
        """`[..., n_bins]` pair -> `[..., 3]` band terms or `[..., 1]` curve term."""
        diff = _log_sq_diff if getattr(self, "log_domain", True) else _rel_sq_diff
        if self.reduction == "curve":
            d = diff(curve_p[..., 1:], curve_t[..., 1:])
            return d.mean(dim=-1, keepdim=True)
        bp = self.nps.band_energy(curve_p.reshape(-1, self.nps.n_bins), self.cutoffs)
        bt = self.nps.band_energy(curve_t.reshape(-1, self.nps.n_bins), self.cutoffs)
        return diff(bp, bt).reshape(*curve_p.shape[:-1], len(self.cutoffs) + 1)


class GlobalNPSLoss(_NPSLossBase):
    """Anatomy blind NPS loss: cells A (uniform) and B (band weighted), and A0.

    `sampling="center"` takes the central ROI (deterministic, used by the parity
    test). `sampling="random"` draws `n_rois` ROIs anywhere in the crop.
    `sampling="body"` draws them from the body mask and is the default for
    training.

    The body restriction matters in the regenerated `noisy_full` domain: air
    outside the patient is exactly 0 in the target (air is -1024 HU, the floor
    of the scale) while the low dose simulation leaves half-wave rectified noise
    there, so an ROI in air compares a finite spectrum against an identically
    zero one and contributes a `log(eps)` term with no anatomical meaning.

    Measured on one validation batch: the tissue anchored patch sampler already
    keeps crops 83% inside the body, so only 8.5% of candidate 32x32 windows are
    under 10% body. The effect is therefore modest rather than fatal - the loss
    moves from 4.25 (body) to 3.87 +- 0.15 (random) - but body sampling is
    still preferred on two counts: it makes the training loss measure the same
    thing the evaluator measures, and being a deterministic top-k it removes the
    sampling variance entirely (+- 0.00 vs +- 0.15 across repeats).
    """

    def __init__(
        self,
        roi_size: int = 64,
        band_weighted: bool = False,
        band_weights: torch.Tensor | None = None,
        cutoffs: tuple[float, float] = DEFAULT_BAND_CUTOFFS,
        sampling: str = "center",
        n_rois: int = 8,
        reduction: str = "band",
        normalise_weights: bool = True,
        body_coverage: float = 0.9,
        field: str = "image",
        detrend: int = 0,
        log_domain: bool = True,
    ) -> None:
        super().__init__(roi_size, cutoffs, reduction, field, detrend, log_domain)
        self.sampling = sampling
        self.n_rois = n_rois
        self.body_coverage = body_coverage
        w = band_weights if band_weights is not None else torch.ones(3)
        if not band_weighted:
            w = torch.ones(3)
        if normalise_weights:
            w = w / w.mean()
        self.register_buffer("band_weights", w.float())

    def _sample(self, image: torch.Tensor, masks: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = image.shape
        s = self.roi_size
        if self.sampling == "center":
            y = torch.full((b, 1, 1), (h - s) // 2, device=image.device, dtype=torch.long)
            x = torch.full((b, 1, 1), (w - s) // 2, device=image.device, dtype=torch.long)
            c = torch.cat([y, x], dim=-1).unsqueeze(1)
            return c, torch.ones(b, 1, 1, dtype=torch.bool, device=image.device)
        if self.sampling == "body" and masks is not None:
            body = masks.amax(dim=1, keepdim=True)
            coords, valid = rois_from_masks(body, s, max(1, s // 4), self.body_coverage, self.n_rois)
            if valid.any():
                return coords, valid
        y = torch.randint(0, h - s + 1, (b, self.n_rois, 1), device=image.device)
        x = torch.randint(0, w - s + 1, (b, self.n_rois, 1), device=image.device)
        c = torch.cat([y, x], dim=-1).unsqueeze(1)
        return c, torch.ones(b, 1, self.n_rois, dtype=torch.bool, device=image.device)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, masks: torch.Tensor | None = None,
        noisy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c, valid = self._sample(pred, masks)
        fp, ft = self._fields(pred, target, noisy)
        rp = extract_patches(fp, c, self.roi_size)
        rt = extract_patches(ft, c, self.roi_size)
        v = valid.float().unsqueeze(-1)
        n = v.sum(dim=2).clamp(min=1.0)
        cp = (self._curves(rp) * v).sum(dim=2) / n  # [B, 1, bins]
        ct = (self._curves(rt) * v).sum(dim=2) / n
        have = (valid.any(dim=2)).float().unsqueeze(-1)  # [B, 1, 1]
        terms = self._reduce(cp, ct) * have
        if self.reduction == "curve":
            return terms.sum() / have.sum().clamp(min=1e-6)
        w = self.band_weights
        return (terms * w).sum() / (have * w).sum().clamp(min=1e-6)


class TCBWNPSLoss(_NPSLossBase):
    """Tissue conditional, band weighted NPS loss: cells C (uniform w) and D.

    `weights` is `[R, 3]`, region major, in `nps.CLASS_NAMES` order. It is
    normalised to unit mean so the total NPS mass is constant across every cell
    of the ablation and a single `lambda_nps` stays comparable (Week 5 asks for
    exactly this).
    """

    def __init__(
        self,
        roi_size: int = 64,
        n_classes: int = 3,
        weights: torch.Tensor | None = None,
        band_weighted: bool = True,
        cutoffs: tuple[float, float] = DEFAULT_BAND_CUTOFFS,
        mode: str = "roi",
        n_rois: int = 4,
        reduction: str = "band",
        normalise_weights: bool = True,
        batch_fallback: bool = True,
        roi_stride: int | None = None,
        min_coverage: float = 0.85,
        field: str = "image",
        detrend: int = 0,
        erode_px: int = 0,
        inverse_scale: bool = False,
        soft_weight: bool = False,
        log_domain: bool = True,
        lowcontrast_gamma: float = 0.0,
    ) -> None:
        super().__init__(roi_size, cutoffs, reduction, field, detrend, log_domain)
        # Weight each admitted ROI by how flat it is. Low contrast regions are
        # where over smoothing actually costs a radiologist a lesion, and they
        # are exactly the windows a coverage-ranked sampler picks last. gamma=0
        # is the current behaviour; gamma=1 makes an ROI's weight inversely
        # proportional to its own standard deviation (normalised to unit mean,
        # so the loss scale and lambda stay comparable).
        self.lowcontrast_gamma = float(lowcontrast_gamma)
        self.erode_px = erode_px
        self.soft_weight = soft_weight
        self.inverse_scale = inverse_scale
        self.n_classes = n_classes
        self.mode = mode
        self.n_rois = n_rois
        self.roi_stride = roi_stride if roi_stride is not None else max(1, roi_size // 4)
        self.min_coverage = min_coverage
        self.batch_fallback = batch_fallback
        # Band count follows `cutoffs`. It was hard coded to 3 here, so cell C
        # (tissue conditioning, uniform over bands) could not be built at any
        # other resolution - which is exactly the control needed to separate
        # "more bands" from "weighted bands".
        n_bands = len(cutoffs) + 1
        w = weights if weights is not None else torch.ones(n_classes, n_bands)
        if not band_weighted:
            w = w.mean(dim=1, keepdim=True).expand(-1, n_bands).contiguous()
        if normalise_weights:
            w = w / w.mean()
        self.register_buffer("weights", w.float())

    # -- masked whole image variant (paper_plan.md Week 4 sketch) -------------
    def _masked_forward(self, pred, target, masks, noisy=None) -> torch.Tensor:
        pred, target = self._fields(pred, target, noisy)

        def profile(img: torch.Tensor) -> torch.Tensor:
            masked = img * masks  # [B, R, H, W]
            area = masks.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            masked = masked - masked.sum(dim=(-2, -1), keepdim=True) / area
            masked = masked * masks
            power = torch.fft.fft2(masked).abs().pow(2) / masked[0, 0].numel()
            return power

        h, w = pred.shape[-2:]
        b_idx, freq, bins = _radial_index(h, w, pred.device)
        pp, pt = profile(pred), profile(target)
        cp = _radial_mean(pp, b_idx, bins)
        ct = _radial_mean(pt, b_idx, bins)
        sel = [
            *[((freq > 0) if i == 0 else (freq >= lo)) & (freq < hi)
              for i, (lo, hi) in enumerate(zip([0.0, *self.cutoffs[:-1]], self.cutoffs))],
            freq >= self.cutoffs[-1],
        ]
        bp = torch.stack([cp[..., s].mean(-1) for s in sel], dim=-1)
        bt = torch.stack([ct[..., s].mean(-1) for s in sel], dim=-1)
        usable = (masks.sum(dim=(-2, -1)) > 0).float().unsqueeze(-1)
        terms = _log_sq_diff(bp, bt) * usable
        cell = terms.sum(0) / usable.sum(0).clamp(min=1.0)
        return cell, (usable.sum(0) > 0).float().expand_as(cell)

    # -- ROI variant ---------------------------------------------------------
    def _roi_forward(self, pred, target, coords, valid, noisy=None) -> torch.Tensor:
        fp, ft = self._fields(pred, target, noisy)
        rp = extract_patches(fp, coords, self.roi_size)
        rt = extract_patches(ft, coords, self.roi_size)
        cp = self._curves(rp)  # [B, R, K, bins]
        ct = self._curves(rt)
        v = valid.float()  # [B, R, K]
        if self.lowcontrast_gamma > 0:
            # contrast measured on the clean target, detached: it is a property
            # of the anatomy, not something the network may push around
            sigma = rt.detach().std(dim=(-2, -1)).clamp(min=1e-6)  # [B, R, K]
            ref = (sigma * v).sum() / v.sum().clamp(min=1.0)  # mean over admitted
            lc = (ref / sigma).pow(self.lowcontrast_gamma)
            lc = lc / ((lc * v).sum() / v.sum().clamp(min=1.0)).clamp(min=1e-6)
            v = v * lc
        v = v.unsqueeze(-1)  # [B, R, K, 1]
        n = v.sum(dim=2)  # [B, R, 1]
        cp_m = (cp * v).sum(dim=2) / n.clamp(min=1.0)
        ct_m = (ct * v).sum(dim=2) / n.clamp(min=1.0)

        have = (n.squeeze(-1) > 0).float()  # [B, R]
        if self.batch_fallback:
            pooled_n = v.sum(dim=(0, 2))  # [R, 1]
            pooled_p = (cp * v).sum(dim=(0, 2)) / pooled_n.clamp(min=1.0)
            pooled_t = (ct * v).sum(dim=(0, 2)) / pooled_n.clamp(min=1.0)
            fill = (pooled_n.squeeze(-1) > 0).float()  # [R]
            miss = (1.0 - have) * fill
            cp_m = cp_m + miss.unsqueeze(-1) * pooled_p.unsqueeze(0)
            ct_m = ct_m + miss.unsqueeze(-1) * pooled_t.unsqueeze(0)
            have = have + miss

        terms = self._reduce(cp_m, ct_m)  # [B, R, 3] or [B, R, 1]
        cell = (terms * have.unsqueeze(-1)).sum(0) / have.sum(0).clamp(min=1.0).unsqueeze(-1)
        cell_valid = (have.sum(0) > 0).float().unsqueeze(-1).expand_as(cell)
        return cell, cell_valid

    def cells(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        noisy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per (region, band) loss terms and their validity, both `[R, 3]`.

        Exposed so the Week 7 meta learners can weight the cells themselves
        instead of re-deriving the spectrum.
        """
        if self.mode == "masked":
            assert masks is not None, "masked mode needs [B, R, H, W] masks"
            return self._masked_forward(pred, target, masks, noisy)
        if self.mode == "center":
            b, _, h, w = pred.shape
            s = self.roi_size
            c = torch.tensor([[(h - s) // 2, (w - s) // 2]], device=pred.device)
            coords = c.view(1, 1, 1, 2).expand(b, self.n_classes, 1, 2)
            valid = torch.ones(b, self.n_classes, 1, device=pred.device)
        if coords is None or valid is None:
            assert masks is not None, "roi mode needs masks or cached ROI coords"
            coords, valid = rois_from_masks(
                masks, self.roi_size, self.roi_stride, self.min_coverage, self.n_rois,
                erode_px=self.erode_px, soft_weight=self.soft_weight,
            )
        return self._roi_forward(pred, target, coords, valid, noisy)

    def forward(self, pred, target, masks=None, coords=None, valid=None, noisy=None) -> torch.Tensor:
        cell, cell_valid = self.cells(pred, target, masks, coords, valid, noisy)
        w = self.weights if self.reduction == "band" else self.weights.mean(1, keepdim=True)
        if self.inverse_scale:
            # Each cell contributes equally regardless of its natural scale.
            # Detached, so this rescales the gradient without adding a path
            # through the weights themselves.
            w = w / cell.detach().clamp(min=1e-6)
        return (cell * w * cell_valid).sum() / (w * cell_valid).sum().clamp(min=1e-6)


class MetaWeightedTCBWNPS(nn.Module):
    """Learn the 3x3 weight matrix instead of hand tuning it (Week 7).

    Three schemes, all over the same `TCBWNPSLoss.cells` output:

    - `uncertainty` — Kendall et al., CVPR 2018, the plan's primary scheme.
      Each cell gets a learnable log variance `s`, and the objective becomes
      `sum_rb exp(-s_rb) L_rb + s_rb`. The `+ s` term is what stops the trivial
      solution of sending every weight to zero.
    - `gradnorm` — a **simplified** GradNorm (Chen et al., ICML 2018), the
      plan's fallback. True GradNorm equalises the per-task gradient norms on
      the shared parameters, which costs one backward pass per cell (nine here).
      This implementation keeps only GradNorm's relative-inverse-training-rate
      term, `r_i = (L_i / L_i(0)) / mean_j(...)`, and sets `w_i ∝ r_i^alpha`.
      It equalises learning *speed* without measuring gradient magnitude, so it
      must be reported as "GradNorm-style rate weighting", not as GradNorm.
    - `hard` — plain gradient ascent on the weights (a GDRO / focal style
      heuristic): mass moves to whichever cell is still mismatched.

    All three keep the total weight mass fixed at `R * B` so that a single
    `lambda_aux` stays comparable with the hand tuned runs.
    """

    def __init__(self, base: "TCBWNPSLoss", scheme: str = "uncertainty", alpha: float = 0.5) -> None:
        super().__init__()
        self.base = base
        self.scheme = scheme
        self.alpha = alpha
        n = base.n_classes
        # Band count follows the base loss. It was hard coded to 3, which kept
        # the learned-weight arms out of the higher band resolutions even though
        # the three-band reduction is what loses to the full curve.
        # One cell per band, or a single cell per tissue when the base matches the
        # whole curve. Deriving this from `cutoffs` alone made the learned-weight
        # schemes unusable with `reduction="curve"` - the cell that has the
        # curve's frequency resolution and tissue conditioning at the same time.
        nb = 1 if base.reduction == "curve" else len(base.cutoffs) + 1
        self.raw = nn.Parameter(torch.zeros(n, nb))
        self.register_buffer("running", torch.ones(n, nb))
        self.register_buffer("initialised", torch.zeros(()))

    def current_weights(self) -> torch.Tensor:
        if self.scheme == "uncertainty":
            w = torch.exp(-self.raw)
        else:
            w = torch.softmax(self.raw.reshape(-1), dim=0).reshape(self.raw.shape) * self.raw.numel()
        return w.detach()

    def forward(self, pred, target, masks=None, coords=None, valid=None, noisy=None) -> torch.Tensor:
        cell, cell_valid = self.base.cells(pred, target, masks, coords, valid, noisy)
        n = cell_valid.sum().clamp(min=1.0)
        if self.scheme == "uncertainty":
            weighted = (torch.exp(-self.raw) * cell + self.raw) * cell_valid
            return weighted.sum() / n
        if self.scheme == "gradnorm":
            # Relative inverse training rate r_i = (L_i / L_i(0)) / mean_j(...),
            # weights proportional to r_i^alpha (Chen et al., ICML 2018). The
            # loss at step 0 is the reference, not a running mean, otherwise the
            # rate collapses onto whichever cell moved last.
            with torch.no_grad():
                if not bool(self.initialised):
                    self.running.copy_(cell.detach().clamp(min=1e-8))
                    self.initialised.fill_(1)
                rate = cell.detach().clamp(min=1e-8) / self.running
                rate = rate / rate[cell_valid > 0].mean().clamp(min=1e-8)
                new_raw = self.alpha * torch.log(rate.clamp(min=1e-3, max=1e3))
                self.raw.data = 0.9 * self.raw.data + 0.1 * new_raw
            w = torch.softmax(self.raw.reshape(-1), 0).reshape(self.raw.shape) * self.raw.numel()
            return (w * cell * cell_valid).sum() / n
        w = torch.softmax(self.raw.reshape(-1), 0).reshape(self.raw.shape) * self.raw.numel()
        return (w * cell * cell_valid).sum() / n


_RADIAL_CACHE: dict = {}


def _radial_index(h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    key = (h, w, str(device))
    if key not in _RADIAL_CACHE:
        import numpy as np

        bins = (min(h, w) >> 1) + 1
        df = 0.5 / (bins - 1)
        fy = np.where(np.arange(h) <= h / 2, np.arange(h), np.arange(h) - h) / h
        fx = np.where(np.arange(w) <= w / 2, np.arange(w), np.arange(w) - w) / w
        r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        b = np.minimum(bins - 1, np.round(r / df)).astype(np.int64)
        b = np.where(r <= 0.5 + 1e-12, b, bins - 1)
        idx = torch.from_numpy(b.reshape(-1)).to(device)
        counts = torch.bincount(idx, minlength=bins).clamp(min=1).float()
        freq = torch.arange(bins, device=device).float() * df
        _RADIAL_CACHE[key] = (idx, counts, freq, bins)
    idx, counts, freq, bins = _RADIAL_CACHE[key]
    return (idx, counts), freq, bins


def _radial_mean(power: torch.Tensor, b_idx, bins: int) -> torch.Tensor:
    idx, counts = b_idx
    shape = power.shape[:-2]
    flat = power.reshape(-1, power.shape[-2] * power.shape[-1])
    out = torch.zeros(flat.shape[0], bins, device=power.device, dtype=flat.dtype)
    out.index_add_(1, idx, flat)
    out = out / counts
    out = torch.cat([torch.zeros_like(out[:, :1]), out[:, 1:]], dim=1)
    return out.reshape(*shape, bins)


class FocalFrequencyLoss(nn.Module):
    """Jiang et al., ICCV 2021. Baseline for 'is CT specific structure needed?'."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = torch.fft.fft2(pred, norm="ortho")
        ft = torch.fft.fft2(target, norm="ortho")
        diff = torch.stack([fp.real - ft.real, fp.imag - ft.imag], dim=-1)
        d = diff.pow(2).sum(dim=-1)
        w = d.detach().pow(self.alpha)
        w = w / w.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
        return (w * d).mean()


class PearsonDivergenceLoss(nn.Module):
    """Texture-preserving histogram loss (Oh et al., Phys Med Biol 2024).

    Matches the intensity distribution of prediction and target with a Pearson
    divergence estimated through a differentiable kernel density estimate, so it
    preserves texture without adversarial training. Listed in the project's
    reference index as a required comparison baseline for the
    "texture-preserving loss" family.
    """

    def __init__(self, bins: int = 64, sigma: float = 0.02, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.sigma = sigma
        self.register_buffer("centers", torch.linspace(0.0, 1.0, bins).view(1, bins, 1))

    def _hist(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], 1, -1)
        k = torch.exp(-0.5 * ((flat - self.centers) / self.sigma) ** 2)
        h = k.mean(dim=-1)
        return h / h.sum(dim=1, keepdim=True).clamp(min=self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p, q = self._hist(pred), self._hist(target).detach()
        return (((p - q) ** 2) / (q + self.eps)).sum(dim=1).mean()


class VGGPerceptualLoss(nn.Module):
    """WGAN-VGG style feature loss (Yang et al., TMI 2018), generator side only."""

    def __init__(self, layer: int = 16) -> None:
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:layer].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1).repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.vgg(self._prep(pred)), self.vgg(self._prep(target)))
