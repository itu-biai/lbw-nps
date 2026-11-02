#!/usr/bin/env python3
"""Minimum Reporting Set (MRS) evaluation.

Metric set follows the sibling project `viz_toolbox_and_nps_eval_metric`:
PSNR, SSIM, global NPS Gap, band NPS ratios (Low/Mid/High), LPIPS, DTS v2, plus
the per tissue NPS Gap that is native to this paper.

Two NPS measurement scales are reported and they are not interchangeable:

- **global**, on 64x64 ROIs inside the body mask, which is the viz_toolbox
  default and keeps these numbers comparable with the lab's prior reports;
- **per tissue**, on 32x32 ROIs inside each class mask. 64x64 was measured
  first and had to be abandoned: no 64x64 window anywhere in the dataset is
  85% bone (0 of 220 test slices), because ribs and cortex are thin. At 32x32
  the bone class is measurable on 23% of slices and, pooled over a batch or a
  split, gives a stable curve. The cost is frequency resolution: 17 radial bins
  instead of 33.
"""

from __future__ import annotations

import numpy as np
import torch

from nps import (
    BAND_NAMES,
    CLASS_NAMES,
    DEFAULT_BAND_CUTOFFS,
    RoiSpec,
    band_gaps,
    band_ratios,
    detrend_operator,
    find_class_rois,
    nps_freq_axis,
    nps_from_roi,
    nps_gap,
    nps_gap_log,
)

GLOBAL_ROI = RoiSpec(size=64, stride=32, min_coverage=0.95, max_rois=16)
TISSUE_ROI = RoiSpec(size=32, stride=8, min_coverage=0.85, max_rois=16)
# AAPM TG 233 specifies a second order polynomial detrend of every NPS ROI.
# Measured effect on this dataset: the ground truth Low band integral in soft
# tissue drops 1.7x (2.12e-5 -> 1.23e-5), i.e. 42% of what mean subtraction
# calls low frequency noise is anatomical shading. Mid and High are unchanged
# to four significant figures (574.4 -> 574.5, 1608 -> 1610), so this is a
# correctness fix for the Low band and leaves the paper's headline bands alone.
DETREND = 2
SOFT_WINDOW_HU = (-175.0, 275.0)


def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(data_range**2 / mse)


def to_soft_window(x: np.ndarray) -> np.ndarray:
    """Push a `gt_full` scaled array through the standard soft tissue window.

    `gt_full` occupies roughly [0, 0.5] of the [0, 1] HU scale, so its PSNR is
    inflated by ~8 dB relative to the windowed `gt_clipped` domain. Reporting a
    windowed PSNR alongside makes the two training domains comparable.
    """
    lo = (SOFT_WINDOW_HU[0] + 1024.0) / 4096.0
    hi = (SOFT_WINDOW_HU[1] + 1024.0) / 4096.0
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(gt, pred, data_range=data_range))


def gmsd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Gradient magnitude similarity deviation (Xue et al., TIP 2014)."""
    from scipy.ndimage import convolve

    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0
    ky = kx.T

    def grad(x):
        return np.sqrt(convolve(x, kx) ** 2 + convolve(x, ky) ** 2)

    gp, gg = grad(pred), grad(gt)
    c = 0.0026
    gms = (2 * gp * gg + c) / (gp**2 + gg**2 + c)
    return float(gms.std())


def global_nps(image: np.ndarray, body: np.ndarray, detrend: int = DETREND) -> np.ndarray | None:
    rois = find_class_rois(body, GLOBAL_ROI, image=image)
    if len(rois) == 0:
        return None
    return np.mean([nps_from_roi(image[y : y + 64, x : x + 64], detrend=detrend)
                    for y, x in rois], axis=0)


def tissue_nps(image: np.ndarray, rois: np.ndarray, n: int, size: int = 32,
               detrend: int = DETREND) -> np.ndarray | None:
    if n <= 0:
        return None
    return np.mean([nps_from_roi(image[y : y + size, x : x + size], detrend=detrend)
                    for y, x in rois[:n]], axis=0)


def roi_sigma_ratio(pred: np.ndarray, gt: np.ndarray, body: np.ndarray,
                    size: int = 64, detrend: int = DETREND) -> float:
    """Grain ratio: how much of the reference's noise texture survived.

    The direct over-smoothing number. In each flat body ROI the second-order
    trend is removed (same detrend as the NPS estimator), so what is left is
    the fluctuation, and its standard deviation is compared to the reference's.
    1.0 = the output is as grainy as normal-dose; 0.5 = half the grain, i.e.
    visibly waxy; >1 = noisier than the reference.
    """
    rois = find_class_rois(body, GLOBAL_ROI, image=gt)
    if len(rois) == 0:
        return float("nan")
    op = detrend_operator(size, detrend) if detrend >= 1 else None

    def fluct(roi: np.ndarray) -> float:
        x = roi.astype(np.float64)
        r = (op @ x.reshape(-1)).reshape(size, size) if op is not None else x - x.mean()
        return float(r.std())

    sp, sg = [], []
    for y, x in rois:
        sp.append(fluct(pred[y : y + size, x : x + size]))
        sg.append(fluct(gt[y : y + size, x : x + size]))
    m = float(np.mean(sg))
    return float(np.mean(sp) / m) if m > 0 else float("nan")


def band_score(ratio: float) -> float:
    """1.0 at a perfect ratio, decaying symmetrically in log space."""
    if not np.isfinite(ratio) or ratio <= 0:
        return 0.0
    return float(np.exp(-abs(np.log(ratio))))


def dts_v2(ratios: np.ndarray, lpips_value: float) -> float:
    """Equal weighted, band normalised, no PSNR term (so it is not circular).

    Implementation of the DTS v2 spec drafted in the sibling evaluation project:
    three band ratio scores plus one perceptual score, averaged.
    """
    comps = [band_score(r) for r in ratios] + [1.0 - float(np.clip(lpips_value, 0, 1))]
    return float(np.mean(comps))


class Evaluator:
    """Per slice MRS over a split, with per tissue NPS on cached ROIs."""

    def __init__(self, cutoffs=DEFAULT_BAND_CUTOFFS, device: str = "cuda", use_lpips: bool = True,
                 class_names: tuple[str, ...] = CLASS_NAMES):
        self.cutoffs = cutoffs
        self.class_names = class_names
        self.freq64 = nps_freq_axis(64)
        self.freq32 = nps_freq_axis(32)
        self.device = device
        self.lpips = None
        if use_lpips:
            import lpips as lpips_lib

            self.lpips = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()

    def _lpips(self, pred: np.ndarray, gt: np.ndarray) -> float:
        if self.lpips is None:
            return float("nan")
        with torch.no_grad():
            a = torch.from_numpy(pred).float()[None, None].to(self.device).clamp(0, 1)
            b = torch.from_numpy(gt).float()[None, None].to(self.device).clamp(0, 1)
            a, b = a.repeat(1, 3, 1, 1) * 2 - 1, b.repeat(1, 3, 1, 1) * 2 - 1
            return float(self.lpips(a, b).item())

    def slice_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        masks: np.ndarray,
        rois: np.ndarray,
        nrois: np.ndarray,
        windowed_psnr: bool = False,
    ) -> dict:
        body = (masks.sum(axis=0) > 0).astype(np.uint8)
        out: dict[str, float] = {
            "psnr": psnr(pred, gt),
            "ssim": ssim(pred, gt),
            "gmsd": gmsd(pred, gt),
            "lpips": self._lpips(pred, gt),
        }
        if windowed_psnr:
            out["psnr_window"] = psnr(to_soft_window(pred), to_soft_window(gt))
            out["ssim_window"] = ssim(to_soft_window(pred), to_soft_window(gt))

        out["sigma_ratio"] = roi_sigma_ratio(pred, gt, body)
        np_pred, np_gt = global_nps(pred, body), global_nps(gt, body)
        if np_pred is not None and np_gt is not None:
            out["nps_gap"] = nps_gap(np_pred, np_gt)
            out["nps_gap_log"] = nps_gap_log(np_pred, np_gt)
            ratios = band_ratios(self.freq64, np_pred, np_gt, self.cutoffs)
            gaps = band_gaps(self.freq64, np_pred, np_gt, self.cutoffs)
            for i, name in enumerate(BAND_NAMES):
                out[f"nps_ratio_{name}"] = float(ratios[i])
                out[f"nps_ratio_log_{name}"] = float(np.log(max(ratios[i], 1e-9)))
                out[f"nps_gap_{name}"] = float(gaps[i])
            out["dts_v2"] = dts_v2(ratios, out["lpips"])

        for c, name in enumerate(self.class_names):
            n = int(nrois[c])
            tp = tissue_nps(pred, rois[c], n)
            tg = tissue_nps(gt, rois[c], n)
            if tp is None or tg is None:
                continue
            out[f"nps_gap_{name}"] = nps_gap(tp, tg)
            out[f"nps_gap_log_{name}"] = nps_gap_log(tp, tg)
            tr = band_ratios(self.freq32, tp, tg, self.cutoffs)
            for i, band in enumerate(BAND_NAMES):
                out[f"nps_ratio_{name}_{band}"] = float(tr[i])
                out[f"nps_ratio_log_{name}_{band}"] = float(np.log(max(tr[i], 1e-9)))
        return out

    def curves(self, pred, gt, masks, rois, nrois) -> dict[str, np.ndarray]:
        """Raw NPS curves for pooled (ensemble) aggregation.

        A band ratio is a ratio of small numbers and its per slice distribution
        is heavy tailed: over 40 test slices the global high band ratio has
        median 0.386 but mean 2.77 and max 45.9, because the NDCT high band
        integral itself spans two decades (8e-8 to 7.4e-6). NPS is an ensemble
        quantity, so the defensible estimator averages the *curves* over slices
        and takes one ratio at the end. These arrays are what makes that
        possible downstream.
        """
        body = (masks.sum(axis=0) > 0).astype(np.uint8)
        out: dict[str, np.ndarray] = {}
        gp, gg = global_nps(pred, body), global_nps(gt, body)
        if gp is not None and gg is not None:
            out["global_pred"], out["global_gt"] = gp, gg
        for c, name in enumerate(self.class_names):
            n = int(nrois[c])
            tp, tg = tissue_nps(pred, rois[c], n), tissue_nps(gt, rois[c], n)
            if tp is not None and tg is not None:
                out[f"{name}_pred"], out[f"{name}_gt"] = tp, tg
        return out

    def tissue_curves(
        self, pred: np.ndarray, gt: np.ndarray, rois: np.ndarray, nrois: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        curves = {}
        for c, name in enumerate(self.class_names):
            n = int(nrois[c])
            tp, tg = tissue_nps(pred, rois[c], n), tissue_nps(gt, rois[c], n)
            if tp is not None:
                curves[name] = (tp, tg)
        return curves


def aggregate(records: list[dict]) -> dict[str, dict[str, float]]:
    keys = sorted({k for r in records for k in r})
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in records if k in r and np.isfinite(r[k])])
        if len(vals):
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": int(len(vals))}
    return out
