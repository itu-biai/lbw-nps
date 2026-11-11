"""Radial NPS estimation in numpy (evaluation) and torch (training losses).

The estimator follows the BIAI Lab viz-toolbox (https://github.com/itu-biai/viz-toolbox):
mean subtraction, separable Hann window, 2D FFT, `|F|^2 / n^2`, radial average
into `n//2 + 1` bins over `f in [0, 0.5]` cycles/pixel with `round(r/df)`
binning, and DC forced to zero. The NPS gap is the root mean square difference
of `log1p` curves (`logRmse` in viz-toolbox).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Default band cutoffs in cycles/pixel. Low [0, 0.15), Mid [0.15, 0.35),
# High [0.35, 0.5].
DEFAULT_BAND_CUTOFFS: tuple[float, float] = (0.15, 0.35)
BAND_NAMES: tuple[str, str, str] = ("low", "mid", "high")
CLASS_NAMES: tuple[str, str, str] = ("lung", "soft", "bone")

# Tissue classes carried by each mask source. All reported runs use `ts`
# (TotalSegmentator); `fine` splits soft tissue into fat, muscle and organ.
CLASS_SETS: dict[str, tuple[str, ...]] = {
    "ts": ("lung", "soft", "bone"),
    "hu": ("lung", "soft", "bone"),
    "cons": ("lung", "soft", "bone"),
    "fine": ("fat", "muscle", "organ", "bone"),
}


# --------------------------------------------------------------------------
# numpy side (evaluation)
# --------------------------------------------------------------------------


def poly_basis(n: int, order: int = 2) -> np.ndarray:
    """Design matrix for a 2D polynomial surface over an n x n ROI.

    AAPM TG 233 (Samei et al., Med Phys 2019) specifies **second order
    polynomial detrending** of every NPS ROI before the Fourier transform. Mean
    subtraction alone (order 0, as in viz-toolbox) leaves any
    residual shading or anatomical gradient inside the ROI, and that gradient is
    broadband low-frequency energy that lands directly in the Low band.
    """
    y, x = np.mgrid[0:n, 0:n]
    y = (y - (n - 1) / 2) / max(n - 1, 1)
    x = (x - (n - 1) / 2) / max(n - 1, 1)
    cols = [np.ones_like(x)]
    if order >= 1:
        cols += [x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.stack([c.ravel() for c in cols], axis=1)


_DETREND_CACHE: dict = {}


def detrend_operator(n: int, order: int = 2) -> np.ndarray:
    """`I - A pinv(A)`: projects an ROI onto the complement of the polynomial."""
    key = (n, order)
    if key not in _DETREND_CACHE:
        a = poly_basis(n, order)
        _DETREND_CACHE[key] = np.eye(n * n) - a @ np.linalg.pinv(a)
    return _DETREND_CACHE[key]


def hann2d(n: int) -> np.ndarray:
    """Return a separable `n x n` Hann window."""
    w = np.hanning(n)
    return np.outer(w, w)


def radial_bins(n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Bin index per FFT sample and the frequency axis (viz-toolbox convention)."""
    bins = (n >> 1) + 1
    df = 0.5 / (bins - 1)
    idx = np.arange(n)
    f = np.where(idx <= n / 2, idx, idx - n) / n
    fy, fx = np.meshgrid(f, f, indexing="ij")
    r = np.sqrt(fx**2 + fy**2)
    b = np.minimum(bins - 1, np.round(r / df)).astype(np.int64)
    b = np.where(r <= 0.5 + 1e-12, b, -1)
    return b, np.arange(bins) * df, bins


def nps_from_roi(roi: np.ndarray, window: bool = True, detrend: int = 0) -> np.ndarray:
    """1D radially averaged NPS of one square ROI.

    `detrend=0` subtracts the mean (viz-toolbox parity, the default).
    `detrend=2` is the AAPM TG-233 second-order polynomial detrend.
    """
    n = roi.shape[0]
    x = roi.astype(np.float64)
    if detrend >= 1:
        centered = (detrend_operator(n, detrend) @ x.reshape(-1)).reshape(n, n)
    else:
        centered = x - x.mean()
    signal = centered * hann2d(n) if window else centered
    power = np.abs(np.fft.fft2(signal)) ** 2 / (n * n)
    b, _, bins = radial_bins(n)
    valid = b >= 0
    sums = np.bincount(b[valid], weights=power[valid], minlength=bins)
    counts = np.bincount(b[valid], minlength=bins)
    nps = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    nps[0] = 0.0
    return nps


def nps_freq_axis(n: int) -> np.ndarray:
    """Frequency axis (cycles/pixel) of the radial NPS of an `n x n` ROI."""
    return radial_bins(n)[1]


def nps_gap(nps_a: np.ndarray, nps_b: np.ndarray) -> float:
    """NPS gap (viz-toolbox `logRmse`): RMS difference of `log1p` curves."""
    va = np.log1p(np.maximum(0.0, nps_a))
    vb = np.log1p(np.maximum(0.0, nps_b))
    d = va - vb
    good = np.isfinite(d)
    return float(np.sqrt(np.mean(d[good] ** 2))) if good.any() else float("nan")


def nps_gap_log(nps_a: np.ndarray, nps_b: np.ndarray, floor: float = 1e-6) -> float:
    """Scale invariant NPS Gap: RMS of the log ratio between two curves.

    The viz-toolbox `logRmse` uses `log1p`, which is linear for arguments far
    below 1. With `|F|^2 / n^2` normalisation a CT ROI's NPS sits around 1e-4,
    so `log1p` never leaves its linear regime and the metric degenerates into an
    absolute difference dominated by the low frequency bins: the raw low dose
    input, whose high band carries 33x the ground truth energy, scores a
    `nps_gap` of 0.0033. This variant divides instead of subtracting, so a
    factor of two is the same distance wherever it happens on the curve.
    """
    b = np.maximum(0.0, nps_b)
    a = np.maximum(0.0, nps_a)
    ref = float(b[b > 0].mean()) if (b > 0).any() else 0.0
    if ref <= 0:
        return float("nan")
    eps = floor * ref
    sel = b > eps
    if sel.sum() < 2:
        return float("nan")
    d = np.log(a[sel] + eps) - np.log(b[sel] + eps)
    return float(np.sqrt(np.mean(d**2)))


def band_slices(freq: np.ndarray, cutoffs: tuple[float, float]) -> list[np.ndarray]:
    """Boolean masks selecting the low, mid and high bands of a frequency axis."""
    lo, hi = cutoffs
    return [
        (freq > 0) & (freq < lo),
        (freq >= lo) & (freq < hi),
        (freq >= hi),
    ]


def band_integrals(
    freq: np.ndarray, nps: np.ndarray, cutoffs: tuple[float, float] = DEFAULT_BAND_CUTOFFS
) -> np.ndarray:
    """Trapezoidal integral of the NPS curve inside each band."""
    out = []
    for sel in band_slices(freq, cutoffs):
        if sel.sum() < 2:
            out.append(float(nps[sel].sum()))
        else:
            out.append(float(np.trapezoid(nps[sel], freq[sel])))
    return np.array(out)


def band_ratios(
    freq: np.ndarray,
    nps_pred: np.ndarray,
    nps_gt: np.ndarray,
    cutoffs: tuple[float, float] = DEFAULT_BAND_CUTOFFS,
) -> np.ndarray:
    """Per band NPS ratio pred/gt. 1.0 is a perfect texture match."""
    p = band_integrals(freq, nps_pred, cutoffs)
    g = band_integrals(freq, nps_gt, cutoffs)
    return p / np.maximum(g, 1e-12)


def band_gaps(
    freq: np.ndarray,
    nps_pred: np.ndarray,
    nps_gt: np.ndarray,
    cutoffs: tuple[float, float] = DEFAULT_BAND_CUTOFFS,
) -> np.ndarray:
    """Per band NPS Gap (logRmse restricted to the band's bins)."""
    return np.array([nps_gap(nps_pred[sel], nps_gt[sel]) for sel in band_slices(freq, cutoffs)])


# --------------------------------------------------------------------------
# ROI sampling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RoiSpec:
    """ROI search settings: size, stride, coverage, maximum count, erosion and homogeneity."""

    size: int = 64
    stride: int = 32
    min_coverage: float = 0.9
    max_rois: int = 16
    erode_px: int = 0
    homogeneity_sd: float = 1.0


def _box_sums(x: np.ndarray, s: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sum of `x` over every `s x s` box with top-left corner on the `ys x xs` grid."""
    integral = np.zeros((x.shape[0] + 1, x.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(x.astype(np.float64), axis=0), axis=1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return integral[yy + s, xx + s] - integral[yy, xx + s] - integral[yy + s, xx] + integral[yy, xx]


def find_class_rois(mask: np.ndarray, spec: RoiSpec, image: np.ndarray | None = None) -> np.ndarray:
    """Top-left coordinates of the ROIs best suited to an NPS measurement.

    Three filters, in the order the published per organ NPS methodology applies
    them (automated in vivo noise texture in liver CT, Med Phys; PMC8543153):

    1. **Erode the class mask** by `erode_px` before searching, so no ROI sits on
       an anatomical boundary. The reference uses a 9x9 structuring element
       (radius 4), but it measures the liver, a large homogeneous organ where
       erosion is free. Measured here on 120 test slices, radius 4 erosion cuts
       the slices with a usable bone ROI from 38% to 14% and muscle from 55% to
       13%, because ribs and muscle bands are only a few ROI-widths across.
       **Default 0**: the >= 85% coverage rule already caps off-class content at
       15% of a 32x32 window, and the coverage rule does not delete the class.
    2. **Coverage**: keep windows at least `min_coverage` inside the eroded mask.
    3. **Homogeneity**: when the reference `image` is given, drop windows whose
       pixel standard deviation exceeds the modal ROI standard deviation by more
       than `homogeneity_sd` standard deviations. This is what removes the ROIs
       that contain a vessel or a lesion, which would otherwise put structure
       into a noise measurement.

    Returns an empty `[0, 2]` array when no window qualifies.
    """
    h, w = mask.shape
    s, st = spec.size, spec.stride
    if h < s or w < s:
        return np.zeros((0, 2), dtype=np.int64)

    m = mask.astype(bool)
    if spec.erode_px > 0:
        from scipy.ndimage import binary_erosion

        r = spec.erode_px
        grid = np.ogrid[-r : r + 1, -r : r + 1]
        m = binary_erosion(m, (grid[0] ** 2 + grid[1] ** 2) <= r**2)
        if not m.any():
            return np.zeros((0, 2), dtype=np.int64)

    ys = np.arange(0, h - s + 1, st)
    xs = np.arange(0, w - s + 1, st)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cov = _box_sums(m, s, ys, xs) / (s * s)
    ok = cov >= spec.min_coverage
    if not ok.any():
        return np.zeros((0, 2), dtype=np.int64)

    if image is not None and spec.homogeneity_sd > 0:
        n = s * s
        mean = _box_sums(image, s, ys, xs) / n
        sq = _box_sums(np.asarray(image, dtype=np.float64) ** 2, s, ys, xs) / n
        std = np.sqrt(np.maximum(sq - mean**2, 0.0))
        vals = std[ok]
        if len(vals) >= 8:
            hist, edges = np.histogram(vals, bins=min(32, max(4, len(vals) // 2)))
            peak = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
            ok = ok & (std <= peak + spec.homogeneity_sd * vals.std())
            if not ok.any():
                return np.zeros((0, 2), dtype=np.int64)

    cand_y, cand_x = yy[ok], xx[ok]
    order = np.argsort(-cov[ok])[: spec.max_rois]
    return np.stack([cand_y[order], cand_x[order]], axis=1).astype(np.int64)


def nps_from_rois(image: np.ndarray, rois: np.ndarray, size: int) -> np.ndarray | None:
    """Mean NPS over a list of ROI top-left coordinates."""
    if len(rois) == 0:
        return None
    curves = [nps_from_roi(image[y : y + size, x : x + size]) for y, x in rois]
    return np.mean(curves, axis=0)


# --------------------------------------------------------------------------
# torch side (training loss)
# --------------------------------------------------------------------------


class TorchNPS(torch.nn.Module):
    """Differentiable radially averaged NPS for a batch of square ROIs.

    Buffers hold the Hann window and the scatter indices so the per step cost is
    one FFT plus one `index_add`.
    """

    def __init__(self, size: int, detrend: int = 0) -> None:
        super().__init__()
        self.size = size
        self.detrend = detrend
        if detrend >= 1:
            self.register_buffer(
                "detrend_op", torch.from_numpy(detrend_operator(size, detrend)).float()
            )
        b, freq, bins = radial_bins(size)
        self.n_bins = bins
        flat = torch.from_numpy(b.reshape(-1)).long()
        counts = torch.zeros(bins).index_add_(0, flat.clamp(min=0), (flat >= 0).float())
        self.register_buffer("bin_index", flat.clamp(min=0))
        self.register_buffer("bin_valid", (flat >= 0).float())
        self.register_buffer("bin_counts", counts.clamp(min=1.0))
        self.register_buffer("window", torch.from_numpy(hann2d(size)).float())
        self.register_buffer("freq", torch.from_numpy(freq).float())

    def forward(self, rois: torch.Tensor) -> torch.Tensor:
        """`[N, size, size]` ROIs -> `[N, n_bins]` NPS curves."""
        if self.detrend >= 1:
            flat = rois.reshape(rois.shape[0], -1)
            x = (flat @ self.detrend_op.T).reshape_as(rois)
        else:
            x = rois - rois.mean(dim=(-2, -1), keepdim=True)
        x = x * self.window
        power = torch.fft.fft2(x).abs().pow(2) / (self.size * self.size)
        flat = (power * self.bin_valid.view(self.size, self.size)).reshape(rois.shape[0], -1)
        sums = torch.zeros(rois.shape[0], self.n_bins, device=rois.device, dtype=flat.dtype)
        sums.index_add_(1, self.bin_index, flat)
        nps = sums / self.bin_counts
        return torch.cat([torch.zeros_like(nps[:, :1]), nps[:, 1:]], dim=1)

    def band_energy(
        self, nps: torch.Tensor, cutoffs: tuple[float, ...] = DEFAULT_BAND_CUTOFFS
    ) -> torch.Tensor:
        """`[N, n_bins]` -> `[N, len(cutoffs)+1]` mean NPS inside each band.

        Any number of cutoffs is allowed; all reported runs use two (three bands).
        """
        f = self.freq
        # Half open [lo, hi), so a bin landing exactly on a cutoff belongs to the
        # band above it and no bin can fall through the gaps. The first band
        # starts just above DC, the last one runs to Nyquist.
        edges = [0.0, *cutoffs]
        sels = []
        for i, lo in enumerate(edges):
            above = (f > 0) if i == 0 else (f >= lo)
            sels.append(above if i == len(edges) - 1 else above & (f < edges[i + 1]))
        return torch.stack([nps[:, s].mean(dim=1) for s in sels], dim=1)
