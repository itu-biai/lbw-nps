"""Texture check that does not reuse the training loss (addresses G_loc circularity).

G_loc compares log band energies on 64/32 px windows, which is close to what LBW-NPS minimises.
This script scores every arm with measures that share neither the band edges, the log band-energy
difference nor the window grid of the loss:

  f_av   AAPM TG-233 average NPS frequency (radial NPS, 1/32 cycles/pixel bins, DC excluded; the
         viz-toolbox "averageFrequency"), error = |f_av(output) - f_av(NDCT)|
  GLCM   grey-level co-occurrence texture (radiomics): contrast and homogeneity, 32 grey levels over
         [-160, 240] HU, distances 1 and 2 px, 4 angles; error = |log(contrast_out / contrast_ref)| and
         |homogeneity_out - homogeneity_ref|
  noise  |log(sd_out / sd_ref)| of the mean-removed windows (secondary; amplitude, not texture)

Windows: homogeneous 32x32 soft-tissue windows chosen on the NDCT image ONLY (smoothed HU in
[-50, 150], structure sd <= 15 HU, inside the body, non-overlapping, at most 8 per slice), so
every arm is scored on the same pixels. Slices without
such a window are skipped and counted.

Statistics: per slice, mean over seeds 42/123/456; LBW-NPS vs every other arm, two-sided paired
Wilcoxon, Holm over the whole family (arms x 3 primary errors), rank-biserial r; repeated on patient
means. Final checkpoints (last.pt) and the re-tuned FFL / VGG / Pearson runs, as in the paper.

This is a post-hoc check added after the main analysis; the metrics and the window rule were fixed
before the script was first run.

    python scripts/eval_independent_texture.py            # all 10 arms x 3 seeds, test split
    python scripts/eval_independent_texture.py --quick    # 20 slices, seed 42 only (smoke test)
Writes outputs/independent_texture/{per_slice.csv, summary.csv, stats.csv, run_info.txt}.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from config import OUT, PACKED, RUNS
from isbi_data import PackedSplit

ARMS = [  # run stem (seed suffix added), label
    ("l1", "L1"),
    ("pooled_uniform", "global uniform NPS"),
    ("pooled_weighted", "global band-weighted NPS"),
    ("ffl", "FFL (full crop)"),
    ("vgg", "VGG"),
    ("pearson", "Pearson"),
    ("tissue_pooling", "tissue masks (D)"),
    ("lbw_nps_p128", "LBW-NPS P=128"),
    ("lbw_nps_p32", "LBW-NPS P=32"),
    ("lbw_nps", "LBW-NPS (ours)"),
]
OURS = "lbw_nps"
SEEDS = (42, 123, 456)
N, MAX_WIN, MAX_SD, HU_RANGE = 32, 8, 15.0, (-50.0, 150.0)
GLCM_LO, GLCM_HI, LEVELS = -160.0, 240.0, 32
PRIMARY = ("fav_err", "glcm_contrast_err", "glcm_homog_err")


def hu(x: np.ndarray) -> np.ndarray:
    """Map normalized intensities to HU (same convention as `src/evaluate.py`)."""
    return x * 4096.0 - 1024.0


def gaussian_filter(a: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Separable Gaussian smoothing with reflect padding."""
    r = int(3 * sigma + 0.5)
    t = np.arange(-r, r + 1)
    g = np.exp(-(t**2) / (2 * sigma**2))
    g /= g.sum()
    b = np.pad(a, r, mode="reflect")
    b = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 0, b)
    return np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 1, b)


def flat_windows(ref_hu: np.ndarray) -> list[tuple[int, int]]:
    """Return top-left corners of up to `MAX_WIN` homogeneous soft-tissue windows."""
    sm = gaussian_filter(ref_hu)
    yy, xx = np.mgrid[0:N, 0:N]
    design = np.c_[np.ones(N * N), yy.ravel(), xx.ravel()]
    cand = []
    for y in range(0, ref_hu.shape[0] - N + 1, 8):
        for x in range(0, ref_hu.shape[1] - N + 1, 8):
            c = sm[y : y + N, x : x + N]
            if c.min() < HU_RANGE[0] or c.max() > HU_RANGE[1]:
                continue  # also excludes air outside the body (< -500)
            v = c.ravel()
            sd = float((v - design @ np.linalg.lstsq(design, v, rcond=None)[0]).std())
            if sd <= MAX_SD:
                cand.append((sd, y, x))
    cand.sort()
    picked: list[tuple[int, int]] = []
    for _, y, x in cand:
        if len(picked) == MAX_WIN:
            break
        if all(abs(y - py) >= N or abs(x - px) >= N for py, px in picked):
            picked.append((y, x))
    return picked


_W = np.outer(np.hanning(N), np.hanning(N))
_F = np.fft.fftfreq(N)
_B = np.round(np.hypot(*np.meshgrid(_F, _F, indexing="ij")) * N).astype(int)
_FB = np.arange(_B.max() + 1) / N
_M = (_FB > 0) & (_FB <= 0.5 + 1e-9)


def nps_stats(img_hu: np.ndarray, wins: list[tuple[int, int]]) -> tuple[float, float]:
    """Pooled radial NPS of the windows -> (average frequency, noise sd of the mean-removed windows)."""
    acc, var = 0.0, []
    for y, x in wins:
        p = img_hu[y : y + N, x : x + N]
        p = p - p.mean()
        var.append(p.var())
        acc = acc + np.abs(np.fft.fft2(p * _W)) ** 2 / (_W**2).sum()
    nps = acc / len(wins)
    prof = np.bincount(_B.ravel(), nps.ravel()) / np.bincount(_B.ravel())
    fav = float((_FB[_M] * prof[_M]).sum() / prof[_M].sum())
    return fav, float(np.sqrt(np.mean(var)))


def glcm_stats(img_hu: np.ndarray, wins: list[tuple[int, int]]) -> tuple[float, float]:
    """Mean GLCM contrast and homogeneity over the windows."""
    from skimage.feature import graycomatrix, graycoprops

    q = np.clip((img_hu - GLCM_LO) / (GLCM_HI - GLCM_LO), 0, 1 - 1e-9)
    q = (q * LEVELS).astype(np.uint8)
    con, hom = [], []
    for y, x in wins:
        g = graycomatrix(
            q[y : y + N, x : x + N],
            distances=[1, 2],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=LEVELS,
            symmetric=True,
            normed=True,
        )
        con.append(graycoprops(g, "contrast").mean())
        hom.append(graycoprops(g, "homogeneity").mean())
    return float(np.mean(con)), float(np.mean(hom))


def load_last(run: str, device: str) -> torch.nn.Module:
    """Load the final (`last.pt`) checkpoint of a run in eval mode."""
    state = torch.load(RUNS / run / "last.pt", map_location=device, weights_only=False)
    cfg = json.loads((RUNS / run / "config.json").read_text())
    from unet import UNet

    net = UNet(upsample=cfg.get("upsample", "transpose")).to(device)
    net.load_state_dict(state["model"])
    return net.eval()


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    order = np.argsort(ps)
    m = len(ps)
    out = [0.0] * m
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


def main() -> None:
    """Score all arms and seeds, then write per-slice values, summaries and paired tests."""
    from scipy.stats import wilcoxon

    quick = "--quick" in sys.argv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = PackedSplit(PACKED, "test", "noisy_full_100k", "gt_full", "ts")
    idx = list(range(len(data.files)))[: (20 if quick else None)]
    seeds = SEEDS[:1] if quick else SEEDS
    dst = OUT / ("independent_texture_quick" if quick else "independent_texture")
    dst.mkdir(parents=True, exist_ok=True)

    # reference side, once per slice
    ref, wins = {}, {}
    for i in idx:
        noisy, gt, _ = data.tensors(i, "cpu")
        g = hu(gt[0, 0].numpy().astype(np.float64))
        w = flat_windows(g)
        if not w:
            continue
        wins[i] = w
        fav, sd = nps_stats(g, w)
        con, hom = glcm_stats(g, w)
        ref[i] = {
            "fav": fav,
            "sd": sd,
            "con": con,
            "hom": hom,
            "patient": data.ids(i)[1],
            "slice": data.ids(i)[0],
        }
    print(
        f"{len(wins)}/{len(idx)} slices have at least one homogeneous window "
        f"(mean {np.mean([len(v) for v in wins.values()]):.1f} windows)"
    )

    rows = []

    def score(arm: str, seed: int, i: int, out_hu: np.ndarray) -> None:
        fav, sd = nps_stats(out_hu, wins[i])
        con, hom = glcm_stats(out_hu, wins[i])
        r = ref[i]
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "slice": r["slice"],
                "patient": r["patient"],
                "n_windows": len(wins[i]),
                "fav": fav,
                "fav_ref": r["fav"],
                "fav_err": abs(fav - r["fav"]),
                "glcm_contrast": con,
                "glcm_contrast_err": abs(np.log((con + 1e-6) / (r["con"] + 1e-6))),
                "glcm_homog": hom,
                "glcm_homog_err": abs(hom - r["hom"]),
                "noise_sd": sd,
                "noise_err": abs(np.log(sd / r["sd"])),
            }
        )

    for i in wins:  # LDCT input for context
        noisy, _, _ = data.tensors(i, "cpu")
        score("LDCT", 0, i, hu(noisy[0, 0].numpy().astype(np.float64)))
    for stem, _ in ARMS:
        for s in seeds:
            run = f"{stem}_s{s}"
            t0 = time.time()
            net = load_last(run, device)
            for i in wins:
                noisy, _, _ = data.tensors(i, device)
                with torch.no_grad():
                    out = net(noisy)[0, 0].float().cpu().numpy().astype(np.float64)
                score(stem, s, i, hu(out))
            del net
            torch.cuda.empty_cache() if device == "cuda" else None
            print(f"{run}: {len(wins)} slices, {time.time() - t0:.0f} s")

    keys = list(rows[0].keys())
    with (dst / "per_slice.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # summary: mean over slices per seed, then mean +- std over seeds
    metrics = ("fav", "fav_err", "glcm_contrast_err", "glcm_homog_err", "noise_err")
    labels = dict(ARMS)
    labels["LDCT"] = "LDCT input"
    with (dst / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "label"] + [f"{m}_{k}" for m in metrics for k in ("mean", "sd")])
        w.writerow(
            ["NDCT", "reference", np.mean([r["fav"] for r in ref.values()]), 0.0]
            + [0.0, 0.0] * (len(metrics) - 1)
        )
        for arm in ["LDCT"] + [a for a, _ in ARMS]:
            per_seed = {}
            for r in rows:
                if r["arm"] == arm:
                    per_seed.setdefault(r["seed"], []).append(r)
            vals = []
            for m in metrics:
                ms = [np.mean([r[m] for r in rr]) for rr in per_seed.values()]
                vals += [np.mean(ms), np.std(ms, ddof=1) if len(ms) > 1 else 0.0]
            w.writerow([arm, labels[arm]] + [f"{v:.5f}" for v in vals])

    # paired tests on seed-averaged per-slice values, ours vs every other arm
    def seed_avg(arm: str, m: str, level: str) -> dict:
        d = {}
        for r in rows:
            if r["arm"] == arm:
                d.setdefault(r["slice"], []).append(r)
        per_slice = {k: (np.mean([x[m] for x in v]), v[0]["patient"]) for k, v in d.items()}
        if level == "slice":
            return {k: v[0] for k, v in per_slice.items()}
        pat = {}
        for val, p in per_slice.values():
            pat.setdefault(p, []).append(val)
        return {p: float(np.mean(v)) for p, v in pat.items()}

    out = []
    for level in ("slice", "patient"):
        fam = []
        for arm, _ in ARMS:
            if arm == OURS:
                continue
            for m in PRIMARY:
                a, b = seed_avg(OURS, m, level), seed_avg(arm, m, level)
                ks = sorted(set(a) & set(b))
                dd = np.array([a[k] - b[k] for k in ks])
                p = float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0
                rk = np.argsort(np.argsort(np.abs(dd))) + 1
                rb = float((rk[dd > 0].sum() - rk[dd < 0].sum()) / rk.sum())
                fam.append(
                    {
                        "level": level,
                        "vs": arm,
                        "metric": m,
                        "n": len(ks),
                        "median_diff": float(np.median(dd)),
                        "ours": float(np.mean([a[k] for k in ks])),
                        "other": float(np.mean([b[k] for k in ks])),
                        "p": p,
                        "r": rb,
                    }
                )
        for f, h in zip(fam, holm([f["p"] for f in fam])):
            f["p_holm"] = h
        out += fam
    with (dst / "stats.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    (dst / "run_info.txt").write_text(
        f"slices with windows: {len(wins)}/{len(idx)}; windows per slice mean "
        f"{np.mean([len(v) for v in wins.values()]):.2f}\nseeds: {seeds}\ndevice: {device}\n"
        f"torch {torch.__version__}\nrule: N={N}, max {MAX_WIN} windows, structure sd <= {MAX_SD} HU, "
        f"smoothed HU in {HU_RANGE}; GLCM {LEVELS} levels over [{GLCM_LO}, {GLCM_HI}] HU, d=1,2, 4 angles\n"
        "negative median_diff / r < 0 = LBW-NPS closer to NDCT\n"
    )
    print("written", dst)
    for f in out:
        if f["level"] == "slice":
            print(
                f"{f['vs']:22s} {f['metric']:18s} ours {f['ours']:.4f} other {f['other']:.4f} "
                f"p_holm {f['p_holm']:.2g} r {f['r']:+.2f}"
            )


if __name__ == "__main__":
    main()
