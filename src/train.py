#!/usr/bin/env python3
"""One training entry point for every variant in the study."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import PackedSlices, PatchDataset, SliceDataset
from evaluate import Evaluator, aggregate
from losses import (
    FocalFrequencyLoss,
    PearsonDivergenceLoss,
    GlobalNPSLoss,
    MetaWeightedTCBWNPS,
    TCBWNPSLoss,
    VGGPerceptualLoss,
)
from nps import CLASS_NAMES, DEFAULT_BAND_CUTOFFS
from unet import UNet, count_params


@dataclass
class Config:
    run_name: str
    packed_root: str = "data/packed"
    out_root: str = "outputs"
    domain: str = "full"  # full -> (noisy_full_100k, gt_full); clipped -> distributed pair
    noise: str = "100k"
    mask_tag: str = "ts"
    variant: str = "l1"
    lambda_aux: float = 0.0
    tissue_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    band_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # The plan's loss is sum_r sum_b w_rb (...)^2 with w_rb free. Building w as
    # the outer product of `tissue_weights` and `band_weights` makes it rank-1 by
    # construction, so no tissue can have its own band profile - and the learned
    # weights show they want different ones (soft flat, lung and bone rising).
    # Setting `weight_matrix` bypasses the outer product and trains the free form.
    weight_matrix: list[list[float]] | None = None
    roi_size: int = 32
    n_rois: int = 4
    min_coverage: float = 0.85
    cutoffs: tuple[float, float] = DEFAULT_BAND_CUTOFFS
    seed: int = 42
    patch: int = 128
    batch_size: int = 16
    iters_per_epoch: int = 150
    epochs: int = 40
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_every: int = 4
    val_slices: int = 60
    num_workers: int = 3
    meta_lr: float = 0.05
    field: str = "image"  # "image" or "residual" (CT-Mamba style noise spectrum)
    warmup_epochs: int = 0  # epochs of pure L1 before the aux term is switched on
    detrend: int = 0  # 0 = mean subtraction (viz_toolbox), 2 = TG 233 quadratic detrend
    # The NPS terms compare log spectra. That is a modelling choice: set
    # `log_domain=False` for the relative-linear comparator, which is the
    # control that says whether the log form is doing any work.
    log_domain: bool = True
    # >0 weights each tissue ROI by 1/sigma^gamma of the clean patch, i.e. the
    # flatter (lower contrast) the region, the more it counts.
    lowcontrast_gamma: float = 0.0
    # Weight each tissue ROI by its coverage instead of admitting it outright.
    # With a soft mask tag (`tsoft`) that coverage is a mean tissue probability,
    # which is what makes the conditioning continuous across a class boundary.
    soft_weight: bool = False
    # Decoder upsampling. "transpose" is the original and the default, so every
    # earlier run reproduces exactly. "resize" is the checkerboard-free
    # alternative (see `unet.make_upsample`), run as its own arm because the
    # NPS term was found to inflate the Nyquist grid by up to 40x.
    upsample: str = "transpose"
    # LBW-NPS (ISBI plan): square tapered windows on a fixed grid, one loss term
    # per window, no tissue labels. `lbw_patch`/`lbw_stride` are the locality
    # axis of that paper; stride is kept at patch/2 across the ablation.
    lbw_patch: int = 64
    lbw_stride: int = 32
    lbw_min_valid: float = 0.9

    @property
    def input_key(self) -> str:
        return f"noisy_full_{self.noise}" if self.domain == "full" else f"noisy_clipped_{self.noise}"

    @property
    def target_key(self) -> str:
        return "gt_full" if self.domain == "full" else "gt_clipped"


def build_aux_loss(cfg: Config, device: str, n_classes: int = 3) -> tuple[nn.Module | None, str]:
    """Return the auxiliary (non L1) loss term and the signature it expects."""
    weights = cfg.tissue_weights
    if len(weights) != n_classes:
        # A partition with a different class count (e.g. the 4-class
        # fat/muscle/organ/bone arm) reuses a uniform tissue allocation unless
        # one of matching length was given explicitly.
        weights = [1.0] * n_classes
    tw = torch.tensor(weights, dtype=torch.float32)
    bw = torch.tensor(cfg.band_weights, dtype=torch.float32)
    if cfg.weight_matrix is not None:
        w = torch.tensor(cfg.weight_matrix, dtype=torch.float32)
        if w.shape != (len(weights), len(cfg.band_weights)):
            raise ValueError(f"weight_matrix {tuple(w.shape)} != "
                             f"({len(weights)}, {len(cfg.band_weights)})")
    else:
        w = tw[:, None] * bw[None, :]
    v = cfg.variant
    if v == "l1":
        return None, "none"
    if v == "nps_curve":  # A0, literature global NPS loss (CT-Mamba style)
        return GlobalNPSLoss(64, sampling="body", n_rois=8, reduction="curve",
                             detrend=cfg.detrend, log_domain=cfg.log_domain).to(device), "masked"
    if v == "nps_global_uniform":  # A
        return GlobalNPSLoss(cfg.roi_size, band_weighted=False, sampling="body", n_rois=8,
                             cutoffs=cfg.cutoffs, field=cfg.field, detrend=cfg.detrend,
                             log_domain=cfg.log_domain).to(device), "masked"
    if v == "nps_global_band":  # B
        return GlobalNPSLoss(cfg.roi_size, band_weighted=True, band_weights=bw, sampling="body",
                             n_rois=8, cutoffs=cfg.cutoffs, detrend=cfg.detrend,
                             log_domain=cfg.log_domain).to(device), "masked"
    if v.startswith("nps_tissue") or v.startswith("nps_tcbw"):  # C, D and the meta variants
        band_weighted = v != "nps_tissue_uniform"
        mode = "masked" if v == "nps_tcbw_masked" else "roi"
        field = "residual" if v.endswith("_residual") else cfg.field
        reduction = "curve" if v.endswith("_curve") else "band"
        inv = v.endswith("_invscale")
        base = TCBWNPSLoss(cfg.roi_size, n_classes, weights=w, band_weighted=band_weighted, mode=mode,
                           n_rois=cfg.n_rois, cutoffs=cfg.cutoffs,
                           min_coverage=cfg.min_coverage, field=field,
                           reduction=reduction, detrend=cfg.detrend,
                           inverse_scale=inv, soft_weight=cfg.soft_weight,
                           log_domain=cfg.log_domain,
                           lowcontrast_gamma=cfg.lowcontrast_gamma).to(device)
        if v.startswith("nps_tcbw_meta"):
            # A trailing `_curve` selects the whole-curve reduction above and is
            # not part of the scheme name, so strip it before the lookup.
            base_v = v[:-len("_curve")] if v.endswith("_curve") else v
            scheme = {"nps_tcbw_meta_unc": "uncertainty", "nps_tcbw_meta_grad": "gradnorm",
                      "nps_tcbw_meta_hard": "hard"}[base_v]
            return MetaWeightedTCBWNPS(base, scheme=scheme).to(device), "masked"
        return base, "masked"
    if v.startswith("lbw"):  # Local image-band loss with a reference validity mask
        from lbw_nps import LBWNPSTrainingLoss

        weights = (1.0, 1.0, 1.0) if v == "lbw_uniform" else tuple(cfg.band_weights)
        return LBWNPSTrainingLoss(patch_size=cfg.lbw_patch, stride=cfg.lbw_stride,
                                  band_weights=weights,
                                  min_valid=cfg.lbw_min_valid).to(device), "masked"
    if v == "ffl":
        return FocalFrequencyLoss().to(device), "pair"
    if v == "perceptual":
        return VGGPerceptualLoss().to(device), "pair"
    if v == "pearson":
        return PearsonDivergenceLoss().to(device), "pair"
    raise ValueError(f"unknown variant {v}")


def run(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.out_root) / "runs" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    tr = PackedSlices(cfg.packed_root, "train", cfg.input_key, cfg.target_key, cfg.mask_tag)
    va = PackedSlices(cfg.packed_root, "val", cfg.input_key, cfg.target_key, cfg.mask_tag)
    train_ds = PatchDataset(tr, cfg.patch, cfg.iters_per_epoch * cfg.batch_size, seed=cfg.seed)
    val_ds = SliceDataset(va, limit=cfg.val_slices)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=1)

    net = UNet(upsample=cfg.upsample).to(device)
    aux, sig = build_aux_loss(cfg, device, n_classes=tr.n_classes)
    l1 = nn.L1Loss()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    meta_opt = None
    is_meta = isinstance(aux, MetaWeightedTCBWNPS)
    if is_meta and aux.scheme in ("uncertainty", "hard"):
        meta_opt = torch.optim.Adam([aux.raw], lr=cfg.meta_lr)

    from nps import CLASS_SETS

    # The reported low/mid/high NPS ratios must be the same measurement for every
    # arm, so the evaluator is pinned to the standard cutoffs. `cfg.cutoffs` is a
    # property of the *loss* - an arm may train on five bands - and letting it
    # reach the evaluator both breaks the 3-band metric and makes arms with
    # different band counts incomparable.
    evaluator = Evaluator(DEFAULT_BAND_CUTOFFS, device=device, use_lpips=True,
                          class_names=CLASS_SETS.get(cfg.mask_tag, CLASS_NAMES))
    history: list[dict] = []
    best = {"val_l1": float("inf"), "epoch": -1}
    t0 = time.time()

    for epoch in range(cfg.epochs):
        net.train()
        train_ds.set_epoch(epoch)
        # CT-Mamba (Li et al., CMIG 2025, Sec. 3.2) holds its NPS term back for
        # the first 10 epochs "to balance the numerical relationship between the
        # Deep NPS loss and the L1 loss and ensure training stability". Our own
        # sweep shows the fidelity cost of the NPS term is a one-off 1.5 dB paid
        # at lambda = 0.005 and essentially unchanged at 0.02, which is the
        # signature of an early-training transient rather than a lambda effect.
        lam_now = 0.0 if epoch < cfg.warmup_epochs else cfg.lambda_aux
        sums = {"l1": 0.0, "aux": 0.0, "total": 0.0}
        for batch in train_dl:
            noisy = batch["noisy"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            pred = net(noisy)
            loss_l1 = l1(pred, gt)
            loss_aux = torch.zeros((), device=device)
            if aux is not None:
                loss_aux = aux(pred, gt, masks, noisy=noisy) if sig == "masked" else aux(pred, gt)
            loss = loss_l1 + lam_now * loss_aux
            opt.zero_grad(set_to_none=True)
            if meta_opt is not None:
                meta_opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if meta_opt is not None:
                if aux.scheme == "hard":
                    # Ascent: mass moves to whichever cell is still mismatched.
                    aux.raw.grad = -aux.raw.grad
                meta_opt.step()
            sums["l1"] += loss_l1.item()
            sums["aux"] += float(loss_aux)
            sums["total"] += loss.item()
        sched.step()
        n = max(1, len(train_dl))
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "lambda": lam_now,
               **{f"train_{k}": v / n for k, v in sums.items()}}

        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            rec.update(validate(net, val_dl, evaluator, device, cfg))
            if rec["val_l1"] < best["val_l1"]:
                best = {"val_l1": rec["val_l1"], "epoch": epoch}
                torch.save({"model": net.state_dict(), "epoch": epoch, "config": asdict(cfg)},
                           out_dir / "best.pt")
        if is_meta:
            rec["meta_weights"] = aux.current_weights().cpu().tolist()
        history.append(rec)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        msg = f"[{cfg.run_name}] ep {epoch + 1}/{cfg.epochs} l1 {rec['train_l1']:.5f} aux {rec['train_aux']:.4f}"
        if "val_l1" in rec:
            msg += f" | val l1 {rec['val_l1']:.5f} psnr {rec.get('val_psnr', float('nan')):.2f}"
        print(msg, flush=True)

    torch.save({"model": net.state_dict(), "epoch": cfg.epochs - 1, "config": asdict(cfg)},
               out_dir / "last.pt")
    summary = {"run": cfg.run_name, "best": best, "minutes": (time.time() - t0) / 60,
               "params_m": count_params(net) / 1e6}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


@torch.no_grad()
def validate(net, val_dl, evaluator: Evaluator, device: str, cfg: Config) -> dict:
    net.eval()
    l1_vals, records = [], []
    for batch in val_dl:
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)
        pred = net(noisy).clamp(0, 1)
        l1_vals.append(float((pred - gt).abs().mean()))
        records.append(evaluator.slice_metrics(
            pred[0, 0].cpu().numpy().astype(np.float64),
            gt[0, 0].cpu().numpy().astype(np.float64),
            batch["masks"][0].numpy(),
            batch["rois"][0].numpy(),
            batch["nrois"][0].numpy(),
            windowed_psnr=cfg.domain == "full",
        ))
    agg = aggregate(records)
    out = {"val_l1": float(np.mean(l1_vals))}
    keys = ["psnr", "ssim", "lpips", "nps_gap", "nps_gap_log", "dts_v2", "psnr_window"]
    keys += [f"nps_gap_log_{c}" for c in evaluator.class_names]
    for k in keys:
        if k in agg:
            out[f"val_{k}"] = agg[k]["mean"]
    # Band ratios are aggregated geometrically. Their per slice distribution is
    # heavy tailed (median 0.386 vs mean 2.77 on the test split) because the
    # NDCT high band integral spans two decades, so an arithmetic mean reports
    # whichever slice had the smoothest reference.
    for band in ("low", "mid", "high"):
        key = f"nps_ratio_log_{band}"
        if key in agg:
            out[f"val_nps_ratio_{band}"] = float(np.exp(agg[key]["mean"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, help="json file with Config fields")
    p.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = p.parse_args()
    payload = json.loads(args.config.read_text()) if args.config else {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        payload[k] = json.loads(v)
    run(Config(**payload))


if __name__ == "__main__":
    main()
