"""One training entry point for every variant in the study."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PackedSlices, PatchDataset, SliceDataset
from evaluate import Evaluator, aggregate
from isbi_data import assert_no_patient_leakage
from losses import (
    FocalFrequencyLoss,
    GlobalNPSLoss,
    PearsonDivergenceLoss,
    TCBWNPSLoss,
    VGGPerceptualLoss,
)
from nps import CLASS_NAMES, DEFAULT_BAND_CUTOFFS
from unet import UNet, count_params


@dataclass
class Config:
    """Training configuration; every field can be set from a JSON file or `--set key=value`."""

    run_name: str
    packed_root: str = "data/packed"
    out_root: str = "outputs"
    domain: str = "full"  # full -> (noisy_full_100k, gt_full); clipped -> display-window pair
    noise: str = "100k"
    mask_tag: str = "ts"
    variant: str = "l1"
    lambda_aux: float = 0.0
    tissue_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    band_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
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
    field: str = "image"  # spectra of the image (all reported runs)
    warmup_epochs: int = 0  # epochs of pure L1 before the aux term is switched on
    detrend: int = 0  # 0 = mean subtraction, 2 = AAPM TG-233 quadratic detrend
    # The NPS terms compare log spectra. That is a modelling choice: set
    # `log_domain=False` for the relative-linear comparator, which is the
    # control that says whether the log form is doing any work.
    log_domain: bool = True
    upsample: str = "transpose"  # decoder upsampling used by all reported runs
    # LBW-NPS: square Hann-tapered windows on a fixed grid, one loss term per
    # window, no tissue labels. Stride is kept at patch/2 in every experiment.
    lbw_patch: int = 64
    lbw_stride: int = 32
    lbw_min_valid: float = 0.9

    @property
    def input_key(self) -> str:
        """Name of the low-dose input array in the packed data."""
        return (
            f"noisy_full_{self.noise}" if self.domain == "full" else f"noisy_clipped_{self.noise}"
        )

    @property
    def target_key(self) -> str:
        """Name of the normal-dose reference array in the packed data."""
        return "gt_full" if self.domain == "full" else "gt_clipped"


def build_aux_loss(cfg: Config, device: str, n_classes: int = 3) -> tuple[nn.Module | None, str]:
    """Return the auxiliary (non L1) loss term and the signature it expects."""
    weights = cfg.tissue_weights if len(cfg.tissue_weights) == n_classes else [1.0] * n_classes
    tw = torch.tensor(weights, dtype=torch.float32)
    bw = torch.tensor(cfg.band_weights, dtype=torch.float32)
    w = tw[:, None] * bw[None, :]
    v = cfg.variant
    if v == "l1":
        return None, "none"
    if v == "nps_global_uniform":  # pooled NPS, uniform bands
        return GlobalNPSLoss(
            cfg.roi_size,
            band_weighted=False,
            sampling="body",
            n_rois=8,
            cutoffs=cfg.cutoffs,
            field=cfg.field,
            detrend=cfg.detrend,
            log_domain=cfg.log_domain,
        ).to(device), "masked"
    if v == "nps_global_band":  # pooled NPS, weighted bands
        return GlobalNPSLoss(
            cfg.roi_size,
            band_weighted=True,
            band_weights=bw,
            sampling="body",
            n_rois=8,
            cutoffs=cfg.cutoffs,
            detrend=cfg.detrend,
            log_domain=cfg.log_domain,
        ).to(device), "masked"
    if v == "nps_tcbw":  # tissue-conditioned ROI pooling
        return TCBWNPSLoss(
            cfg.roi_size,
            n_classes,
            weights=w,
            band_weighted=True,
            mode="roi",
            n_rois=cfg.n_rois,
            cutoffs=cfg.cutoffs,
            min_coverage=cfg.min_coverage,
            field=cfg.field,
            reduction="band",
            detrend=cfg.detrend,
            log_domain=cfg.log_domain,
        ).to(device), "masked"
    if v == "lbw_nps":  # LBW-NPS: local band powers with a reference validity mask
        from lbw_nps import LBWNPSTrainingLoss

        return LBWNPSTrainingLoss(
            patch_size=cfg.lbw_patch,
            stride=cfg.lbw_stride,
            band_weights=tuple(cfg.band_weights),
            min_valid=cfg.lbw_min_valid,
        ).to(device), "masked"
    if v == "ffl":
        return FocalFrequencyLoss().to(device), "pair"
    if v == "perceptual":
        return VGGPerceptualLoss().to(device), "pair"
    if v == "pearson":
        return PearsonDivergenceLoss().to(device), "pair"
    raise ValueError(f"unknown variant {v}")


def run(cfg: Config) -> dict:
    """Train one model; save `best.pt` (lowest validation L1), `last.pt` and logs.

    Returns:
        A summary with the best validation epoch, run time and parameter count.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.out_root) / "runs" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Refuse to train if any patient appears in more than one split.
    patients_per_split = assert_no_patient_leakage(Path(cfg.packed_root))
    print(f"patient-disjoint splits: {patients_per_split}", flush=True)

    tr = PackedSlices(cfg.packed_root, "train", cfg.input_key, cfg.target_key, cfg.mask_tag)
    va = PackedSlices(cfg.packed_root, "val", cfg.input_key, cfg.target_key, cfg.mask_tag)
    train_ds = PatchDataset(tr, cfg.patch, cfg.iters_per_epoch * cfg.batch_size, seed=cfg.seed)
    val_ds = SliceDataset(va, limit=cfg.val_slices)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=1)

    net = UNet(upsample=cfg.upsample).to(device)
    aux, sig = build_aux_loss(cfg, device, n_classes=tr.n_classes)
    l1 = nn.L1Loss()
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    from nps import CLASS_SETS

    # The validation NPS ratios must be the same measurement for every loss, so
    # the evaluator always uses the standard cutoffs; `cfg.cutoffs` only
    # configures the loss.
    evaluator = Evaluator(
        DEFAULT_BAND_CUTOFFS,
        device=device,
        use_lpips=True,
        class_names=CLASS_SETS.get(cfg.mask_tag, CLASS_NAMES),
    )
    history: list[dict] = []
    best = {"val_l1": float("inf"), "epoch": -1}
    t0 = time.time()

    for epoch in range(cfg.epochs):
        net.train()
        train_ds.set_epoch(epoch)
        # Auxiliary terms are switched on after an L1-only warm-up
        # (`warmup_epochs`, 10 in all reported runs), as in CT-Mamba.
        lam_now = 0.0 if epoch < cfg.warmup_epochs else cfg.lambda_aux
        sums = {"l1": 0.0, "aux": 0.0, "total": 0.0}
        for batch in tqdm(train_dl, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False):
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
            loss.backward()
            opt.step()
            sums["l1"] += loss_l1.item()
            sums["aux"] += float(loss_aux)
            sums["total"] += loss.item()
        sched.step()
        n = max(1, len(train_dl))
        rec = {
            "epoch": epoch,
            "lr": sched.get_last_lr()[0],
            "lambda": lam_now,
            **{f"train_{k}": v / n for k, v in sums.items()},
        }

        if (epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1:
            rec.update(validate(net, val_dl, evaluator, device, cfg))
            if rec["val_l1"] < best["val_l1"]:
                best = {"val_l1": rec["val_l1"], "epoch": epoch}
                torch.save(
                    {"model": net.state_dict(), "epoch": epoch, "config": asdict(cfg)},
                    out_dir / "best.pt",
                )
        history.append(rec)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        msg = f"[{cfg.run_name}] ep {epoch + 1}/{cfg.epochs} l1 {rec['train_l1']:.5f} aux {rec['train_aux']:.4f}"
        if "val_l1" in rec:
            msg += f" | val l1 {rec['val_l1']:.5f} psnr {rec.get('val_psnr', float('nan')):.2f}"
        print(msg, flush=True)

    torch.save(
        {"model": net.state_dict(), "epoch": cfg.epochs - 1, "config": asdict(cfg)},
        out_dir / "last.pt",
    )
    summary = {
        "run": cfg.run_name,
        "best": best,
        "minutes": (time.time() - t0) / 60,
        "params_m": count_params(net) / 1e6,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


@torch.no_grad()
def validate(
    net: nn.Module, val_dl: DataLoader, evaluator: Evaluator, device: str, cfg: Config
) -> dict[str, float]:
    """Return validation L1 and the aggregated validation metrics."""
    net.eval()
    l1_vals, records = [], []
    for batch in val_dl:
        noisy = batch["noisy"].to(device)
        gt = batch["gt"].to(device)
        pred = net(noisy).clamp(0, 1)
        l1_vals.append(float((pred - gt).abs().mean()))
        records.append(
            evaluator.slice_metrics(
                pred[0, 0].cpu().numpy().astype(np.float64),
                gt[0, 0].cpu().numpy().astype(np.float64),
                batch["masks"][0].numpy(),
                batch["rois"][0].numpy(),
                batch["nrois"][0].numpy(),
                windowed_psnr=cfg.domain == "full",
            )
        )
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
    """Parse the command line and run training."""
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
