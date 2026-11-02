#!/usr/bin/env python3
"""Score any checkpoint with the ISBI metric code and write per-slice rows.

Every Table 1 row goes through this script, so no row is scored by a
different implementation.

    python scripts/evaluate_runs.py --runs l1_s42 pooled_weighted_s42 --split test
    python scripts/evaluate_runs.py --runs input --split test     # the LDCT row
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from config import OUT, PACKED, RUNS, PROTOCOL  # noqa: E402
from isbi_data import PackedSplit  # noqa: E402
from metrics import LpipsAlex, slice_metrics, valid_map  # noqa: E402
from unet import UNet  # noqa: E402

_BASE = ["configuration", "seed", "slice_id", "patient_id", "psnr", "ssim", "lpips",
         "r_h", "r_h_error"]
_DUAL = [f"{k}_w{w}" for w in (64, 32)
         for k in ("gloc", "patches", "gloc_lung", "patches_lung", "gloc_soft",
                   "patches_soft", "gloc_bone", "patches_bone")]
_ALIAS = ["local_nps_gap", "patches_valid", "local_nps_gap_lung", "patches_lung",
          "local_nps_gap_soft", "patches_soft", "local_nps_gap_bone", "patches_bone"]
FIELDS = _BASE + _DUAL + _ALIAS


def load_net(run_dir: Path, device: str) -> tuple[torch.nn.Module, dict]:
    cfg = json.loads((run_dir / "config.json").read_text())
    state = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
    net = UNet(upsample=cfg.get("upsample", "transpose")).to(device)
    net.load_state_dict(state["model"])
    return net.eval(), cfg


@torch.no_grad()
def score(run: str, split: str, device: str, lpips_fn, limit: int | None) -> list[dict]:
    if run == "input":
        cfg = {"domain": PROTOCOL.domain, "noise": PROTOCOL.noise,
               "mask_tag": PROTOCOL.mask_tag, "seed": 0}
        net = None
    else:
        net, cfg = load_net(RUNS / run, device)
    ikey = (f"noisy_full_{cfg['noise']}" if cfg["domain"] == "full"
            else f"noisy_clipped_{cfg['noise']}")
    tkey = "gt_full" if cfg["domain"] == "full" else "gt_clipped"
    data = PackedSplit(PACKED, split, ikey, tkey, cfg.get("mask_tag", "ts"))
    n = len(data) if limit is None else min(limit, len(data))
    rows = []
    for i in range(n):
        noisy, gt, masks = data.tensors(i, device)
        pred = noisy if net is None else net(noisy)
        row = slice_metrics(pred, gt, valid_map(gt, masks.amax(dim=1, keepdim=True)),
                            masks, lpips_fn=lpips_fn,
                            min_valid=PROTOCOL.min_valid, min_tissue=PROTOCOL.min_tissue)
        stem, patient = data.ids(i)
        rows.append({"configuration": run, "seed": cfg.get("seed", 0),
                     "slice_id": stem, "patient_id": patient, **row})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = LpipsAlex(device)
    out = a.out or OUT / "metrics" / f"per_slice_{a.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    rows: list[dict] = []
    if out.exists():
        with out.open() as f:
            rows = list(csv.DictReader(f))
        done = {r["configuration"] for r in rows}

    for run in a.runs:
        if run in done:
            print(f"[atlandi] {run} zaten var", flush=True)
            continue
        t0 = time.time()
        new = score(run, a.split, device, lpips_fn, a.limit)
        rows += new
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[bitti] {run}: {len(new)} kesit, {(time.time() - t0) / 60:.1f} dk -> {out}",
              flush=True)


if __name__ == "__main__":
    main()
