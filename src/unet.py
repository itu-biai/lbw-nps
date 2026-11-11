"""Fixed U-Net backbone. Identical for every loss in the study."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by batch normalization and LeakyReLU."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to `x`."""
        return self.body(x)


def make_upsample(cin: int, cout: int, kind: str) -> nn.Module:
    """Return one decoder upsampling stage.

    `transpose` (a 2x2 stride-2 transposed convolution) is the default and is
    used by every reported run. `resize` (bilinear upsampling followed by a 3x3
    convolution, Odena et al., 2016) is an optional checkerboard-free
    alternative; its parameter count differs slightly, so it is not a drop-in
    replacement for the reported models.
    """
    if kind == "transpose":
        return nn.ConvTranspose2d(cin, cout, 2, stride=2)
    if kind == "resize":
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(cin, cout, 3, padding=1),
        )
    raise ValueError(f"unknown upsample kind: {kind!r}")


class UNet(nn.Module):
    """~7.7M parameter U-Net that predicts the residual noise.

    Residual (`out = x - f(x)`) rather than direct prediction: the denoiser only
    has to model the noise, which is what every LDCT baseline since RED-CNN
    does, and it keeps the low frequency content of the input intact so the NPS
    terms are not fighting a DC drift.
    """

    def __init__(
        self, base: int = 32, depth: int = 4, in_ch: int = 1, upsample: str = "transpose"
    ) -> None:
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        prev = in_ch
        for c in chans[:-1]:
            self.down.append(ConvBlock(prev, c))
            prev = c
        self.bottleneck = ConvBlock(prev, chans[-1])
        self.up_convs = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up_samples.append(make_upsample(chans[i + 1], chans[i], upsample))
            self.up_convs.append(ConvBlock(chans[i] * 2, chans[i]))
        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the denoised image `x - f(x)` for an input of shape `[B, 1, H, W]`."""
        skips = []
        h = x
        for block in self.down:
            h = block(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for up, conv, skip in zip(self.up_samples, self.up_convs, reversed(skips)):
            h = up(h)
            h = conv(torch.cat([h, skip], dim=1))
        return x - self.head(h)


def count_params(model: nn.Module) -> int:
    """Return the number of parameters of `model`."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import time

    net = UNet().cuda()
    print(f"params: {count_params(net) / 1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    for shape in [(2, 1, 512, 512), (4, 1, 256, 256), (16, 1, 128, 128), (8, 1, 256, 256)]:
        try:
            x = torch.rand(*shape, device="cuda")
            for _ in range(3):
                opt.zero_grad()
                loss = (net(x) - x).abs().mean()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(5):
                opt.zero_grad()
                loss = (net(x) - x).abs().mean()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t) / 5
            px = shape[0] * shape[2] * shape[3]
            mem = torch.cuda.max_memory_allocated() / 2**20
            print(
                f"{shape!s:22s} {dt * 1000:7.1f} ms/iter  {px / dt / 1e6:6.2f} Mpx/s  peak {mem:.0f} MiB"
            )
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as exc:
            print(f"{shape!s:22s} FAILED: {str(exc)[:80]}")
            torch.cuda.empty_cache()
