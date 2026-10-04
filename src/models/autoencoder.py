"""Convolutional denoising autoencoder (Task 1 universal model and Task 2 specialists).

    corrupted x~ (3x128x128)
      -> encoder: 4 stages, each halves H,W and increases channels   (128 -> 8)
      -> bottleneck: 1x1 conv to `latent_channels` at 8x8            (the compressed code z)
      -> decoder: 4 stages of bilinear upsample + conv                (8 -> 128)
      -> 1x1-conv head + sigmoid -> x^ (3x128x128)

Bottleneck dimension = latent_channels * 8 * 8. With latent_channels = 16 the code
has 1,024 values for a 49,152-value input (48x compression).

Skip connections are OFF by default (``skip_levels=()``), so all information must
pass through z. ``skip_levels`` can enable *limited* skips for the ablation in the
report: each enabled skip carries only ``skip_channels`` (default 4) channels through
a 1x1 conv, and only at the resolutions listed:
    level 0 -> 16x16, level 1 -> 32x32, level 2 -> 64x64.
There is intentionally no skip at 128x128 and no global input->output residual,
since those would let the network copy the (corrupted) input and bypass z.

Upsampling uses bilinear interpolation followed by a convolution, rather than
transposed convolution, to avoid checkerboard artefacts
(Odena et al., Distill 2016, "Deconvolution and Checkerboard Artifacts").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    )


class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        channel_mults=(1, 2, 4, 8),
        latent_channels: int = 16,
        dropout: float = 0.1,
        skip_levels=(),
        skip_channels: int = 4,
        image_size: int = 128,
    ):
        super().__init__()
        chs = [base_channels * m for m in channel_mults]
        self.config = dict(
            in_channels=in_channels, base_channels=base_channels, channel_mults=list(channel_mults),
            latent_channels=latent_channels, dropout=dropout, skip_levels=list(skip_levels),
            skip_channels=skip_channels, image_size=image_size,
        )
        n = len(chs)
        self.latent_hw = image_size // (2**n)
        self.bottleneck_dim = latent_channels * self.latent_hw**2

        # ----- encoder: stride-2 conv (downsample) + conv -----
        self.enc = nn.ModuleList()
        cin = in_channels
        for c in chs:
            self.enc.append(nn.Sequential(conv_block(cin, c, stride=2), conv_block(c, c)))
            cin = c

        # ----- bottleneck -----
        self.to_latent = nn.Sequential(
            nn.Conv2d(chs[-1], latent_channels, 1),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.latent_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.from_latent = conv_block(latent_channels, chs[-1])

        # ----- decoder -----
        # decoder stage j upsamples to resolution latent_hw * 2^(j+1)
        # and may receive a skip from encoder stage (n-2-j) at that resolution.
        self.skip_levels = set(int(s) for s in skip_levels)
        if any(s < 0 or s > n - 2 for s in self.skip_levels):
            raise ValueError(f"skip_levels must be in [0, {n - 2}]")
        self.skip_proj = nn.ModuleDict()
        self.dec = nn.ModuleList()
        dec_out = list(reversed(chs[:-1])) + [chs[0]]
        cin = chs[-1]
        for j, cout in enumerate(dec_out):
            extra = 0
            if j in self.skip_levels:
                enc_c = chs[n - 2 - j]
                self.skip_proj[str(j)] = nn.Conv2d(enc_c, skip_channels, 1)
                extra = skip_channels
            self.dec.append(nn.Sequential(conv_block(cin + extra, cout), conv_block(cout, cout)))
            cin = cout
        self.dec_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(chs[0], in_channels, 1)

    def encode(self, x: torch.Tensor):
        feats = []
        h = x
        for stage in self.enc:
            h = stage(h)
            feats.append(h)
        z = self.latent_dropout(self.to_latent(h))
        return z, feats

    def decode(self, z: torch.Tensor, feats=None) -> torch.Tensor:
        h = self.from_latent(z)
        n = len(self.enc)
        for j, stage in enumerate(self.dec):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            if j in self.skip_levels:
                h = torch.cat([h, self.skip_proj[str(j)](feats[n - 2 - j])], dim=1)
            h = stage(h)
            if j == 0:
                h = self.dec_dropout(h)
        return torch.sigmoid(self.head(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, feats = self.encode(x)
        return self.decode(z, feats if self.skip_levels else None)


def build_autoencoder(cfg: dict) -> DenoisingAutoencoder:
    return DenoisingAutoencoder(**cfg)
