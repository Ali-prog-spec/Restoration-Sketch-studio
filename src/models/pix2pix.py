"""Style-conditioned pix2pix for face -> sketch (Task 4).

Generator G(x, s): U-Net (Ronneberger 2015; pix2pix, Isola 2017) adapted to 128x128
(7 downsamplings: 128 -> 1). Discriminator D(x, y, s): 70x70 PatchGAN.

STYLE CONDITIONING — learned embedding + spatial replication/concatenation
    e_s = Embedding(3, d)[s]                    (learned, d = style_dim)
    G: e_s tiled to HxW and concatenated to the photo at the INPUT, and tiled to
       1x1 and concatenated at the BOTTLENECK (so the style also reaches the deepest,
       most global features, not only the first layer).
    D: its own embedding, tiled and concatenated to (photo, sketch) at the input.
This is the concatenation strategy of conditional GANs (Mirza & Osindero 2014) and
StarGAN-style label maps; alternatives (conditional normalisation / FiLM,
projection discriminator) are discussed in docs/research_notes.md.

Both G and D therefore have trainable style parameters that change their outputs —
verified by tests/test_models.py.

Images are in [-1, 1] (tanh output), as in pix2pix.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm(kind: str, c: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(c)
    if kind == "instance":
        return nn.InstanceNorm2d(c, affine=True)
    return nn.Identity()


class Down(nn.Module):
    def __init__(self, cin, cout, norm="instance", use_norm=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=not use_norm),
            _norm(norm, cout) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, cin, cout, norm="instance", dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False),
            _norm(norm, cout),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StyleUNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=64, num_styles=3,
                 style_dim=16, dropout=0.5, norm="instance", image_size=128):
        super().__init__()
        self.config = dict(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels,
                           num_styles=num_styles, style_dim=style_dim, dropout=dropout, norm=norm,
                           image_size=image_size)
        ngf = base_channels
        self.style_emb = nn.Embedding(num_styles, style_dim)
        # encoder channel plan for 128x128: 64,32,16,8,4,2,1
        enc = [ngf, ngf * 2, ngf * 4, ngf * 8, ngf * 8, ngf * 8, ngf * 8]
        self.downs = nn.ModuleList()
        cin = in_channels + style_dim
        for i, c in enumerate(enc):
            # no norm on the first layer (pix2pix) nor on the 1x1 innermost layer
            self.downs.append(Down(cin, c, norm, use_norm=(0 < i < len(enc) - 1)))
            cin = c
        # decoder: first up-layer receives bottleneck + style embedding
        self.ups = nn.ModuleList()
        dec = [ngf * 8, ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf]
        cin = enc[-1] + style_dim
        for j, c in enumerate(dec):
            self.ups.append(Up(cin, c, norm, dropout=dropout if j < 3 else 0.0))
            cin = c + enc[len(enc) - 2 - j]  # concat with mirrored encoder feature
        self.final = nn.Sequential(nn.ConvTranspose2d(cin, out_channels, 4, stride=2, padding=1), nn.Tanh())

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        e = self.style_emb(style)                      # (B, d)
        b, _, h, w = x.shape
        h_ = torch.cat([x, e[:, :, None, None].expand(b, e.shape[1], h, w)], dim=1)
        feats = []
        for down in self.downs:
            h_ = down(h_)
            feats.append(h_)
        h_ = torch.cat([h_, e[:, :, None, None].expand(b, e.shape[1], h_.shape[2], h_.shape[3])], dim=1)
        for j, up in enumerate(self.ups):
            h_ = up(h_)
            h_ = torch.cat([h_, feats[len(feats) - 2 - j]], dim=1)
        return self.final(h_)


class StylePatchDiscriminator(nn.Module):
    """70x70 PatchGAN (C64-C128-C256-C512, then 1-channel conv). For 128x128 inputs
    the output is a 14x14 map of real/fake logits."""

    def __init__(self, photo_channels=3, sketch_channels=1, base_channels=64, num_styles=3,
                 style_dim=16, norm="instance"):
        super().__init__()
        self.config = dict(photo_channels=photo_channels, sketch_channels=sketch_channels,
                           base_channels=base_channels, num_styles=num_styles, style_dim=style_dim, norm=norm)
        ndf = base_channels
        self.style_emb = nn.Embedding(num_styles, style_dim)
        cin = photo_channels + sketch_channels + style_dim
        self.net = nn.Sequential(
            nn.Conv2d(cin, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), _norm(norm, ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), _norm(norm, ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False), _norm(norm, ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, photo, sketch, style):
        e = self.style_emb(style)
        b, _, h, w = photo.shape
        x = torch.cat([photo, sketch, e[:, :, None, None].expand(b, e.shape[1], h, w)], dim=1)
        return self.net(x)


def init_weights(m: nn.Module) -> None:
    """pix2pix initialisation: N(0, 0.02) for conv weights, N(1, 0.02) for norm scales."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) and m.weight is not None:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 1.0)
