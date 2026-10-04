"""Tensor shapes and structural guarantees of every model."""

import pytest
import torch

from src.models import (CorruptionClassifier, DenoisingAutoencoder, SoftMoE, StylePatchDiscriminator,
                        StyleUNetGenerator, hard_route, init_weights)


def test_autoencoder_shapes_and_bottleneck():
    m = DenoisingAutoencoder(base_channels=16, latent_channels=8).eval()
    x = torch.rand(2, 3, 128, 128)
    y = m(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    z, _ = m.encode(x)
    assert z.shape == (2, 8, 8, 8) and m.bottleneck_dim == 512
    assert z.numel() / 2 < x[0].numel() / 10  # >10x compression


def test_no_skip_by_default_output_depends_only_on_latent():
    """With skip_levels=() the decoder receives nothing but z: two inputs with the same
    latent code must give the same output."""
    m = DenoisingAutoencoder(base_channels=8, latent_channels=4, dropout=0.0).eval()
    assert len(m.skip_proj) == 0
    z, feats = m.encode(torch.rand(1, 3, 128, 128))
    other_feats = [torch.randn_like(f) for f in feats]
    assert torch.allclose(m.decode(z, feats), m.decode(z, other_feats))


def test_limited_skip_option():
    m = DenoisingAutoencoder(base_channels=8, latent_channels=4, skip_levels=[0], skip_channels=2).eval()
    assert m(torch.rand(1, 3, 128, 128)).shape == (1, 3, 128, 128)
    with pytest.raises(ValueError):
        DenoisingAutoencoder(skip_levels=[3])  # no full-resolution skip allowed


def test_classifier_outputs_four_logits():
    c = CorruptionClassifier(base_channels=8)
    assert c(torch.rand(5, 3, 128, 128)).shape == (5, 4)


def _moe(tau=1.0):
    return SoftMoE(CorruptionClassifier(base_channels=8), *[DenoisingAutoencoder(base_channels=8) for _ in range(3)], tau=tau).eval()


def test_moe_weights_sum_to_one_and_shapes():
    moe = _moe(tau=0.7)
    out, w, logits = moe(torch.rand(4, 3, 128, 128))
    assert out.shape == (4, 3, 128, 128) and w.shape == (4, 4) and logits.shape == (4, 4)
    assert torch.allclose(w.sum(1), torch.ones(4), atol=1e-6) and (w >= 0).all()


def test_moe_output_is_weighted_sum():
    moe = _moe()
    x = torch.rand(2, 3, 128, 128)
    out, w, _ = moe(x)
    branches = [x] + [e(x) for e in moe.experts]
    manual = sum(w[:, k, None, None, None] * branches[k] for k in range(4))
    assert torch.allclose(out, manual, atol=1e-6)


def test_moe_clean_identity_branch():
    """If the gate is certain the input is clean, the output equals the input exactly."""
    moe = _moe()
    for p in moe.gate.fc.parameters():
        torch.nn.init.zeros_(p)
    moe.gate.fc.bias.data = torch.tensor([100.0, 0.0, 0.0, 0.0])
    x = torch.rand(2, 3, 128, 128)
    out, w, _ = moe(x)
    assert torch.allclose(w[:, 0], torch.ones(2)) and torch.allclose(out, x, atol=1e-6)


def test_temperature_controls_sharpness():
    moe = _moe()
    x = torch.rand(3, 3, 128, 128)
    moe.tau.fill_(0.2)
    _, w_sharp, _ = moe(x)
    moe.tau.fill_(5.0)
    _, w_soft, _ = moe(x)
    assert w_sharp.max(1).values.mean() >= w_soft.max(1).values.mean()


def test_hard_route_identity_and_experts():
    experts = [DenoisingAutoencoder(base_channels=8).eval() for _ in range(3)]
    x = torch.rand(4, 3, 128, 128)
    y = hard_route(x, torch.tensor([0, 1, 2, 3]), experts)
    assert torch.equal(y[0], x[0])  # clean -> identity bypass
    for k in (1, 2, 3):
        with torch.no_grad():
            assert torch.allclose(y[k], experts[k - 1](x[k : k + 1])[0], atol=1e-6)


def test_generator_and_discriminator_shapes():
    G = StyleUNetGenerator(base_channels=16, style_dim=8, out_channels=1)
    D = StylePatchDiscriminator(base_channels=16, style_dim=8, sketch_channels=1)
    G.apply(init_weights)
    D.apply(init_weights)
    x, s = torch.rand(2, 3, 128, 128) * 2 - 1, torch.tensor([0, 2])
    y = G(x, s)
    assert y.shape == (2, 1, 128, 128) and y.abs().max() <= 1
    assert D(x, y, s).shape == (2, 1, 14, 14)  # 70x70 PatchGAN on 128x128


def test_style_embedding_influences_generator_and_discriminator():
    torch.manual_seed(0)
    G = StyleUNetGenerator(base_channels=16, style_dim=8, dropout=0.0).eval()
    D = StylePatchDiscriminator(base_channels=16, style_dim=8).eval()
    G.apply(init_weights)
    D.apply(init_weights)
    x = torch.rand(1, 3, 128, 128) * 2 - 1
    with torch.no_grad():
        g0, g1 = G(x, torch.tensor([0])), G(x, torch.tensor([1]))
        y = torch.rand(1, 1, 128, 128) * 2 - 1
        d0, d2 = D(x, y, torch.tensor([0])), D(x, y, torch.tensor([2]))
    assert (g0 - g1).abs().mean() > 1e-4
    assert (d0 - d2).abs().mean() > 1e-4
    # the embeddings are trainable parameters of BOTH networks
    assert G.style_emb.weight.requires_grad and D.style_emb.weight.requires_grad
