import pytest
import torch
import torch.nn.functional as F

from grokking.config import Config
from grokking.model import (
    LayerNorm,
    SelfAttention,
    activation,
    build_model,
    n_parameters,
)


def test_layernorm_matches_torch():
    mine = LayerNorm(32)
    torch.nn.init.normal_(mine.weight)
    torch.nn.init.normal_(mine.bias)
    x = torch.randn(7, 3, 32)
    reference = F.layer_norm(x, (32,), mine.weight, mine.bias, eps=mine.eps)
    assert torch.allclose(mine(x), reference, atol=1e-6)


def test_attention_matches_torch_reference():
    """The hand-written attention must equal torch's fused implementation."""
    torch.manual_seed(0)
    d_model, n_heads, d_head = 64, 4, 16
    attention = SelfAttention(d_model, n_heads)
    x = torch.randn(5, 3, d_model)

    batch, seq, _ = x.shape
    q, k, v = (
        w(x).view(batch, seq, n_heads, d_head).transpose(1, 2)
        for w in (attention.W_Q, attention.W_K, attention.W_V)
    )
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    reference = attention.W_O(reference.transpose(1, 2).reshape(batch, seq, d_model))
    assert torch.allclose(attention(x), reference, atol=1e-5)


def test_attention_is_causal():
    """Changing a later token must not change an earlier position's output."""
    torch.manual_seed(0)
    attention = SelfAttention(32, 2)
    x = torch.randn(1, 3, 32)
    changed = x.clone()
    changed[0, 2] = torch.randn(32)

    before, after = attention(x), attention(changed)
    assert torch.allclose(before[0, :2], after[0, :2], atol=1e-6)
    assert not torch.allclose(before[0, 2], after[0, 2], atol=1e-6)


def test_attention_pattern_is_lower_triangular():
    attention = SelfAttention(32, 2)
    cache: dict[str, torch.Tensor] = {}
    attention(torch.randn(4, 3, 32), cache)
    pattern = cache["pattern"]
    assert pattern.shape == (4, 2, 3, 3)
    assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)))
    upper = torch.triu(torch.ones(3, 3), diagonal=1).bool()
    assert pattern[..., upper].abs().max() == 0


@pytest.mark.parametrize("arch", ["transformer", "mlp"])
def test_forward_shape_and_cache(arch):
    cfg = Config(arch=arch, p=23, d_model=32, d_mlp=64, n_heads=4)
    model = build_model(cfg, cfg.p + 1, cfg.p)
    tokens = torch.stack(
        [torch.arange(6) % cfg.p, torch.arange(6) % cfg.p, torch.full((6,), cfg.p)], dim=1
    )
    cache: dict[str, torch.Tensor] = {}
    logits = model(tokens, cache)
    assert logits.shape == (6, cfg.p)
    assert cache, "passing a cache must record activations"
    assert any("neuron_post" in key for key in cache)


@pytest.mark.parametrize("arch", ["transformer", "mlp"])
def test_every_parameter_receives_gradient(arch):
    cfg = Config(arch=arch, p=23, d_model=32, d_mlp=64, n_heads=4)
    model = build_model(cfg, cfg.p + 1, cfg.p)
    tokens = torch.stack(
        [torch.arange(8) % cfg.p, torch.arange(8) % cfg.p, torch.full((8,), cfg.p)], dim=1
    )
    loss = F.cross_entropy(model(tokens), torch.arange(8) % cfg.p)
    loss.backward()
    dead = [
        name
        for name, param in model.named_parameters()
        if param.grad is None or param.grad.abs().sum() == 0
    ]
    # Only the unused rows of the embedding may legitimately have no gradient.
    assert dead == [], f"no gradient reached {dead}"


def test_build_model_is_deterministic_in_the_seed():
    cfg = Config(p=23, d_model=32, d_mlp=64)
    first = build_model(cfg, 24, 23)
    same = build_model(cfg, 24, 23)
    other = build_model(Config(p=23, d_model=32, d_mlp=64, seed=1), 24, 23)
    assert torch.equal(first.embed, same.embed)
    assert not torch.equal(first.embed, other.embed)


def test_init_scale_scales_the_weights():
    small = build_model(Config(p=23, d_model=32, d_mlp=64), 24, 23)
    large = build_model(Config(p=23, d_model=32, d_mlp=64, init_scale=4.0), 24, 23)
    ratio = large.embed.std() / small.embed.std()
    assert 3.5 < ratio < 4.5


def test_parameter_count_matches_the_arithmetic():
    cfg = Config(p=23, d_model=32, n_layers=1, n_heads=4, d_mlp=64, use_ln=False)
    model = build_model(cfg, cfg.p + 1, cfg.p)
    expected = (
        (cfg.p + 1) * 32  # token embedding
        + 3 * 32  # positional embedding
        + 4 * 32 * 32  # Q, K, V, O
        + 32 * 64 + 64  # feed forward in
        + 64 * 32 + 32  # feed forward out
        + 32 * cfg.p  # unembedding
    )
    assert n_parameters(model) == expected


def test_unknown_activation_and_arch():
    with pytest.raises(KeyError):
        activation("swish")
    with pytest.raises(KeyError):
        build_model(Config(arch="rnn"), 24, 23)


def test_head_divisibility_is_checked():
    with pytest.raises(ValueError, match="divisible"):
        Config(d_model=100, n_heads=3)
