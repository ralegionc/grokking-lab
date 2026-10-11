"""The models, written out by hand.

Nothing here is exotic. It is a decoder-only transformer with learned positional
embeddings, pre-norm blocks and multi-head attention, spelled out rather than
delegated to torch.nn.Transformer so that every tensor the analysis later
reaches for actually exists and has a name.

Both models accept an optional `cache` dictionary. When one is passed, the
forward pass writes its intermediate activations into it, which is how
grokking.analysis gets at attention patterns and neuron activations without
monkey-patching anything.
"""

from __future__ import annotations

import math
from typing import MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Cache = MutableMapping[str, Tensor]


def activation(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "quadratic":
        return lambda x: x * x
    raise KeyError(f"unknown activation {name!r}")


class LayerNorm(nn.Module):
    """Written out rather than imported, because the analysis needs the scale."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - trivial
        return x


class SelfAttention(nn.Module):
    """Causal multi-head self attention over a three token sequence."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cache: Cache | None = None, prefix: str = "") -> Tensor:
        batch, seq, _ = x.shape
        shape = (batch, seq, self.n_heads, self.d_head)
        q = self.W_Q(x).view(shape).transpose(1, 2)  # (B, H, T, d_head)
        k = self.W_K(x).view(shape).transpose(1, 2)
        v = self.W_V(x).view(shape).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        causal = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        pattern = scores.softmax(dim=-1)

        z = pattern @ v  # (B, H, T, d_head)
        z = z.transpose(1, 2).reshape(batch, seq, -1)
        if cache is not None:
            cache[prefix + "pattern"] = pattern.detach()
            cache[prefix + "z"] = z.detach()
        return self.W_O(z)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act: str) -> None:
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        self.act = activation(act)

    def forward(self, x: Tensor, cache: Cache | None = None, prefix: str = "") -> Tensor:
        pre = self.W_in(x)
        post = self.act(pre)
        if cache is not None:
            cache[prefix + "neuron_pre"] = pre.detach()
            cache[prefix + "neuron_post"] = post.detach()
        return self.W_out(post)


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        act: str,
        use_ln: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model) if use_ln else Identity()
        self.attn = SelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model) if use_ln else Identity()
        self.mlp = FeedForward(d_model, d_mlp, act)
        self.drop = nn.Dropout(dropout) if dropout else Identity()

    def forward(self, x: Tensor, cache: Cache | None = None, prefix: str = "") -> Tensor:
        x = x + self.drop(self.attn(self.ln1(x), cache, prefix + "attn."))
        x = x + self.drop(self.mlp(self.ln2(x), cache, prefix + "mlp."))
        if cache is not None:
            cache[prefix + "resid_post"] = x.detach()
        return x


class Transformer(nn.Module):
    """Decoder-only transformer that reads logits off the final position."""

    def __init__(
        self,
        vocab_size: int,
        n_out: int,
        seq_len: int = 3,
        d_model: int = 128,
        n_layers: int = 1,
        n_heads: int = 4,
        d_mlp: int = 512,
        act: str = "relu",
        use_ln: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Parameter(torch.empty(vocab_size, d_model))
        self.pos_embed = nn.Parameter(torch.empty(seq_len, d_model))
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_mlp, act, use_ln, dropout) for _ in range(n_layers)
        )
        self.ln_final = LayerNorm(d_model) if use_ln else Identity()
        self.unembed = nn.Linear(d_model, n_out, bias=False)
        self.reset_parameters()

    def reset_parameters(self, scale: float = 1.0) -> None:
        std = 1.0 / math.sqrt(self.d_model)
        for name, param in self.named_parameters():
            if param.dim() >= 2 or name in {"embed", "pos_embed"}:
                nn.init.normal_(param, mean=0.0, std=std * scale)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, tokens: Tensor, cache: Cache | None = None) -> Tensor:
        x = self.embed[tokens] + self.pos_embed
        if cache is not None:
            cache["resid_pre"] = x.detach()
        for i, block in enumerate(self.blocks):
            x = block(x, cache, f"blocks.{i}.")
        return self.unembed(self.ln_final(x)[:, -1])


class MLP(nn.Module):
    """The two layer baseline: embed both operands, concatenate, one hidden layer.

    It groks on the same tasks for a fraction of the compute, which makes it the
    right model for sweeps. With act='quadratic' the learned solution sits very
    close to the exact trigonometric identity a transformer arrives at the long
    way round.
    """

    def __init__(
        self,
        vocab_size: int,
        n_out: int,
        d_model: int = 128,
        d_mlp: int = 512,
        act: str = "relu",
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Parameter(torch.empty(vocab_size, d_model))
        self.W_in = nn.Linear(2 * d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, n_out)
        self.act = activation(act)
        self.drop = nn.Dropout(dropout) if dropout else Identity()
        self.reset_parameters()

    def reset_parameters(self, scale: float = 1.0) -> None:
        std = 1.0 / math.sqrt(self.d_model)
        nn.init.normal_(self.embed, std=std * scale)
        for layer in (self.W_in, self.W_out):
            nn.init.normal_(layer.weight, std=std * scale)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: Tensor, cache: Cache | None = None) -> Tensor:
        x = self.embed[tokens[:, :2]].flatten(1)
        pre = self.W_in(x)
        post = self.act(pre)
        if cache is not None:
            cache["resid_pre"] = x.detach()
            cache["mlp.neuron_pre"] = pre.detach()
            cache["mlp.neuron_post"] = post.detach()
        return self.W_out(self.drop(post))


def build_model(cfg, vocab_size: int, n_out: int) -> nn.Module:
    """Construct the model described by a Config."""
    torch.manual_seed(cfg.seed + 1000)
    common = dict(
        vocab_size=vocab_size,
        n_out=n_out,
        d_model=cfg.d_model,
        d_mlp=cfg.d_mlp,
        act=cfg.act,
        dropout=cfg.dropout,
    )
    if cfg.arch == "transformer":
        model = Transformer(
            seq_len=3,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            use_ln=cfg.use_ln,
            **common,
        )
    elif cfg.arch == "mlp":
        model = MLP(**common)
    else:
        raise KeyError(f"unknown arch {cfg.arch!r}")
    if cfg.init_scale != 1.0:
        model.reset_parameters(scale=cfg.init_scale)
    return model


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
