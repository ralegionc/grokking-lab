"""Modular arithmetic datasets.

Every task is a binary operation on Z_p. The whole universe of examples is the
p * p grid of input pairs, so there is no sampling noise anywhere: a run sees a
fixed, deterministic subset of a finite table, and the validation set is exactly
the rest of that table. This is what makes grokking measurable. "Generalisation"
here means filling in the cells of a multiplication table that were never shown.

A sequence is three tokens long:

    [ a , b , = ]        with  = encoded as the integer p

and the model is asked for the answer at the final position, so the vocabulary
is p + 1 tokens on the input side and p on the output side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class Operation:
    """A binary operation on Z_p together with its domain."""

    name: str
    symbol: str
    fn: Callable[[Tensor, Tensor, int], Tensor]
    b_nonzero: bool = False
    """True when b = 0 must be excluded (division)."""

    requires_prime: bool = False

    def __call__(self, a: Tensor, b: Tensor, p: int) -> Tensor:
        return self.fn(a, b, p) % p


def _modular_inverse(b: Tensor, p: int) -> Tensor:
    """b^(p-2) mod p, which is b^-1 for prime p by Fermat's little theorem."""
    result = torch.ones_like(b)
    base = b.clone() % p
    exponent = p - 2
    while exponent > 0:
        if exponent & 1:
            result = (result * base) % p
        base = (base * base) % p
        exponent >>= 1
    return result


OPERATIONS: dict[str, Operation] = {
    "add": Operation("add", "a + b", lambda a, b, p: a + b),
    "sub": Operation("sub", "a - b", lambda a, b, p: a - b),
    "mul": Operation("mul", "a * b", lambda a, b, p: a * b),
    "div": Operation(
        "div",
        "a / b",
        lambda a, b, p: a * _modular_inverse(b, p),
        b_nonzero=True,
        requires_prime=True,
    ),
    "quad": Operation("quad", "a^2 + ab + b^2", lambda a, b, p: a * a + a * b + b * b),
    "cube": Operation("cube", "a^3 + ab", lambda a, b, p: a * a * a + a * b),
}


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


@dataclass
class Dataset:
    """A frozen train/validation split of one arithmetic table."""

    op: Operation
    p: int
    train_x: Tensor  # (n_train, 3) int64
    train_y: Tensor  # (n_train,)   int64
    val_x: Tensor
    val_y: Tensor
    all_x: Tensor
    all_y: Tensor
    train_index: Tensor
    """Indices into all_x of the examples that were shown during training."""

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def eq_token(self) -> int:
        return self.p

    @property
    def seq_len(self) -> int:
        return 3

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset({self.op.symbol} mod {self.p}: "
            f"{len(self.train_x)} train / {len(self.val_x)} val)"
        )


def build_dataset(op: str, p: int, train_frac: float, seed: int = 0) -> Dataset:
    """Enumerate the full arithmetic table and split it deterministically."""
    if op not in OPERATIONS:
        raise KeyError(f"unknown op {op!r}; choose from {sorted(OPERATIONS)}")
    operation = OPERATIONS[op]
    if operation.requires_prime and not is_prime(p):
        raise ValueError(f"op {op!r} needs a prime modulus, got p={p}")

    a = torch.arange(p, dtype=torch.long)
    b_lo = 1 if operation.b_nonzero else 0
    b = torch.arange(b_lo, p, dtype=torch.long)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    aa, bb = aa.reshape(-1), bb.reshape(-1)

    eq = torch.full_like(aa, p)
    all_x = torch.stack([aa, bb, eq], dim=1)
    all_y = operation(aa, bb, p)

    n = len(all_x)
    n_train = int(round(n * train_frac))
    if not 0 < n_train < n:
        raise ValueError(f"train_frac={train_frac} leaves an empty split for p={p}")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    train_index, val_index = perm[:n_train], perm[n_train:]

    return Dataset(
        op=operation,
        p=p,
        train_x=all_x[train_index],
        train_y=all_y[train_index],
        val_x=all_x[val_index],
        val_y=all_y[val_index],
        all_x=all_x,
        all_y=all_y,
        train_index=train_index,
    )
