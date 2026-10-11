"""Discrete Fourier analysis of what the network learned.

The reason this module exists at all: a network that has genuinely solved
modular addition has essentially no choice about *how*. Since

    cos(w(a + b)) = cos(wa)cos(wb) - sin(wa)sin(wb)
    sin(w(a + b)) = sin(wa)cos(wb) + cos(wa)sin(wb)

the only cheap way to turn a pair of inputs into a function of their sum is to
represent each input on a circle, rotate, and read the angle back off. So a
network that has understood the task must have put waves in its embedding
matrix, and a network that has merely memorised the training set has no reason
to. The gap between those two statements is measurable, and this module measures
it.

Everything below works in the orthonormal Fourier basis of Z_p, which for odd p
is the constant vector plus a cosine and a sine at each of the (p - 1) / 2
frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from grokking.data import is_prime


@lru_cache(maxsize=16)
def fourier_basis(n: int, device=None, dtype=torch.float32) -> tuple[Tensor, list[str]]:
    """Return the (n, n) orthonormal Fourier basis of the cyclic group Z_n.

    Row 0 is the constant, then cosine and sine at each frequency. When n is
    even the highest frequency n/2 has no sine part, since sin(pi x) vanishes at
    every integer, so it contributes a single row. Both cases are needed here:
    the additive group of Z_p has odd order p, and the multiplicative group used
    for division has even order p - 1.

    The result is cached because the training loop asks for it every time it
    logs. Treat the returned tensor as read only.
    """
    x = torch.arange(n, device=device, dtype=dtype)
    rows = [torch.ones(n, device=device, dtype=dtype)]
    labels = ["const"]
    for k in range(1, (n + 1) // 2):
        angle = 2 * torch.pi * k * x / n
        rows.append(torch.cos(angle))
        rows.append(torch.sin(angle))
        labels += [f"cos {k}", f"sin {k}"]
    if n % 2 == 0:
        rows.append(torch.cos(torch.pi * x))
        labels.append(f"cos {n // 2}")
    basis = torch.stack(rows)
    basis = basis / basis.norm(dim=1, keepdim=True)
    return basis, labels


def frequency_power(matrix: Tensor, n: int | None = None) -> Tensor:
    """Squared norm carried by each frequency of an (n, d) matrix.

    Returns a vector of length 1 + n // 2: the constant component followed by
    the combined cosine and sine power of frequencies 1 .. n // 2. By Parseval
    the entries sum to the squared Frobenius norm of the input.
    """
    matrix = matrix.detach().float()
    n = n if n is not None else matrix.shape[0]
    matrix = matrix[:n]
    basis, _ = fourier_basis(n, device=matrix.device)
    coeffs = basis @ matrix  # (n, d)
    power = coeffs.pow(2).sum(dim=1)

    combined = torch.empty(1 + n // 2, device=matrix.device)
    combined[0] = power[0]
    paired = power[1 : 1 + 2 * ((n - 1) // 2)]
    combined[1 : 1 + (n - 1) // 2] = paired.view(-1, 2).sum(dim=1)
    if n % 2 == 0:  # the lone Nyquist row
        combined[-1] = power[-1]
    return combined


def normalised_power(matrix: Tensor, p: int | None = None) -> Tensor:
    power = frequency_power(matrix, p)
    total = power.sum()
    return power / total.clamp_min(1e-12)


@dataclass
class Spectrum:
    """The Fourier profile of an embedding matrix at one moment in training."""

    power: np.ndarray
    """Normalised power per frequency, index 0 being the constant term."""

    key_freqs: np.ndarray
    """Frequencies (1-indexed) that carry an outsized share of the power."""

    concentration: float
    """Share of power held by the top-k frequencies, excluding the constant."""

    participation: float
    """Effective number of active frequencies; 1 / sum(share^2)."""

    @property
    def n_key(self) -> int:
        return int(len(self.key_freqs))


def analyse_embedding(
    matrix: Tensor, p: int | None = None, top_k: int = 5, threshold: float = 4.0
) -> Spectrum:
    """Summarise where an embedding matrix keeps its energy.

    `concentration` is the progress measure that matters. For a random matrix it
    sits a little above top_k / (p // 2), since the top few of many noisy
    frequencies always win something; for a network that has found the circular
    representation it approaches 1. It starts climbing long before validation
    accuracy moves, which is the whole point.
    """
    power = normalised_power(matrix, p).cpu().numpy()
    non_constant = power[1:]
    share = non_constant / max(non_constant.sum(), 1e-12)

    order = np.argsort(share)[::-1]
    concentration = float(share[order[:top_k]].sum())
    participation = float(1.0 / max((share**2).sum(), 1e-12))

    # A frequency counts as "key" when it carries several times more power than
    # both the typical frequency and a flat spectrum would give it. The second
    # half of that test matters: in a clean spectrum the median is numerically
    # zero, and without a floor every rounding error looks significant.
    uniform = 1.0 / len(share)
    baseline = max(float(np.median(share)), uniform)
    key = np.array([i + 1 for i in order if share[i] > threshold * baseline], dtype=int)
    return Spectrum(
        power=power,
        key_freqs=np.sort(key),
        concentration=concentration,
        participation=participation,
    )


def primitive_root(p: int) -> int:
    """The smallest g whose powers run through every non-zero residue mod p.

    Such a g exists for every prime, which is the fact that makes the
    multiplicative group of Z_p a circle of length p - 1.
    """
    if not is_prime(p):
        raise ValueError(f"primitive_root needs a prime, got {p}")
    for g in range(2, p):
        value, seen = 1, set()
        for _ in range(p - 1):
            value = (value * g) % p
            seen.add(value)
        if len(seen) == p - 1:
            return g
    raise ValueError(f"no primitive root found for {p}")  # pragma: no cover


def multiplicative_order(p: int, g: int | None = None) -> list[int]:
    """The residues 1 .. p-1 listed as g^0, g^1, g^2, ... instead of 1, 2, 3, ...

    Reading an embedding in this order turns multiplication into addition, which
    is the whole point: a network solving a / b mod p has every reason to lay the
    residues out on a circle, but on this circle, not the obvious one.
    """
    g = g if g is not None else primitive_root(p)
    return [pow(g, k, p) for k in range(p - 1)]


def reindex_multiplicatively(matrix: Tensor, p: int, g: int | None = None) -> Tensor:
    """Reorder an embedding's rows into g^0, g^1, ... and drop the row for 0.

    The result has p - 1 rows and should be analysed as a function on Z_{p-1}.
    """
    matrix = matrix.detach()
    order = torch.tensor(multiplicative_order(p, g), device=matrix.device)
    return matrix[order]


def analyse_in_both_bases(
    matrix: Tensor, p: int, top_k: int = 5
) -> dict[str, "Spectrum"]:
    """Score an embedding as a function on Z_p and on the multiplicative group.

    Addition lights up the first, division the second. Running both is the only
    honest way to use this measure, because a flat spectrum in one basis says
    nothing about whether structure exists in the other.
    """
    return {
        "additive": analyse_embedding(matrix, p=p, top_k=top_k),
        "multiplicative": analyse_embedding(
            reindex_multiplicatively(matrix, p), p=p - 1, top_k=top_k
        ),
    }


def restrict_to_frequencies(
    matrix: Tensor, freqs: np.ndarray | list[int], p: int | None = None, keep: bool = True
) -> Tensor:
    """Project an embedding onto (keep=True) or out of (keep=False) frequencies.

    This is the causal check on the Fourier story. If the waves at the key
    frequencies really are the mechanism, projecting everything else away should
    leave accuracy intact, and deleting just those frequencies should destroy it.
    """
    original = matrix
    matrix = matrix.detach().float()
    p = p if p is not None else matrix.shape[0]
    head, tail = matrix[:p], matrix[p:]

    basis, _ = fourier_basis(p, device=matrix.device)
    coeffs = basis @ head

    mask = torch.zeros(p, dtype=torch.bool, device=matrix.device)
    mask[0] = True  # the constant term is a bias, never the mechanism
    for f in freqs:
        mask[2 * int(f) - 1] = True
        mask[2 * int(f)] = True
    if not keep:
        mask = ~mask
        mask[0] = True

    filtered = basis.T @ (coeffs * mask[:, None])
    out = torch.cat([filtered, tail], dim=0)
    return out.to(original.dtype)


def circle_coordinates(matrix: Tensor, freq: int, p: int | None = None) -> np.ndarray:
    """Project each token onto the cos and sin directions of one frequency.

    Plotting the result is the fastest way to see the representation: after
    grokking, the p tokens land on a circle in frequency order (or in the order
    given by multiplying the index by the frequency, modulo p).
    """
    matrix = matrix.detach().float()
    p = p if p is not None else matrix.shape[0]
    head = matrix[:p]
    basis, _ = fourier_basis(p, device=matrix.device)
    cos_dir = basis[2 * freq - 1] @ head
    sin_dir = basis[2 * freq] @ head
    cos_dir = cos_dir / cos_dir.norm().clamp_min(1e-12)
    sin_dir = sin_dir / sin_dir.norm().clamp_min(1e-12)
    coords = torch.stack([head @ cos_dir, head @ sin_dir], dim=1)
    return coords.cpu().numpy()


def frequency_pair_power(values: np.ndarray) -> np.ndarray:
    """Power of a function on the (a, b) grid, binned by frequency pair.

    The raw two dimensional transform is p by p and puts each frequency's
    cosine and sine in separate rows, which spreads one real structure over four
    cells. Summing those quartets gives a (1 + p//2) square where cell (i, j)
    means "waves at frequency i in a times waves at frequency j in b", and a
    grokked neuron is one or two bright cells in it.
    """
    coeffs = two_dimensional_dft(values) ** 2
    p = values.shape[0]
    n = 1 + p // 2
    grouped = np.zeros((n, n))
    bounds = [(0, 1)] + [(2 * k - 1, 2 * k + 1) for k in range(1, n)]
    for i, (r0, r1) in enumerate(bounds):
        for j, (c0, c1) in enumerate(bounds):
            grouped[i, j] = coeffs[r0:r1, c0:c1].sum()
    return grouped


def two_dimensional_dft(values: np.ndarray) -> np.ndarray:
    """Fourier transform of a function on the (a, b) grid, in the real basis.

    Used on neuron activations: a neuron in a grokked network lights up as a
    handful of bright points in this plane rather than as a diffuse cloud.
    """
    p = values.shape[0]
    basis, _ = fourier_basis(p)
    basis_np = basis.cpu().numpy()
    return basis_np @ values @ basis_np.T
