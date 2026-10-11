import numpy as np
import pytest
import torch

from grokking.fourier import (
    analyse_embedding,
    analyse_in_both_bases,
    circle_coordinates,
    fourier_basis,
    frequency_power,
    multiplicative_order,
    primitive_root,
    reindex_multiplicatively,
    restrict_to_frequencies,
    two_dimensional_dft,
)

P = 97


def waves(frequencies, p=P, width=8, seed=0):
    """An embedding built from pure cosines at the given frequencies."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.arange(p, dtype=torch.float32)
    columns = []
    for k in frequencies:
        for _ in range(width):
            phase = torch.rand(1, generator=generator) * 2 * torch.pi
            columns.append(torch.cos(2 * torch.pi * k * x / p + phase))
    return torch.stack(columns, dim=1)


def test_basis_is_orthonormal():
    basis, labels = fourier_basis(P)
    assert basis.shape == (P, P)
    assert labels[0] == "const" and labels[1] == "cos 1"
    gram = basis @ basis.T
    assert torch.allclose(gram, torch.eye(P), atol=1e-5)


def test_basis_covers_the_whole_space_for_an_odd_order():
    """p rows for Z_p: the constant plus a cosine and a sine per frequency."""
    basis, labels = fourier_basis(P)
    assert basis.shape == (P, P)
    assert len(labels) == P == 1 + 2 * (P // 2)


def test_parseval():
    matrix = torch.randn(P, 16)
    assert frequency_power(matrix).sum().item() == pytest.approx(
        matrix.pow(2).sum().item(), rel=1e-4
    )


def test_pure_waves_are_recovered_exactly():
    spectrum = analyse_embedding(waves([7, 23]), p=P)
    assert spectrum.key_freqs.tolist() == [7, 23]
    assert spectrum.concentration == pytest.approx(1.0, abs=1e-4)
    assert spectrum.participation == pytest.approx(2.0, abs=1e-3)


def test_random_embeddings_have_no_key_frequencies():
    torch.manual_seed(0)
    spectrum = analyse_embedding(torch.randn(P, 128), p=P)
    assert spectrum.n_key == 0
    assert spectrum.concentration < 0.25
    assert spectrum.participation > P // 4


def test_the_extra_rows_of_the_embedding_are_ignored():
    """The '=' token sits at row p and must not take part in the transform."""
    matrix = waves([11])
    with_eq = torch.cat([matrix, torch.randn(1, matrix.shape[1]) * 50], dim=0)
    assert analyse_embedding(with_eq, p=P).key_freqs.tolist() == [11]


def test_restriction_keeps_what_it_says_and_drops_the_rest():
    matrix = waves([7, 23])
    kept = restrict_to_frequencies(matrix, [7, 23], P, keep=True)
    assert torch.allclose(kept, matrix, atol=1e-4)

    without = restrict_to_frequencies(matrix, [7], P, keep=False)
    assert analyse_embedding(without, p=P).key_freqs.tolist() == [23]

    only = restrict_to_frequencies(matrix, [7], P, keep=True)
    assert analyse_embedding(only, p=P).key_freqs.tolist() == [7]


def test_restriction_preserves_shape_including_the_eq_row():
    matrix = torch.randn(P + 1, 32)
    out = restrict_to_frequencies(matrix, [3], P, keep=True)
    assert out.shape == matrix.shape
    assert torch.allclose(out[P], matrix[P])


def test_circle_coordinates_land_on_a_circle():
    """A single clean frequency puts the tokens at equal angles on a circle."""
    x = torch.arange(P, dtype=torch.float32)
    freq = 5
    matrix = torch.stack(
        [torch.cos(2 * torch.pi * freq * x / P), torch.sin(2 * torch.pi * freq * x / P)],
        dim=1,
    )
    coords = circle_coordinates(matrix, freq, P)
    radii = np.hypot(coords[:, 0], coords[:, 1])
    assert radii.std() < 1e-4 * radii.mean() + 1e-5

    angles = np.unwrap(np.arctan2(coords[:, 1], coords[:, 0]))
    gaps = np.diff(angles) % (2 * np.pi)
    assert gaps.std() < 1e-4


def test_two_dimensional_dft_finds_a_planted_product_of_waves():
    a = np.arange(P)[:, None]
    b = np.arange(P)[None, :]
    freq = 9
    values = np.cos(2 * np.pi * freq * a / P) * np.cos(2 * np.pi * freq * b / P)
    spectrum = np.abs(two_dimensional_dft(values))

    peak = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    assert peak == (2 * freq - 1, 2 * freq - 1)
    share = spectrum[peak] ** 2 / (spectrum**2).sum()
    assert share > 0.9


# ---------------------------------------------------------------------------
# the multiplicative basis, which is what division turns out to need


@pytest.mark.parametrize("n", [7, 8, 58, 59, 96, 97])
def test_basis_is_orthonormal_for_both_parities(n):
    """The multiplicative group has even order p-1, so even n must work too."""
    basis, labels = fourier_basis(n)
    assert basis.shape == (n, n)
    assert torch.allclose(basis @ basis.T, torch.eye(n), atol=1e-5)
    assert len(labels) == n
    # an even group has a lone Nyquist cosine and no matching sine
    assert labels[-1] == (f"cos {n // 2}" if n % 2 == 0 else f"sin {n // 2}")


@pytest.mark.parametrize("n", [8, 58, 96])
def test_parseval_holds_for_even_orders(n):
    matrix = torch.randn(n, 9)
    power = frequency_power(matrix)
    assert len(power) == 1 + n // 2
    assert power.sum().item() == pytest.approx(matrix.pow(2).sum().item(), rel=1e-4)


@pytest.mark.parametrize("p,expected", [(59, 2), (97, 5), (11, 2), (7, 3)])
def test_primitive_roots_are_the_known_ones(p, expected):
    assert primitive_root(p) == expected


def test_primitive_root_rejects_composites():
    with pytest.raises(ValueError, match="prime"):
        primitive_root(57)


@pytest.mark.parametrize("p", [11, 59, 97])
def test_multiplicative_order_is_a_permutation_of_the_non_zero_residues(p):
    order = multiplicative_order(p)
    assert len(order) == p - 1
    assert sorted(order) == list(range(1, p))
    assert order[0] == 1


def test_reindexing_drops_the_zero_row_and_keeps_the_vectors():
    p = 11
    matrix = torch.arange(p * 3, dtype=torch.float32).reshape(p, 3)
    reindexed = reindex_multiplicatively(matrix, p)
    assert reindexed.shape == (p - 1, 3)
    order = multiplicative_order(p)
    for k, residue in enumerate(order):
        assert torch.equal(reindexed[k], matrix[residue])


def test_a_multiplicative_wave_is_invisible_in_the_additive_basis():
    """The point of the whole exercise, as a controlled experiment.

    Plant a clean wave over the residues in the order 1, g, g^2, ... . The
    multiplicative reading must find it; the additive reading must see noise.
    """
    p = 59
    order = multiplicative_order(p)
    k = np.arange(p - 1)
    columns = [np.cos(2 * np.pi * 7 * k / (p - 1) + phase) for phase in np.linspace(0, 3, 12)]
    planted = np.stack(columns, axis=1)

    matrix = np.zeros((p, planted.shape[1]), dtype=np.float32)
    for index, residue in enumerate(order):
        matrix[residue] = planted[index]

    both = analyse_in_both_bases(torch.from_numpy(matrix), p)
    assert both["multiplicative"].key_freqs.tolist() == [7]
    assert both["multiplicative"].concentration > 0.99
    assert both["additive"].n_key == 0
    assert both["additive"].concentration < 0.3


def test_an_additive_wave_is_invisible_in_the_multiplicative_basis():
    """And the converse, so the test is not just measuring one direction."""
    p = 59
    matrix = waves([9], p=p, width=12)
    both = analyse_in_both_bases(matrix, p)
    assert both["additive"].key_freqs.tolist() == [9]
    assert both["multiplicative"].n_key == 0
