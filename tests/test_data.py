import pytest
import torch

from grokking.data import OPERATIONS, build_dataset, is_prime


def test_primality_helper():
    assert [n for n in range(2, 20) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]
    assert not is_prime(1)
    assert not is_prime(97 * 2)


@pytest.mark.parametrize("op", sorted(OPERATIONS))
def test_targets_match_python_arithmetic(op):
    """The vectorised table must agree with plain Python on every cell."""
    p = 23
    data = build_dataset(op, p, 0.5, seed=0)
    a, b = data.all_x[:, 0].tolist(), data.all_x[:, 1].tolist()
    expected = []
    for x, y in zip(a, b):
        if op == "add":
            expected.append((x + y) % p)
        elif op == "sub":
            expected.append((x - y) % p)
        elif op == "mul":
            expected.append((x * y) % p)
        elif op == "div":
            expected.append((x * pow(y, p - 2, p)) % p)
        elif op == "quad":
            expected.append((x * x + x * y + y * y) % p)
        elif op == "cube":
            expected.append((x**3 + x * y) % p)
    assert data.all_y.tolist() == expected


def test_division_is_the_inverse_of_multiplication():
    p = 31
    data = build_dataset("div", p, 0.5, seed=0)
    a, b = data.all_x[:, 0], data.all_x[:, 1]
    assert ((b * data.all_y) % p == a % p).all()
    assert (b > 0).all(), "division must never be asked about b = 0"


def test_division_needs_a_prime_modulus():
    with pytest.raises(ValueError, match="prime"):
        build_dataset("div", 20, 0.5)


def test_split_is_disjoint_and_covers_the_table():
    data = build_dataset("add", 47, 0.4, seed=3)
    n = len(data.all_x)
    assert len(data.train_x) + len(data.val_x) == n
    assert len(data.train_x) == round(n * 0.4)

    def as_set(x):
        return {tuple(row) for row in x.tolist()}

    train, val = as_set(data.train_x), as_set(data.val_x)
    assert train.isdisjoint(val), "a validation pair leaked into training"
    assert len(train | val) == n


def test_split_depends_only_on_the_seed():
    first = build_dataset("add", 47, 0.4, seed=3)
    same = build_dataset("add", 47, 0.4, seed=3)
    other = build_dataset("add", 47, 0.4, seed=4)
    assert torch.equal(first.train_index, same.train_index)
    assert not torch.equal(first.train_index, other.train_index)


def test_token_layout():
    p = 17
    data = build_dataset("add", p, 0.5, seed=0)
    assert data.seq_len == 3
    assert data.vocab_size == p + 1
    assert (data.all_x[:, 2] == data.eq_token).all()
    assert data.all_x[:, :2].max() < p


def test_degenerate_splits_are_rejected():
    with pytest.raises(ValueError):
        build_dataset("add", 5, 0.001)
    with pytest.raises(ValueError):
        build_dataset("add", 5, 1.0)


def test_unknown_operation():
    with pytest.raises(KeyError):
        build_dataset("xor", 17, 0.5)
