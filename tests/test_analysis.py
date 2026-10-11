import numpy as np
import pytest
import torch

from grokking.analysis import (
    load_run,
    neuron_grid,
    per_frequency_ablation,
    prediction_grid,
    spectrum_over_time,
)
from grokking.config import Config
from grokking.train import train

from .conftest import SMALL_P


def test_prediction_grid_agrees_with_the_summary(small_run):
    run = load_run(small_run)
    correct, is_train = prediction_grid(run)

    assert correct.shape == (SMALL_P, SMALL_P) == is_train.shape
    assert is_train.sum() == run.summary["n_train"]
    assert (~is_train).sum() == run.summary["n_val"]

    measured = correct[is_train].mean()
    assert measured == pytest.approx(run.summary["final_train_acc"], abs=1e-6)
    measured_val = correct[~is_train].mean()
    assert measured_val == pytest.approx(run.summary["final_val_acc"], abs=1e-6)


def test_prediction_grid_handles_an_operation_with_a_restricted_domain(tmp_path):
    """Division excludes b = 0, so the table is p by p-1, not p by p."""
    cfg = Config.from_dict(
        dict(
            op="div",
            p=23,
            train_frac=0.6,
            arch="mlp",
            d_model=32,
            d_mlp=64,
            steps=60,
            eval_every=30,
            n_snapshots=4,
            threads=1,
            out_dir=str(tmp_path),
            name="div",
        )
    )
    train(cfg, progress=False)
    correct, is_train = prediction_grid(load_run(cfg.run_dir))
    assert correct.shape == (23, 22)
    assert is_train.sum() + (~is_train).sum() == 23 * 22


def test_neuron_grid_covers_the_whole_table(small_run):
    run = load_run(small_run)
    grid = neuron_grid(run)
    assert grid.shape[:2] == (SMALL_P, SMALL_P)
    assert grid.shape[2] == run.config.d_mlp
    assert np.isfinite(grid).all()


def test_per_frequency_ablation_names_the_frequencies_that_matter(small_run):
    run = load_run(small_run)
    damage = per_frequency_ablation(run)

    assert set(damage) == set(range(1, run.p // 2 + 1))
    assert all(0.0 <= v <= 1.0 for v in damage.values())

    key = set(run.summary["key_frequencies"])
    if key:
        # Deleting a key frequency must hurt more than deleting a typical one.
        worst = min(damage, key=damage.get)
        assert worst in key
        others = [v for f, v in damage.items() if f not in key]
        assert damage[worst] < np.median(others)


def test_spectrum_over_time_is_a_distribution_at_every_snapshot(small_run):
    power = spectrum_over_time(load_run(small_run))
    assert (power >= 0).all()
    assert np.allclose(power.sum(axis=1), 1.0, atol=1e-4)


def test_the_run_actually_grokked(small_run):
    """A guard on the fixture: the analysis tests mean nothing otherwise."""
    run = load_run(small_run)
    assert run.summary["final_val_acc"] > 0.9
    assert run.summary["key_frequencies"], "no Fourier structure to analyse"


def test_model_round_trips_through_the_checkpoint(small_run):
    run = load_run(small_run)
    model, data = run.model(), run.dataset()
    with torch.no_grad():
        accuracy = (model(data.val_x).argmax(-1) == data.val_y).float().mean().item()
    assert accuracy == pytest.approx(run.summary["final_val_acc"], abs=1e-6)
