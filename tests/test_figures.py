import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grokking import style  # noqa: E402
from grokking.figures import FIGURES, render, render_sweep  # noqa: E402
from grokking.fourier import frequency_pair_power  # noqa: E402
from scripts.report import to_markdown  # noqa: E402

P = 47


def test_frequency_pair_power_finds_a_planted_product():
    a = np.arange(P)[:, None]
    b = np.arange(P)[None, :]
    values = np.cos(2 * np.pi * 6 * a / P) * np.sin(2 * np.pi * 11 * b / P)
    grouped = frequency_pair_power(values)

    assert grouped.shape == (1 + P // 2, 1 + P // 2)
    assert np.unravel_index(np.argmax(grouped), grouped.shape) == (6, 11)
    assert grouped[6, 11] / grouped.sum() > 0.99


def test_frequency_pair_power_conserves_energy():
    values = np.random.default_rng(0).normal(size=(P, P))
    assert frequency_pair_power(values).sum() == pytest.approx((values**2).sum(), rel=1e-6)


def test_label_stacking_separates_coincident_series():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    style.label_series_ends(ax, [(1.0, "a", "#000"), (1.0, "b", "#111")], min_gap=0.08)
    positions = sorted(text.xy[1] for text in ax.texts)
    assert len(positions) == 2
    assert positions[1] - positions[0] >= 0.079
    plt.close(fig)


def test_header_wraps_a_long_subtitle():
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots(figsize=(6, 4))
    style.header(fig, "Headline", "word " * 80, style.LIGHT)
    subtitle = [t for t in fig.texts if t.get_text() != "Headline"][0]
    assert subtitle.get_text().count("\n") >= 3
    assert fig.subplotpars.top < 0.86, "a tall header must push the axes down"
    plt.close(fig)


def test_every_figure_renders_in_both_themes(small_run, tmp_path):
    written = render(small_run, tmp_path)
    assert len(written) == 2 * len(FIGURES)
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 5_000, f"{path.name} looks empty"

    names = {path.name for path in written}
    assert names == {f"{name}.png" for name in FIGURES}


def test_sweep_figure_renders_including_runs_that_never_grokked(tmp_path):
    blob = {
        "field": "train_frac",
        "values": [0.2, 0.4],
        "runs": [
            {"train_frac": 0.2, "grok_step_95": None, "steps": 1000},
            {"train_frac": 0.4, "grok_step_95": 300, "steps": 1000},
        ],
    }
    path = tmp_path / "sweep_train_frac.json"
    path.write_text(__import__("json").dumps(blob), encoding="utf-8")
    written = render_sweep(path, tmp_path / "figures")
    assert len(written) == 2
    assert all(p.exists() for p in written)


def test_report_table_renders_a_run_that_never_grokked():
    summaries = [
        {
            "run_name": "a",
            "op": "add",
            "p": 59,
            "arch": "mlp",
            "train_frac": 0.5,
            "weight_decay": 2.0,
            "memorisation_step": 100,
            "grok_step_95": 900,
            "final_val_acc": 1.0,
            "key_frequencies": [3, 5],
        },
        {
            "run_name": "b",
            "op": "add",
            "p": 59,
            "arch": "mlp",
            "train_frac": 0.2,
            "weight_decay": 0.0,
            "memorisation_step": 80,
            "grok_step_95": None,
            "final_val_acc": 0.02,
            "key_frequencies": [],
        },
    ]
    table = to_markdown(summaries)
    assert "9x" in table and "never" in table and "none" in table
    assert len(table.splitlines()) == 4


# ---------------------------------------------------------------------------
# the two basis comparison and the seed spread


def _tiny_run(tmp_path, name, op="add", seed=0, p=11):
    from grokking.config import Config
    from grokking.train import train

    cfg = Config.from_dict(
        dict(
            op=op,
            p=p,
            train_frac=0.7,
            arch="mlp",
            act="quadratic",
            d_model=24,
            d_mlp=48,
            weight_decay=2.0,
            steps=80,
            eval_every=20,
            n_snapshots=4,
            threads=1,
            seed=seed,
            out_dir=str(tmp_path),
            name=name,
        )
    )
    train(cfg, progress=False)
    return cfg.run_dir


def test_two_basis_figure_renders(tmp_path):
    from grokking.figures import render_two_bases

    add_dir = _tiny_run(tmp_path, "add_small", op="add")
    div_dir = _tiny_run(tmp_path, "div_small", op="div")
    written = render_two_bases(add_dir, div_dir, tmp_path / "figures")
    assert len(written) == 2
    assert all(path.exists() and path.stat().st_size > 5_000 for path in written)


def test_seed_spread_figure_renders(tmp_path):
    from grokking.figures import render_seed_spread

    sweep = tmp_path / "sweeps"
    for seed in (0, 1):
        _tiny_run(sweep, f"seed_{seed}", seed=seed)
    written = render_seed_spread(sweep, tmp_path / "figures")
    assert len(written) == 2
    assert all(path.exists() for path in written)


def test_seed_spread_needs_seed_runs(tmp_path):
    from grokking.figures import render_seed_spread

    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no seed"):
        render_seed_spread(tmp_path / "empty", tmp_path / "figures")


def test_a_run_without_snapshots_still_loads_its_curves(tmp_path):
    """Sweep points are shipped without weights or snapshots."""
    from grokking.analysis import load_run

    run_dir = _tiny_run(tmp_path, "stripped")
    (run_dir / "snapshots.npz").unlink()
    (run_dir / "model.pt").unlink()

    run = load_run(run_dir)
    assert not run.has_snapshots
    assert len(run.history["val_acc"]) >= 4
    assert run.summary["final_val_acc"] >= 0.0
    with pytest.raises(FileNotFoundError, match="retrain"):
        run.model()
