import json

import numpy as np
import pytest

from grokking.analysis import frequency_ablations, load_run, phase_table, spectrum_over_time
from grokking.config import Config
from grokking.metrics import History
from grokking.train import parse_overrides, snapshot_steps, train


def tiny(tmp_path, **overrides) -> Config:
    base = dict(
        op="add",
        p=17,
        train_frac=0.6,
        arch="mlp",
        d_model=16,
        d_mlp=32,
        steps=120,
        eval_every=20,
        n_snapshots=6,
        threads=1,
        out_dir=str(tmp_path),
        name="tiny",
    )
    base.update(overrides)
    return Config.from_dict(base)


def test_snapshot_steps_are_log_spaced_and_bounded():
    steps = snapshot_steps(10_000, 20)
    assert steps[0] == 0 and steps[-1] == 10_000
    assert steps == sorted(set(steps))
    assert all(0 <= s <= 10_000 for s in steps)
    # log spacing means far more snapshots in the first tenth than the last
    assert sum(s < 1_000 for s in steps) > sum(s >= 1_000 for s in steps)


def test_snapshot_steps_degenerate_counts():
    assert snapshot_steps(500, 2) == [0, 500]


def test_config_round_trips_through_json(tmp_path):
    cfg = Config(p=23, betas=(0.9, 0.95), name="x")
    path = tmp_path / "config.json"
    cfg.save(path)
    assert Config.from_json(path).to_dict() == cfg.to_dict()


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_dict({"not_a_field": 1})


def test_overrides_are_parsed_as_json():
    out = parse_overrides(["steps=50", "act=\"gelu\"", "betas=[0.9, 0.99]", "name=run1"])
    assert out == {"steps": 50, "act": "gelu", "betas": [0.9, 0.99], "name": "run1"}
    with pytest.raises(SystemExit):
        parse_overrides(["oops"])


def test_history_threshold_lookup():
    history = History()
    for step, acc in [(0, 0.1), (10, 0.5), (20, 0.97), (30, 0.99)]:
        history.append({"step": step, "val_acc": acc})
    assert history.first_step_where("val_acc", 0.95) == 20
    assert history.first_step_where("val_acc", 1.1) is None
    assert history.column("val_acc") == [0.1, 0.5, 0.97, 0.99]


def test_a_short_run_writes_everything_and_learns(tmp_path):
    cfg = tiny(tmp_path)
    summary = train(cfg, progress=False)
    run_dir = cfg.run_dir

    for name in ("config.json", "metrics.jsonl", "snapshots.npz", "summary.json"):
        assert (run_dir / name).exists(), f"{name} was not written"

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) >= 6
    assert rows[-1]["train_loss"] < rows[0]["train_loss"], "training did not reduce the loss"
    assert summary["n_train"] + summary["n_val"] == cfg.p * cfg.p
    assert summary["final_train_acc"] >= summary["final_val_acc"]


def test_run_artifacts_load_back(tmp_path):
    cfg = tiny(tmp_path)
    train(cfg, progress=False)
    run = load_run(cfg.run_dir)

    assert run.config.to_dict() == cfg.to_dict()
    assert run.snapshot_embeds.shape[1:] == (cfg.p + 1, cfg.d_model)
    assert len(run.snapshot_steps) == len(run.snapshot_embeds)

    power = spectrum_over_time(run)
    assert power.shape == (len(run.snapshot_steps), 1 + cfg.p // 2)
    assert np.allclose(power.sum(axis=1), 1.0, atol=1e-4)

    table = phase_table(run)
    assert set(table) >= {"memorised_at", "generalised_at", "gap"}

    model = run.model()
    assert model(run.dataset().val_x).shape[1] == cfg.p


def test_ablations_report_three_conditions(tmp_path):
    cfg = tiny(tmp_path)
    train(cfg, progress=False)
    results = frequency_ablations(load_run(cfg.run_dir))
    assert len(results) == 3
    assert all(0.0 <= r.val_acc <= 1.0 for r in results)
    assert "all frequencies" in results[0].label


def test_minibatch_training_also_runs(tmp_path):
    cfg = tiny(tmp_path, batch_size=32, name="tiny_batched")
    summary = train(cfg, progress=False)
    assert summary["steps"] == 120


def test_full_batch_training_is_reproducible(tmp_path):
    first = train(tiny(tmp_path, name="a"), progress=False)
    second = train(tiny(tmp_path, name="b"), progress=False)
    assert first["final_train_acc"] == second["final_train_acc"]
    assert first["final_val_loss"] == pytest.approx(second["final_val_loss"], rel=1e-9)


def test_a_stop_file_ends_the_run_but_leaves_it_analysable(tmp_path):
    """A long run must be interruptible without losing everything."""
    cfg = tiny(tmp_path, steps=100_000, eval_every=10, name="stopped")
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "STOP").write_text("", encoding="utf-8")

    summary = train(cfg, progress=False)

    assert summary["steps"] <= 10, "the STOP file should be noticed at the first check"
    assert summary["steps_requested"] == 100_000
    assert "STOP" in summary["stopped_early"]

    for name in ("metrics.jsonl", "snapshots.npz", "summary.json", "model.pt"):
        assert (cfg.run_dir / name).exists(), f"{name} missing after an early stop"

    run = load_run(cfg.run_dir)
    assert len(run.snapshot_steps) == len(run.snapshot_embeds) >= 1
    assert run.snapshot_steps[-1] == summary["steps"]
    assert run.model() is not None


def test_a_completed_run_records_no_early_stop(tmp_path):
    summary = train(tiny(tmp_path, name="complete"), progress=False)
    assert summary["stopped_early"] is None
    assert summary["steps"] == summary["steps_requested"] == 120
