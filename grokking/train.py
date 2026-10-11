"""The training loop.

One run is one call to `train`. It writes four files into its run directory:

    config.json      the exact settings, including the resolved defaults
    metrics.jsonl    one JSON object per evaluation, appended as it goes
    snapshots.npz    log-spaced embedding matrices, for the Fourier analysis
    summary.json     the headline numbers, including when the model grokked

The loop is full batch by default. That is not laziness: the whole point of the
setup is that the training set is a fixed finite table, so there is no sampling
noise to explain anything away. What happens, happens because of the optimiser
and the weight decay.

These runs are long, so a run can be cut short two ways and both leave a
complete, analysable run directory behind: press Ctrl-C, or drop a file named
STOP into the run directory. The summary records how far it actually got in
`steps` and why it stopped in `stopped_early`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from grokking.config import Config
from grokking.data import build_dataset
from grokking.fourier import analyse_embedding
from grokking.metrics import (
    History,
    circuit_measures,
    embedding_matrix,
    evaluate,
    weight_norm,
)
from grokking.model import build_model, n_parameters

HEADER = (
    f"{'step':>7}  {'train loss':>10}  {'val loss':>9}  "
    f"{'train acc':>9}  {'val acc':>8}  {'|w|':>7}  {'freq conc':>9}"
)


def snapshot_steps(total: int, count: int) -> list[int]:
    """Log-spaced steps, always including 0 and the final step.

    Log spacing because everything interesting about grokking happens on a log
    axis: the memorisation is over within a few hundred steps and the
    generalisation arrives tens of thousands of steps later.
    """
    if count <= 2:
        return [0, total]
    raw = np.unique(np.round(np.logspace(0, np.log10(total), count - 1)).astype(int))
    return sorted({0, *(int(s) for s in raw), total})


def lr_lambda(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup)


def train(cfg: Config, progress: bool = True, print_every: int = 500) -> dict[str, Any]:
    """Run one experiment end to end and return its summary."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    data = build_dataset(cfg.op, cfg.p, cfg.train_frac, cfg.seed)
    model = build_model(cfg, data.vocab_size, cfg.p).to(device)
    train_x, train_y = data.train_x.to(device), data.train_y.to(device)
    val_x, val_y = data.val_x.to(device), data.val_y.to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: lr_lambda(s, cfg.warmup_steps)
    )

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    history = History()
    snaps = set(snapshot_steps(cfg.steps, cfg.n_snapshots))
    snap_steps: list[int] = []
    snap_embeds: list[np.ndarray] = []

    if progress:
        print(f"{data}   {cfg.arch}, {n_parameters(model):,} parameters")
        print(f"run directory: {run_dir}")
        print(HEADER)
        print("-" * len(HEADER))

    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 7)
    started = time.time()
    step = 0
    stopped_early: str | None = None
    metrics_file = metrics_path.open("a", encoding="utf-8")

    try:
        for step in range(cfg.steps + 1):
            # ---- one optimiser step ---------------------------------------
            if step > 0:
                if cfg.batch_size is None:
                    batch_x, batch_y = train_x, train_y
                else:
                    pick = torch.randint(
                        len(train_x), (cfg.batch_size,), generator=generator
                    )
                    batch_x, batch_y = train_x[pick], train_y[pick]

                logits = model(batch_x)
                loss = F.cross_entropy(logits, batch_y)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimiser.step()
                schedule.step()

            # ---- evaluation ------------------------------------------------
            if step % cfg.eval_every == 0 or step == cfg.steps:
                train_eval = evaluate(model, train_x, train_y)
                val_eval = evaluate(model, val_x, val_y)
                row: dict[str, Any] = {"step": step}
                row.update(train_eval.as_dict("train"))
                row.update(val_eval.as_dict("val"))
                row["weight_norm"] = weight_norm(model)
                row["lr"] = schedule.get_last_lr()[0]
                row.update(circuit_measures(model, cfg.p))
                row["elapsed"] = round(time.time() - started, 2)

                history.append(row)
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()

                if progress and (step % print_every == 0 or step == cfg.steps):
                    print(
                        f"{step:>7}  {row['train_loss']:>10.4f}  {row['val_loss']:>9.4f}  "
                        f"{row['train_acc']:>9.3f}  {row['val_acc']:>8.3f}  "
                        f"{row['weight_norm']:>7.2f}  {row['freq_concentration']:>9.3f}",
                        flush=True,
                    )

            # ---- snapshot ---------------------------------------------------
            if step in snaps:
                snap_steps.append(step)
                snap_embeds.append(
                    embedding_matrix(model).cpu().numpy().astype(np.float16)
                )

            # ---- an early exit that still leaves a usable run ----------------
            if step and step % cfg.eval_every == 0 and (run_dir / "STOP").exists():
                stopped_early = f"a STOP file appeared in {run_dir}"
                break
    except KeyboardInterrupt:
        stopped_early = "interrupted from the keyboard"
    finally:
        metrics_file.close()

    # Everything below runs even when the loop was cut short, so a run that is
    # stopped after an hour is a shorter run rather than a wasted one.
    if stopped_early and progress:
        print(f"stopping at step {step:,}: {stopped_early}", flush=True)
    if step not in snaps:
        snap_steps.append(step)
        snap_embeds.append(embedding_matrix(model).cpu().numpy().astype(np.float16))

    np.savez_compressed(
        run_dir / "snapshots.npz",
        steps=np.array(snap_steps, dtype=np.int64),
        embed=np.stack(snap_embeds),
    )
    torch.save(
        {"config": cfg.to_dict(), "state_dict": model.state_dict()}, run_dir / "model.pt"
    )

    spectrum = analyse_embedding(embedding_matrix(model), p=cfg.p)
    final = history.rows[-1]

    summary = {
        "run_name": cfg.run_name,
        "op": cfg.op,
        "p": cfg.p,
        "arch": cfg.arch,
        "train_frac": cfg.train_frac,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        "steps": step,
        "steps_requested": cfg.steps,
        "stopped_early": stopped_early,
        "n_parameters": n_parameters(model),
        "n_train": len(train_x),
        "n_val": len(val_x),
        "memorisation_step": history.first_step_where("train_acc", 0.99),
        "grok_step_95": history.first_step_where("val_acc", 0.95),
        "grok_step_99": history.first_step_where("val_acc", 0.99),
        "final_train_acc": final["train_acc"],
        "final_val_acc": final["val_acc"],
        "final_val_loss": final["val_loss"],
        "final_weight_norm": final["weight_norm"],
        "key_frequencies": [int(f) for f in spectrum.key_freqs],
        "freq_concentration": spectrum.concentration,
        "wall_seconds": round(time.time() - started, 1),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if progress:
        print("-" * len(HEADER))
        memo, grok = summary["memorisation_step"], summary["grok_step_95"]
        print(f"memorised at step {memo}, generalised at step {grok}")
        if memo is not None and grok is not None:
            print(f"the gap between the two is {grok - memo:,} steps")
        print(f"key frequencies: {summary['key_frequencies']}")
        print(f"wall clock: {summary['wall_seconds'] / 60:.1f} min")
    return summary


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn --set key=value arguments into typed config overrides."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m grokking.train",
        description="Train one model on modular arithmetic and watch it grok.",
    )
    parser.add_argument("--config", type=Path, help="path to a JSON config file")
    parser.add_argument("--op", help="add, sub, mul, div, quad or cube")
    parser.add_argument("--p", type=int, help="modulus")
    parser.add_argument("--arch", help="transformer or mlp")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weight-decay", type=float, dest="weight_decay")
    parser.add_argument("--train-frac", type=float, dest="train_frac")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--name", help="run directory name")
    parser.add_argument("--out-dir", dest="out_dir", help="where run directories go")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any other config field, value parsed as JSON",
    )
    parser.add_argument("--print-every", type=int, default=500, dest="print_every")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    overrides = parse_overrides(args.set)
    for field in (
        "op",
        "p",
        "arch",
        "steps",
        "seed",
        "weight_decay",
        "train_frac",
        "threads",
        "name",
        "out_dir",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value

    cfg = (
        Config.from_json(args.config, **overrides)
        if args.config
        else Config.from_dict(overrides)
    )
    train(cfg, progress=not args.quiet, print_every=args.print_every)


if __name__ == "__main__":  # pragma: no cover
    main()
