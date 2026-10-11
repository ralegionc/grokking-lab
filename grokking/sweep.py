"""Sweep one setting and collect the summaries.

The two sweeps worth running are the ones that decide whether grokking happens
at all rather than how fast it looks:

    --vary train_frac      how much of the table the model is shown
    --vary weight_decay    how hard the optimiser is pushed toward a small solution

Both are cheap with the MLP configuration and expensive with the transformer, so
the defaults point at the MLP.

    python -m grokking.sweep --config configs/quick_mlp.json \
        --vary train_frac --values 0.2 0.3 0.4 0.5 0.6 --workers 3
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from grokking.config import Config
from grokking.train import train


def _run_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point. Takes a plain dict so it pickles cleanly."""
    cfg = Config.from_dict(payload)
    return train(cfg, progress=False)


def sweep(
    base: Config,
    field: str,
    values: list[Any],
    out_dir: Path,
    workers: int = 1,
    threads_per_worker: int = 3,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    for value in values:
        raw = base.to_dict()
        raw[field] = value
        raw["name"] = f"{field}_{value}"
        raw["out_dir"] = str(out_dir)
        raw["threads"] = threads_per_worker
        raw["n_snapshots"] = min(base.n_snapshots, 30)
        payloads.append(raw)

    print(f"sweeping {field} over {values} with {workers} worker(s)")
    if workers <= 1:
        summaries = [_run_one(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(_run_one, payloads))

    for value, summary in zip(values, summaries):
        summary[field] = value
        grok = summary["grok_step_95"]
        print(
            f"  {field}={value!r:>8}  grokked at {grok if grok is not None else 'never':>8}"
            f"  final held out accuracy {summary['final_val_acc']:.3f}"
        )

    path = out_dir / f"sweep_{field}.json"
    path.write_text(
        json.dumps({"field": field, "values": values, "runs": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
    return summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m grokking.sweep")
    parser.add_argument("--config", type=Path, default=Path("configs/quick_mlp.json"))
    parser.add_argument("--vary", required=True, help="config field to sweep")
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/sweeps"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=3, dest="threads")
    args = parser.parse_args(argv)

    parsed: list[Any] = []
    for raw in args.values:
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            parsed.append(raw)

    base = Config.from_json(args.config)
    sweep(base, args.vary, parsed, args.out, args.workers, args.threads)


if __name__ == "__main__":  # pragma: no cover
    main()
