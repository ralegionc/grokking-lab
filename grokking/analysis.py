"""Post-hoc analysis of a finished run.

Loading a run gives you the metric history, the log-spaced embedding snapshots
and the trained model. From those three things everything in the paper-style
figures follows, including the causal tests: the Fourier story is only worth
telling if deleting the frequencies it names actually breaks the model.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from grokking.config import Config
from grokking.data import Dataset, build_dataset
from grokking.fourier import (
    analyse_embedding,
    normalised_power,
    restrict_to_frequencies,
)
from grokking.metrics import evaluate
from grokking.model import build_model


@dataclass
class Run:
    """Everything one run left on disk."""

    path: Path
    config: Config
    summary: dict[str, Any]
    history: dict[str, np.ndarray]
    snapshot_steps: np.ndarray
    snapshot_embeds: np.ndarray  # (n_snapshots, vocab, d_model)

    @property
    def p(self) -> int:
        return self.config.p

    def dataset(self) -> Dataset:
        cfg = self.config
        return build_dataset(cfg.op, cfg.p, cfg.train_frac, cfg.seed)

    def model(self) -> nn.Module:
        path = self.path / "model.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Sweep points are stored without their weights; "
                f"retrain this one if you need the model itself."
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(self.config, self.config.p + 1, self.config.p)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model

    @property
    def has_snapshots(self) -> bool:
        return len(self.snapshot_steps) > 0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Run({self.path.name}, {len(self.history['step'])} evaluations)"


def load_run(path: str | Path) -> Run:
    path = Path(path)
    config = Config.from_json(path / "config.json")
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))

    rows = [
        json.loads(line)
        for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = sorted({k for row in rows for k in row})
    history = {k: np.array([row.get(k, np.nan) for row in rows], dtype=float) for k in keys}

    # A run keeps its metrics forever but may have had its snapshots stripped,
    # which is how the sweep points are stored. Loading one of those for its
    # curves alone has to work.
    snapshots = path / "snapshots.npz"
    if snapshots.exists():
        with np.load(snapshots) as blob:
            snapshot_steps = blob["steps"]
            snapshot_embeds = blob["embed"].astype(np.float32)
    else:
        snapshot_steps = np.zeros(0, dtype=np.int64)
        snapshot_embeds = np.zeros((0, 0, 0), dtype=np.float32)

    return Run(
        path=path,
        config=config,
        summary=summary,
        history=history,
        snapshot_steps=snapshot_steps,
        snapshot_embeds=snapshot_embeds,
    )


def spectrum_over_time(run: Run) -> np.ndarray:
    """(n_snapshots, 1 + p // 2) normalised Fourier power of the embedding.

    Plotted as a heatmap this is the clearest single picture of grokking: a flat
    field of noise that slowly resolves into a handful of bright horizontal
    lines, and the lines appear while the validation loss is still flat.
    """
    p = run.p
    out = []
    for embed in run.snapshot_embeds:
        power = normalised_power(torch.from_numpy(embed), p=p).numpy()
        out.append(power)
    return np.stack(out)


@dataclass
class Ablation:
    """Accuracy of the model when the embedding is filtered in some way."""

    label: str
    train_acc: float
    val_acc: float
    freqs: list[int]
    """The key frequencies the condition was defined against."""


@torch.no_grad()
def _accuracy_with_embedding(model: nn.Module, data: Dataset, embed: torch.Tensor):
    patched = copy.deepcopy(model)
    patched.embed.data = embed.to(patched.embed.dtype)
    train = evaluate(patched, data.train_x, data.train_y)
    val = evaluate(patched, data.val_x, data.val_y)
    return train.accuracy, val.accuracy


def frequency_ablations(run: Run, top_k: int = 6) -> list[Ablation]:
    """Ask whether the key frequencies are the mechanism or a side effect.

    Three conditions. Keep everything; keep only the key frequencies; keep
    everything except the key frequencies. If the Fourier reading is right, the
    middle one costs almost nothing and the last one is fatal.
    """
    model, data = run.model(), run.dataset()
    embed = model.embed.detach().clone()
    spectrum = analyse_embedding(embed, p=run.p, top_k=top_k)
    key = spectrum.key_freqs
    if len(key) == 0:  # a memorising model has no key frequencies to speak of
        key = np.argsort(spectrum.power[1:])[::-1][:top_k] + 1

    freqs = [int(f) for f in key]
    conditions = [
        ("all frequencies", embed),
        ("only the key frequencies", restrict_to_frequencies(embed, key, run.p, keep=True)),
        (
            "every frequency except those",
            restrict_to_frequencies(embed, key, run.p, keep=False),
        ),
    ]
    results = []
    for label, matrix in conditions:
        train_acc, val_acc = _accuracy_with_embedding(model, data, matrix)
        results.append(Ablation(label, train_acc, val_acc, freqs))
    return results


def per_frequency_ablation(run: Run) -> dict[int, float]:
    """Validation accuracy after deleting each single frequency in turn."""
    model, data = run.model(), run.dataset()
    embed = model.embed.detach().clone()
    out: dict[int, float] = {}
    for freq in range(1, run.p // 2 + 1):
        matrix = restrict_to_frequencies(embed, [freq], run.p, keep=False)
        _, val_acc = _accuracy_with_embedding(model, data, matrix)
        out[freq] = val_acc
    return out


@torch.no_grad()
def neuron_grid(run: Run, layer: str = "mlp.neuron_post") -> np.ndarray:
    """Activation of every hidden neuron over the whole (a, b) grid.

    Shape (p, p, n_neurons). A neuron in a grokked network is a product of waves
    in a and b, which is obvious the moment you look at one.
    """
    model, data = run.model(), run.dataset()
    p = run.p
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    tokens = torch.stack([a, b, torch.full_like(a, p)], dim=1)

    cache: dict[str, torch.Tensor] = {}
    model(tokens, cache)
    key = layer if layer in cache else f"blocks.0.{layer}"
    acts = cache[key]
    if acts.dim() == 3:  # transformer: take the position the answer is read from
        acts = acts[:, -1]
    return acts.reshape(p, p, -1).numpy()


@torch.no_grad()
def prediction_grid(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """Correctness over the full arithmetic table, plus the train/val mask.

    Returns (correct, is_train), both (p, p) and both aligned to the (a, b) grid
    with b restricted to the operation's domain.
    """
    model, data = run.model(), run.dataset()
    logits = model(data.all_x)
    correct = (logits.argmax(dim=-1) == data.all_y).numpy()

    is_train = np.zeros(len(data.all_x), dtype=bool)
    is_train[data.train_index.numpy()] = True

    n_b = len(torch.unique(data.all_x[:, 1]))
    return correct.reshape(run.p, n_b), is_train.reshape(run.p, n_b)


def phase_table(run: Run) -> dict[str, Any]:
    """The three numbers that describe the shape of a grokking run."""
    history = run.history
    step = history["step"]

    def first(metric: str, threshold: float) -> int | None:
        hits = np.nonzero(history[metric] >= threshold)[0]
        return int(step[hits[0]]) if len(hits) else None

    memorised = first("train_acc", 0.99)
    generalised = first("val_acc", 0.95)
    peak_norm = int(step[int(np.argmax(history["weight_norm"]))])
    return {
        "memorised_at": memorised,
        "generalised_at": generalised,
        "gap": None if None in (memorised, generalised) else generalised - memorised,
        "peak_weight_norm_at": peak_norm,
        "final_val_acc": float(history["val_acc"][-1]),
        "key_frequencies": run.summary.get("key_frequencies", []),
    }
