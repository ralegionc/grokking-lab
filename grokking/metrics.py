"""Everything the training loop records.

Two kinds of measurement live here. The first kind is the usual one: loss and
accuracy on each split. The second kind looks inside the weights and asks how
far the Fourier circuit has come, and that one is the reason the repository
exists, because it moves while the first kind is flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from grokking.fourier import analyse_embedding


@dataclass
class Evaluation:
    loss: float
    accuracy: float

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {f"{prefix}_loss": self.loss, f"{prefix}_acc": self.accuracy}


@torch.no_grad()
def evaluate(model: nn.Module, x: Tensor, y: Tensor, batch_size: int = 4096) -> Evaluation:
    """Cross entropy and top-1 accuracy over a whole split."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    for start in range(0, len(x), batch_size):
        xb, yb = x[start : start + batch_size], y[start : start + batch_size]
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
    model.train()
    n = max(len(x), 1)
    return Evaluation(loss=total_loss / n, accuracy=total_correct / n)


@torch.no_grad()
def weight_norm(model: nn.Module) -> float:
    """L2 norm of every trainable parameter, stacked into one vector."""
    return float(
        torch.sqrt(sum(p.detach().pow(2).sum() for p in model.parameters())).item()
    )


@torch.no_grad()
def embedding_matrix(model: nn.Module) -> Tensor:
    """The token embedding, whichever architecture we are looking at."""
    return model.embed.detach()


@torch.no_grad()
def circuit_measures(model: nn.Module, p: int, top_k: int = 5) -> dict[str, float]:
    """How much of a Fourier circuit is present in the embedding right now."""
    spectrum = analyse_embedding(embedding_matrix(model), p=p, top_k=top_k)
    return {
        "freq_concentration": spectrum.concentration,
        "freq_participation": spectrum.participation,
        "n_key_freqs": float(spectrum.n_key),
    }


@dataclass
class History:
    """The metrics log, in memory and on disk at the same time."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def column(self, name: str) -> list[float]:
        return [row[name] for row in self.rows if name in row]

    def first_step_where(self, name: str, threshold: float) -> int | None:
        """The first logged step at which a metric reached a threshold.

        Used for the two numbers that define a grokking run: when the model
        memorised the training set, and when it finally generalised.
        """
        for row in self.rows:
            if row.get(name, float("-inf")) >= threshold:
                return int(row["step"])
        return None
