"""Run configuration.

A run is fully described by this dataclass. It is serialised next to the
metrics so that any figure in the repository can be traced back to the exact
settings that produced it.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # ---- task -------------------------------------------------------------
    op: str = "add"
    """Binary operation on Z_p. See grokking.data.OPERATIONS."""

    p: int = 97
    """Modulus. Must be prime for 'div' to be well defined."""

    train_frac: float = 0.4
    """Fraction of the p*p (or p*(p-1)) pairs used for training."""

    # ---- model ------------------------------------------------------------
    arch: str = "transformer"
    """'transformer' or 'mlp'."""

    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 4
    d_mlp: int | None = None
    """Width of the feed-forward hidden layer. Defaults to 4 * d_model."""

    act: str = "relu"
    """Feed-forward nonlinearity: 'relu', 'gelu' or 'quadratic'."""

    use_ln: bool = True
    dropout: float = 0.0

    # ---- optimisation -----------------------------------------------------
    steps: int = 25_000
    batch_size: int | None = None
    """None means full batch, which is the standard grokking setup."""

    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 10
    grad_clip: float | None = None
    init_scale: float = 1.0
    """Multiplier applied to all initial weights. Large values grok sooner."""

    # ---- bookkeeping ------------------------------------------------------
    seed: int = 0
    eval_every: int = 25
    n_snapshots: int = 60
    """Number of log-spaced parameter snapshots kept for later analysis."""

    device: str = "cpu"
    threads: int | None = None
    out_dir: str = "runs"
    name: str | None = None
    notes: str = ""

    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        self.betas = tuple(self.betas)  # JSON gives a list
        if self.arch == "transformer" and self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}"
            )
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError("train_frac must lie strictly between 0 and 1")

    @property
    def run_name(self) -> str:
        if self.name:
            return self.name
        return f"{self.op}_p{self.p}_{self.arch}_wd{self.weight_decay:g}_s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    # -----------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path, **overrides: Any) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.pop("$schema", None)
        raw.update(overrides)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["betas"] = list(self.betas)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
