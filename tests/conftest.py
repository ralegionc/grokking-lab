import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grokking.config import Config  # noqa: E402
from grokking.train import train  # noqa: E402

SMALL_P = 47


@pytest.fixture(scope="session")
def small_run(tmp_path_factory) -> Path:
    """One real, tiny, grokking run that the slower tests all share.

    Small enough to train in a few seconds and still reach the state the
    analysis is about: a network that has generalised and has a clean Fourier
    structure in its embedding.
    """
    out = tmp_path_factory.mktemp("runs")
    cfg = Config.from_dict(
        dict(
            op="add",
            p=SMALL_P,
            train_frac=0.7,
            arch="mlp",
            act="quadratic",
            d_model=48,
            d_mlp=128,
            weight_decay=2.0,
            steps=700,
            eval_every=25,
            n_snapshots=8,
            threads=2,
            out_dir=str(out),
            name="small",
        )
    )
    train(cfg, progress=False)
    return cfg.run_dir
