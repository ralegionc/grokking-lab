"""Grokking Lab: watching a transformer memorise, then understand.

The package is deliberately small and readable. Four modules do the work:

    data      modular arithmetic datasets and deterministic splits
    model     a decoder-only transformer written out by hand
    fourier   the discrete Fourier analysis that exposes the learned circuit
    train     the training loop and the metrics it records
"""

__version__ = "1.0.0"

from grokking.config import Config

__all__ = ["Config", "__version__"]
