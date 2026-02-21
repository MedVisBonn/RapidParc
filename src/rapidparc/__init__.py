try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Please install :\n"
        "https://pytorch.org/get-started/locally/"
    )

from .run import rapidParc
from .test import test
from .train import train

__all__ = ["rapidParc", "test", "train"]