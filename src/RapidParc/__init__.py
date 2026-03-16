try:
    import torch
except ImportError:
    raise ImportError(
        "\n\nPyTorch is not installed, but it is required for RapidParc.\n" \
        "There are two options: \n"
        "(i)\t For GPU support, please install PyTorch according to your hardware (recommended): \n"
        "\t\t https://pytorch.org/get-started/locally \n"
        "(ii)\t To install the CPU version, run:\n"
        "\t\t pip install rapidparc[cpu]\n"
    )


from .run import RapidParc, RapidParcTckEval
from .test import test
from .train import train

__all__ = ["RapidParc", "RapidParcTckEval", "test", "train"]