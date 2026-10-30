"""InDivoice model families."""

from .b0_gmm import LFCCGMM
from .b1_cnn import LogMelCNN, count_parameters
from .b2_raw import RawWaveTiny


def build_model(name: str, variant: str = "small"):
    """Build a neural model by configured name."""
    if name == "b1":
        return LogMelCNN(variant=variant)
    if name == "b2":
        return RawWaveTiny()
    if name == "b3":
        from .b3_ssl import FrozenSSLHead

        return FrozenSSLHead()
    raise ValueError(f"Unknown neural model: {name}")


__all__ = ["LFCCGMM", "LogMelCNN", "RawWaveTiny", "build_model", "count_parameters"]
