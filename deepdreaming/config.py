from dataclasses import dataclass
import torch


@dataclass
class DreamConfig:
    # -- General --
    learning_rate: float = 0.07
    num_iterations: int = 10
    optimizer_class: type = torch.optim.Adam

    # -- Norm --
    gradient_norm: bool = False
    norm: int = 2

    # -- Smoothing --
    gradient_smooth: bool = False

    # -- Other --
    pyramid_layers: int = 5
    pyramid_ratio: float = 2 / 3
    shift_size: int = 32
