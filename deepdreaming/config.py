from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


@dataclass
class DreamConfig:
    optimizer_class: type = torch.optim.Adam
    learning_rate: float = 0.07
    num_iterations: int = 10
    pyramid_layers: int = 5
    pyramid_ratio: float = 2/3
    shift_size: int = 32
