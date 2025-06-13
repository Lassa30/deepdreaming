from dataclasses import dataclass
import torch


@dataclass
class DreamConfig:
    """Configuration settings for DeepDream algorithm.
    
    This dataclass contains all hyperparameters and settings needed to control
    the DeepDream generation process, including optimization, image pyramid,
    gradient processing, and augmentation parameters.
    
    Attributes:
        learning_rate (float): Step size for gradient ascent optimization. Defaults to 0.07.
        num_iterations (int): Number of optimization steps per pyramid layer. Defaults to 10.
        optimizer_class (type): PyTorch optimizer class to use. Defaults to torch.optim.Adam.
        gradient_norm (bool): Whether to normalize gradients to unit norm. Defaults to False.
        norm (int): L-norm type for gradient normalization (1, 2, etc.). Defaults to 2.
        gradient_smooth (bool): Whether to apply gradient smoothing. Defaults to False.
        pyramid_layers (int): Number of pyramid layers for multi-scale processing. Defaults to 5.
        pyramid_ratio (float): Scale ratio between pyramid layers (0 < ratio < 1). Defaults to 2/3.
        shift_size (int): Maximum pixel shift for random augmentation. Defaults to 32.
    """

    # -- General --
    learning_rate: float = 0.09
    num_iter: int = 10
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
