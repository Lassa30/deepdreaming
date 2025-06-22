from dataclasses import dataclass

import torch


@dataclass
class GradSmoothingMode:
    BoxSmoothing = "Box Smoothing"
    GaussianSmoothing = "Gaussian Smoothing"
    Disable = "No Smoothing"


@dataclass
class DreamConfig:
    """Configuration settings for DeepDream algorithm.

    This dataclass contains all hyperparameters and settings needed to control
    the DeepDream generation process, including optimization, image pyramid,
    gradient processing, and augmentation parameters.

    Attributes:
        learning_rate (float): Step size for gradient ascent optimization. Defaults to 0.09.
        num_iter (int): Number of optimization steps per pyramid layer. Defaults to 10.
        optimizer_class (type): PyTorch optimizer class to use. Defaults to torch.optim.Adam.
        gradient_norm (bool): Whether to normalize gradients to unit norm. Defaults to True.
        norm (int): L-norm type for gradient normalization (1, 2, etc.). Defaults to 2.
        grad_smoothing (str): Mode for gradient smoothing. Options are "box", "gaussian", and "no".
            Defaults to "gaussian".
        grad_smoothing_kernel_size (int): Size of the smoothing kernel. Must be an odd integer. Defaults to 3.
        grad_smoothing_padding_mode (str): Padding mode for smoothing.
            See `torch.nn.functional.pad` for details. Defaults to "reflect".
        grad_smoothing_gaussian_sigmas (int | float | tuple): Standard deviation(s) for Gaussian smoothing.
            A larger sigma results in more blurring. Multiple sigmas can be provided as a tuple,
            in which case multiple Gaussian kernels are generated and blended together. Defaults to (0.5, 1, 1.5).
            If `grad_smoothing` is set to "gaussian", this parameter is used.
        grad_smoothing_gaussian_blending_weights (int | float | tuple | None): Blending weights for multiple
            Gaussian kernels. If None, all kernels are weighted equally. If not None, the number of blending
            weights must match the number of sigmas. Defaults to None.
            If `grad_smoothing` is set to "gaussian", this parameter is used.
        pyramid_layers (int): Number of pyramid layers for multi-scale processing. Defaults to 5.
        pyramid_ratio (float): Scale ratio between pyramid layers (0 < ratio < 1). Defaults to 2/3.
        shift_size (int): Maximum pixel shift for random augmentation. Defaults to 32.

    Note:
        - When `grad_smoothing` is set to "gaussian", the following restrictions apply:
            - `grad_smoothing_kernel_size` must be an odd integer.
            - The number of `grad_smoothing_gaussian_sigmas` and
              `grad_smoothing_gaussian_blending_weights` must match, if blending weights are provided.
              See `deepdreaming/smoothing.py` for implementation details and assertions.
            - If `grad_smoothing_gaussian_blending_weights` don't sum up to 1 then the normalization is implicitly applied.
              It's better to pass normalized weights to make it explicit.
    """

    # -- General --
    learning_rate: float = 0.09
    num_iter: int = 10
    optimizer_class: type = torch.optim.Adam

    # -- Norm --
    gradient_norm: bool = True
    norm: int = 2

    # -- Smoothing --
    grad_smoothing: str = GradSmoothingMode.GaussianSmoothing
    # params for both smoothers
    grad_smoothing_kernel_size: int = 3
    grad_smoothing_padding_mode: str = "reflect"
    # gaussian ONLY
    grad_smoothing_gaussian_sigmas: tuple[int | float, ...] = (0.5,)
    grad_smoothing_gaussian_blending_weights: tuple[int | float, ...] = (1.0,)

    # -- Other --
    pyramid_layers: int = 5
    pyramid_ratio: float = 2 / 3
    shift_size: int = 32
