import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


def validate_input_shape(input_tensor: torch.Tensor):
    assert input_tensor.dim() == 4, "Invalid number of input data dimensions"
    B, C, H, W = input_tensor.shape
    assert B == 1, "Batched inputs are not allowed here."


def get_gaussian_kernel(kernel_size, sigma):
    half_size = kernel_size // 2
    x_dim, y_dim = torch.arange(-half_size, half_size + 1), torch.arange(-half_size, half_size + 1)
    x_grid, y_grid = torch.meshgrid(x_dim, y_dim, indexing="ij")
    weight = (1 / (2 * torch.pi) ** 0.5 / sigma) * torch.exp((x_grid**2 + y_grid**2) / -2 / sigma**2)
    assert isinstance(weight, torch.Tensor), f"Weight should be a `torch.Tensor`\nGot {type(weight)} instead."
    return (weight / weight.sum()).repeat(3, 1, 1, 1)


def validate_blend(sigmas, blending_weigths):
    # fmt: off
    assert len(blending_weigths) == len(sigmas), \
        f"""
        Different number of sigmas and kernel weights.
            Input: {len(blending_weigths)}
            Expected input: {len(sigmas)}
        """
    # fmt: on


def normalize_blend(blending_weights: tuple):
    # normalization is not needed - it's alright
    if abs(1 - sum(blending_weights)) <= 1e-6:
        return blending_weights

    # we need to normalize otherwise
    warnings.warn(
        f"""
        Kernel weights don't sum up to 1:
            Weights: {blending_weights}
            Their sum: {sum(blending_weights)}
        Weights will be normalized implicitly.
        """
    )

    return tuple(coef / sum(blending_weights) for coef in blending_weights)


def gaussian_smoothing(
    input_tensor: torch.Tensor,
    kernel_size: int = 3,
    padding_mode: str = "reflect",
    sigmas: int | float | tuple = 2,
    blending_weights: Optional[tuple | int | float] = None,
) -> torch.Tensor:
    if not isinstance(sigmas, tuple):
        sigmas = (sigmas,)
    if isinstance(blending_weights, int | float | None):
        blending_weights = tuple(1 / len(sigmas) for _ in range(len(sigmas)))

    validate_input_shape(input_tensor)
    validate_blend(sigmas, blending_weights)

    C = input_tensor.shape[1]
    blending_weights = normalize_blend(blending_weights)

    kernels = []
    for sigma in sigmas:
        kernel = get_gaussian_kernel(kernel_size, sigma)
        kernels.append(kernel)

    pad = kernel_size // 2
    padded_input_tensor = F.pad(input_tensor, (pad, pad, pad, pad), mode=padding_mode)

    output_tensor = torch.zeros_like(input=input_tensor)
    for blending_weight, kernel in zip(blending_weights, kernels):
        output_tensor += blending_weight * F.conv2d(padded_input_tensor, kernel, groups=C)
    return output_tensor


def box_smoothing(input_tensor: torch.Tensor, kernel_size: int = 3, padding_mode: str = "reflect") -> torch.Tensor:
    validate_input_shape(input_tensor)
    C = input_tensor.shape[1]

    padding = kernel_size // 2
    padded_image = F.pad(input_tensor, (padding, padding, padding, padding), mode=padding_mode)
    weights = torch.ones(C, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)

    return F.conv2d(padded_image, weights, groups=C)
