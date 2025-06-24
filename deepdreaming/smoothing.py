import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from .config import DreamConfig


def get_box_smoothing_params(config: DreamConfig):
    return config.grad_smoothing_kernel_size, config.grad_smoothing_padding_mode


def get_gaussian_smoothing_params(config: DreamConfig):
    return (
        config.grad_smoothing_kernel_size,
        config.grad_smoothing_padding_mode,
        config.grad_smoothing_gaussian_sigmas,
        config.grad_smoothing_gaussian_blending_weights,
    )


def validate_input_shape(input_tensor: torch.Tensor):
    """Validates the shape of the input tensor.

    Args:
        input_tensor (torch.Tensor): The input tensor to validate.

    Raises:
        AssertionError: If the input tensor does not have 4 dimensions or if the batch size is not 1.
    """
    assert input_tensor.dim() == 4, "Invalid number of input data dimensions"
    B, C, H, W = input_tensor.shape
    assert B == 1, "Batched inputs are not allowed here."


def validate_blend(sigmas, blending_weigths):
    """Validates the blending weights.

    Args:
        sigmas (tuple): The sigma values.
        blending_weigths (tuple): The blending weights to validate.

    Raises:
        AssertionError: If the number of sigmas and blending weights do not match.
    """
    # fmt: off
    assert len(blending_weigths) == len(sigmas), \
        f"""
        Different number of sigmas and blending weights.
            sigmas: {sigmas}
            blending_weights: {blending_weigths}
        """
    # fmt: on


def get_gaussian_kernel(kernel_size, sigma):
    """Generates a Gaussian kernel.

    Args:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: The normalized Gaussian kernel.
    """
    half_size = kernel_size // 2
    x_dim, y_dim = torch.arange(-half_size, half_size + 1), torch.arange(-half_size, half_size + 1)
    x_grid, y_grid = torch.meshgrid(x_dim, y_dim, indexing="ij")
    weight = (1 / (2 * torch.pi) ** 0.5 / sigma) * torch.exp((x_grid**2 + y_grid**2) / -2 / sigma**2)
    assert isinstance(weight, torch.Tensor), f"Weight should be a `torch.Tensor`\nGot {type(weight)} instead."
    return (weight / weight.sum()).repeat(3, 1, 1, 1)


def normalize_blend(blending_weights: tuple):
    """Normalizes the blending weights.

    Args:
        blending_weights (tuple): The blending weights to normalize.

    Returns:
        tuple: The normalized blending weights.
    """
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
    sigma: tuple[int | float, ...] = (0.5,),
    blending_weights: Optional[tuple[int | float, ...]] = None,
) -> torch.Tensor:
    """Applies Gaussian smoothing to an input tensor.

    This function performs Gaussian smoothing on the input image tensor using a
    convolutional approach. It generates Gaussian kernels based on the specified
    sigma values and applies them to the input.  Multiple sigmas can be used,
    with the results blended together using blending weights.

    Args:
        input_tensor (torch.Tensor): The input image tensor of shape (B, C, H, W), where
            B is the batch size, C is the number of color channels, H is the height, and W is the width.
            Note that B must be equal to 1.
        kernel_size (int): The size of the Gaussian kernel. Must be an odd integer.
            A larger kernel_size results in more blurring.
        padding_mode (str): The padding mode to use.  See `torch.nn.functional.pad` for details.
        sigmas (int | float | tuple): The standard deviation(s) of the Gaussian distribution.
            A larger sigma results in more blurring.  Multiple sigmas can be provided as a tuple,
            in which case multiple Gaussian kernels are generated and blended together.
        blending_weights (Optional[tuple | int | float]): The blending weights to use when blending
            multiple Gaussian kernels.  If None, all kernels are weighted equally.  If not None, the
            number of blending weights must match the number of sigmas.

    Returns:
        torch.Tensor: The smoothed image tensor of the same shape as the input.
    """
    assert kernel_size % 2, "Kernel size must be odd."
    if blending_weights is None:
        blending_weights = tuple(1 / len(sigma) for _ in range(len(sigma)))

    validate_input_shape(input_tensor)
    validate_blend(sigma, blending_weights)

    C = input_tensor.shape[1]
    blending_weights = normalize_blend(blending_weights)

    kernels = []
    for s in sigma:
        kernel = get_gaussian_kernel(kernel_size, s)
        kernels.append(kernel)

    pad = kernel_size // 2
    padded_input_tensor = F.pad(input_tensor, (pad, pad, pad, pad), mode=padding_mode)

    output_tensor = torch.zeros_like(input=input_tensor, device=padded_input_tensor.device)
    for blending_weight, kernel in zip(blending_weights, kernels):
        output_tensor += blending_weight * F.conv2d(padded_input_tensor, kernel, groups=C)
    return output_tensor


def box_smoothing(input_tensor: torch.Tensor, kernel_size: int = 3, padding_mode: str = "reflect") -> torch.Tensor:
    """Applies box smoothing to an input tensor.

    This function performs box smoothing on the input image tensor using a
    convolutional approach. It generates a box kernel of the specified size
    and applies it to each color channel of the image.

    Args:
        input_tensor (torch.Tensor): The input image tensor of shape (B, C, H, W), where
            B is the batch size, C is the number of color channels, H is the height, and W is the width.
            Note that B must be equal to 1.
        kernel_size (int): The size of the box kernel. Must be an odd integer.
            A larger kernel_size results in more blurring.
        padding_mode (str): The padding mode to use.  See `torch.nn.functional.pad` for details.

    Returns:
        torch.Tensor: The smoothed image tensor of the same shape as the input.
    """
    assert kernel_size % 2, "Kernel size must be odd."
    assert padding_mode in ("constant", "reflect", "replicate", "circular")

    validate_input_shape(input_tensor)
    C = input_tensor.shape[1]

    padding = kernel_size // 2
    padded_input_tensor = F.pad(input_tensor, (padding, padding, padding, padding), mode=padding_mode)
    weights = torch.ones(C, 1, kernel_size, kernel_size, device=padded_input_tensor.device)
    weights /= kernel_size * kernel_size

    return F.conv2d(padded_input_tensor, weights, groups=C)
