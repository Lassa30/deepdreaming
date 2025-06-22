import pytest
import torch

from deepdreaming import smoothing


@pytest.mark.parametrize(
    ("kernel_size", "padding_mode", "smoothing_func"),
    zip(
        (3, 5, 7, 9),
        ("constant", "reflect", "replicate", "circular"),
        (smoothing.gaussian_smoothing, smoothing.box_smoothing, smoothing.gaussian_smoothing, smoothing.box_smoothing),
    ),
)
def test_shape_property(kernel_size, padding_mode, smoothing_func):
    tensor = torch.rand((1, 3, 224, 224))

    after_smoothing = smoothing_func(tensor, kernel_size, padding_mode)

    assert isinstance(after_smoothing, torch.Tensor)
    assert tensor.shape == after_smoothing.shape
