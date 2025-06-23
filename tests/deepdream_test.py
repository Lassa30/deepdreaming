import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
import torch
from torchvision.models import resnet18

from deepdreaming.deepdream import DeepDream
from deepdreaming.config import DreamConfig, GradSmoothingMode


@composite
def dream_config_strategy(draw):
    """Strategy to generate valid DreamConfig objects within constrained parameter ranges."""
    lr = draw(st.floats(min_value=1e-3, max_value=1))  # learning_rate
    num_iter = draw(st.integers(min_value=2, max_value=10))
    shift_size = draw(st.integers(min_value=0, max_value=256))
    smoothing_mode = draw(
        st.sampled_from(
            [
                GradSmoothingMode.BoxSmoothing,
                GradSmoothingMode.GaussianSmoothing,
                GradSmoothingMode.Disable,
            ]
        )
    )
    kernel_size = draw(st.sampled_from([3, 5, 7]))
    padding_mode = draw(st.sampled_from(["constant", "reflect", "replicate", "circular"]))
    pyramid_layers = draw(st.integers(min_value=1, max_value=3))
    pyramid_ratio = draw(st.floats(min_value=0.5, max_value=0.9))

    return DreamConfig(
        learning_rate=lr,
        num_iter=num_iter,
        shift_size=shift_size,
        grad_smoothing=smoothing_mode,
        grad_smoothing_kernel_size=kernel_size,
        grad_smoothing_padding_mode=padding_mode,
        pyramid_layers=pyramid_layers,
        pyramid_ratio=pyramid_ratio,
    )


@settings(max_examples=5, deadline=None)
@given(
    height=st.integers(min_value=128, max_value=256),
    width=st.integers(min_value=128, max_value=256),
    config=dream_config_strategy(),
)
def test_dream_basic_properties(height, width, config):
    image = np.random.rand(height, width, 3).astype(np.float32)
    assert image.min() >= 0 and image.max() <= 1

    model = resnet18(weights=None).cpu().eval()
    deepdreamer = DeepDream(model, ["layer1.0.conv1"])

    output = deepdreamer.dream(image, config=config)

    assert output.shape == image.shape, "Output shape differs from input shape."
    assert np.all(output >= 0.0) and np.all(output <= 1.0), "Output not clipped to [0,1]."


@settings(max_examples=3, deadline=None)
@given(
    height=st.integers(min_value=128, max_value=256),
    width=st.integers(min_value=128, max_value=256),
    config=dream_config_strategy(),
)
def test_dream_guided_basic_properties(height, width, config):
    input_img = np.random.rand(height, width, 3).astype(np.float32)
    ref_img = np.random.rand(height, width, 3).astype(np.float32)

    model = resnet18(weights=None).cpu().eval()
    deepdreamer = DeepDream(model, ["layer1.0.conv1"])

    output = deepdreamer.dream_guided(input_img, ref_img, config=config)

    assert output.shape == input_img.shape, "Output shape differs from input shape"
    assert np.all(output >= 0.0) and np.all(output <= 1.0), "Output not clipped to [0,1]"

    assert not np.allclose(output, input_img), "Output should differ from input"
    assert not np.allclose(output, ref_img), "Output should differ from reference"
