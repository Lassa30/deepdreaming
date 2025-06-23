import numpy as np
import pytest
import torch

from deepdreaming.img import io, proc


@pytest.fixture
def static_image_path():
    return "tests/static/tree.png"


@pytest.fixture
def static_image_shape():
    return (350, 750, 3)


@pytest.fixture
def static_image(static_image_path):
    return io.read_image(static_image_path)


class TestImgIo:
    def test_read_with_shape(self, static_image_path):
        """
        Checks if read_image respects the desired shape.
        """
        desired_shape = (224, 224, 3)
        image = io.read_image(static_image_path, desired_shape)
        assert image.shape == desired_shape, "The image shape should match desired_shape."
        assert image.max() <= 1.0, "Pixel values should be normalized to [0,1]."

    def test_read_without_shape(self, static_image_path, static_image_shape):
        """
        Checks if read_image uses the original shape when shape is not provided.
        """
        image = io.read_image(static_image_path, static_image_shape)
        assert image.shape == static_image_shape, "The image shape should match the original shape."
        assert image.max() <= 1.0

    def test_read_raises(self, static_image_path):
        with pytest.raises(AssertionError):
            io.read_image(static_image_path + "_invalid_stuff")
        with pytest.raises(AssertionError):
            io.read_image(static_image_path, (224, 224, 3, 4))  # type: ignore
        with pytest.raises(AssertionError):
            io.read_image(static_image_path, (224,))  # type: ignore


class TestImgProc:
    def test_preprocess_discard(self, static_image):
        """
        Checks that pre-processing can be applied and then discarded correctly.
        """
        # apply preprocessing
        pp_image = proc.pre_process_image(static_image)

        assert pp_image.shape == static_image.shape, "Preprocessed image should keep the same shape."
        assert not np.array_equal(pp_image, static_image)

        assert np.all(pp_image.mean(axis=(0, 1)) != static_image.mean(axis=(0, 1)))
        assert np.all(pp_image.std(axis=(0, 1)) != static_image.std(axis=(0, 1)))

        assert pp_image.min() <= 0
        assert pp_image.max() >= 1

        # discard preprocessing
        discard_pp_image = proc.discard_pre_processing(pp_image)
        assert np.allclose(discard_pp_image, static_image, atol=1e-5), "Discarding pre-processing should recover the original image."

    def test_convert_to_smth(self, static_image):
        """
        Checks conversions to torch.Tensor, NumPy image, and OpenCV image.
        """
        # to tensor
        tensor = proc.to_tensor(static_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 4, "The output tensor should have 4 dimensions (NCHW)."
        assert tensor.shape[0] == 1 and tensor.shape[1] == 3
        # to image
        image = proc.to_image(tensor)
        image_from_squeezed = proc.to_image(tensor.squeeze(0))
        assert isinstance(image, np.ndarray), "The output should be NumPy array."
        assert np.allclose(image, static_image, atol=1e-5)
        assert np.allclose(image, image_from_squeezed, atol=1e-6)
        # to cv
        cv_image = proc.to_cv(image)
        assert np.any(cv_image > 1), "OpenCV image should not be normalized to [0,1]."

    def test_reshape(self, static_image):
        """
        Checks that reshape_image changes dimensions as expected and can revert correctly.
        """
        reshaped = proc.reshape_image(static_image, (224, 224))
        assert reshaped.shape != static_image.shape

        reshaped_back = proc.reshape_image(reshaped, static_image.shape)
        assert reshaped_back.shape == static_image.shape, "Reshaping back should restore the original shape."
