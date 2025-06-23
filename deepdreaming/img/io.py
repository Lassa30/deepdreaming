import os
from typing import Optional

import cv2 as cv
import numpy as np


def read_image(image_path: str, target_size: Optional[tuple[int, int, int]] = None) -> np.ndarray:
    """Read and preprocess image file for DeepDream processing.

    Args:
        image_path (str): Path to the image file to read.
        target_size (tuple, optional): Target size as (height, width, channels).
                                     If provided, image will be resized to (height, width).
                                     The channels dimension is ignored during resizing.

    Returns:
        np.ndarray: Preprocessed image as np.float32 array with shape (H, W, C)
                   and pixel values normalized to [0, 1] range. Color format is RGB.

    Raises:
        AssertionError: If image_path does not exist.
        Exception: If target_size is not a tuple.

    Note:
        This function automatically converts BGR (OpenCV format) to RGB and normalizes
        pixel values from [0, 255] to [0, 1] range. Output format is compatible with
        DeepDream.dream() method requirements.

    Examples:
        >>> image = read_image("photo.jpg")  # Load image at original size
        >>> image = read_image("photo.jpg", (224, 224, 3))  # Resize to 224x224
    """
    assert os.path.exists(image_path), f"Invalid image path: {image_path}\nCurrent directory is: {os.getcwd()}"

    image = cv.imread(image_path)[:, :, ::-1]  # BGR to RGB

    if target_size is not None:
        assert isinstance(target_size, tuple), "Tuple is expected as a target_size argument type."
        assert len(target_size) in [2, 3], "Please provide from 2 to 3 values as a target_shape."

        dim1, dim2 = target_size[:2]
        image = cv.resize(image, (dim2, dim1))

    return image.astype(np.float32) / 255.0
