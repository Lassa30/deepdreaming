import os
import cv2 as cv
import numpy as np
from torchvision.transforms import ToTensor

from ..constants import IMAGE_NET_MEAN, IMAGE_NET_STD


def read_image(image_path, target_size=None):
    assert os.path.exists(image_path), f"Invalid image path: {image_path}\nCurrent directory is: {os.getcwd()}"

    image = cv.imread(image_path)[:, :, ::-1]  # BGR to RGB

    if target_size is not None:
        if isinstance(target_size, tuple):
            dim1, dim2, _ = target_size
            image = cv.resize(image, (dim2, dim1))
        else:
            raise Exception("Tuple is expected as a target_size argument type.")

    return image.astype(np.float32) / 255.0
