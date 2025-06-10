import torch
from torchvision.transforms import ToTensor
import cv2 as cv
import numpy as np

from ..constants import IMAGE_NET_MEAN, IMAGE_NET_STD


def pre_process_image(image):
    image = (image - IMAGE_NET_MEAN) / IMAGE_NET_STD
    return image


def discard_pre_processing(image):
    image = image * IMAGE_NET_STD + IMAGE_NET_MEAN
    return image


def to_tensor(image) -> torch.Tensor:
    assert image is not None, "Image is None"
    image_tensor = ToTensor()(image).unsqueeze(0)
    return image_tensor


def to_image(tensor):
    image = tensor.squeeze(0).permute(1, 2, 0)
    return image.detach().cpu().numpy()


def to_cv(image):
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image

def reshape_image(image, shape):
    image = cv.resize(image, (shape[1], shape[0]))
    return image