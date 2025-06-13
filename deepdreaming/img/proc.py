import torch
from torchvision.transforms import ToTensor
import cv2 as cv
import numpy as np

from ..constants import IMAGE_NET_MEAN, IMAGE_NET_STD
from ..utils import return_none
print(IMAGE_NET_MEAN)


@return_none
def pre_process_image(image):
    """Normalize image using ImageNet statistics for neural network input.
    
    Args:
        image (np.ndarray): Input image with values in [0, 1] range and shape (H, W, C).
    
    Returns:
        np.ndarray: Normalized image with ImageNet mean subtracted and scaled by ImageNet std.
                   Values will be approximately in [-2, 2] range after normalization.
    """
    image = (image - IMAGE_NET_MEAN) / IMAGE_NET_STD
    return image


@return_none
def discard_pre_processing(image):
    """Reverse ImageNet normalization to restore original pixel value range.
    
    Args:
        image (np.ndarray): Normalized image from pre_process_image function.
    
    Returns:
        np.ndarray: Denormalized image with values restored to [0, 1] range.
    """
    image = image * IMAGE_NET_STD + IMAGE_NET_MEAN
    return image


@return_none
def to_tensor(image) -> torch.Tensor:
    """Convert numpy image to PyTorch tensor with batch dimension and move to device.
    
    Args:
        image (np.ndarray): Input image with shape (H, W, C) and values in [0, 1].
    
    Returns:
        torch.Tensor: Tensor with shape (1, C, H, W) on appropriate device (GPU if available).
                     Channel order is converted from HWC to CHW format.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    return image_tensor


@return_none
def to_image(tensor):
    """Convert PyTorch tensor back to numpy image format.
    
    Args:
        tensor (torch.Tensor): Input tensor with shape (1, C, H, W) or (C, H, W).
    
    Returns:
        np.ndarray: Image array with shape (H, W, C) on CPU as numpy array.
                   Channel order is converted from CHW to HWC format.
    """
    image = tensor.squeeze(0).permute(1, 2, 0)
    return image.detach().cpu().numpy()


@return_none
def to_cv(image):
    """Convert image to OpenCV format for saving or display.
    
    Args:
        image (np.ndarray): Input image with values in [0, 1] and RGB channel order.
    
    Returns:
        np.ndarray: Image converted to BGR format with uint8 values in [0, 255] range,
                   ready for OpenCV operations like cv.imwrite().
    """
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image


@return_none
def reshape_image(image, shape):
    """Resize image to specified dimensions using OpenCV.
    
    Args:
        image (np.ndarray): Input image to resize.
        shape (tuple): Target shape as (height, width). Aspect ratio may not be preserved.
    
    Returns:
        np.ndarray: Resized image with new dimensions. Uses linear interpolation.
    """
    image = cv.resize(image, (shape[1], shape[0]))
    return image
