import os
import tempfile
import time
from io import BytesIO
from typing import Tuple

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from app.model_handler import get_model_input_size
from deepdreaming.config import DreamConfig, GradSmoothingMode
from deepdreaming.deepdream import DeepDream
from deepdreaming.img import io as img_io


def process_image(
    image: Image.Image, config: DreamConfig, model: torch.nn.Module, layers: list[str], model_name: str
) -> np.ndarray:
    """Process an image using DeepDream.

    Args:
        image (PIL.Image): Input image
        config (DreamConfig): Configuration parameters
        model (torch.nn.Module): Neural network model
        layers (list[str]): Layer identifiers to target
        model_name (str): Model name for input size determination

    Returns:
        np.ndarray: Processed dreamed image
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_file_path = temp_file.name

    # Get appropriate input shape for the model
    input_shape = get_model_input_size(model_name)

    # Load the image using img_io.read_image
    image_np = img_io.read_image(temp_file_path, input_shape)

    # Clean up the temporary file
    os.remove(temp_file_path)

    try:
        # Process the image
        deep_dream = DeepDream(model, layers)
        dreamed_image = deep_dream.dream(image_np, config=config)
        return dreamed_image
    except Exception as e:
        if isinstance(e, RuntimeError) and model_name == "Inception":
            raise ValueError(
                "Error processing image with Inception model.\n\n"
                "The Inception v3 model requires input images of sufficient size (minimum 75x75 pixels). "
                "Your image has likely become too small during pyramid processing.\n\n"
                "To fix this issue, you can:\n"
                "1. Reduce the number of pyramid layers (try 3-4 instead of 5+)\n"
                "2. Increase the pyramid ratio (try 0.8-0.9 instead of 0.67)\n"
                "3. Use another model like VGG16 or ResNet50 which can handle smaller images\n\n"
                "Technical details: Inception v3 has specific architecture constraints that "
                "require larger input dimensions than other models."
            )
        else:
            raise e


def generate_filename(model_info: str, config: DreamConfig) -> str:
    """Generate a unique, descriptive filename for the dream image.

    Args:
        model_info (str): Model information
        config (DreamConfig): Configuration used

    Returns:
        str: Generated filename with timestamp
    """
    timestamp = int(time.time())
    filename = f"{model_info}_lr{config.learning_rate}_iter{config.num_iter}"

    if config.gradient_norm:
        filename += "_gradnorm"
    if config.grad_smoothing == GradSmoothingMode.GaussianSmoothing:
        filename += "_gauss"
    elif config.grad_smoothing == GradSmoothingMode.BoxSmoothing:
        filename += "_box"

    return f"{filename}_{timestamp}.jpg"


def prepare_image_for_download(dreamed_image: np.ndarray, initial_size: Tuple[int, int]) -> BytesIO:
    """Resize and prepare the processed image for download.

    Args:
        dreamed_image (np.ndarray): Processed image array
        initial_size (tuple): Original image dimensions (width, height)

    Returns:
        BytesIO: Image bytes ready for download
    """
    # Resize back to original dimensions
    resized_image = cv.resize(dreamed_image, initial_size)

    # Convert to PIL Image and save to bytes
    image_bytes = BytesIO()
    Image.fromarray((resized_image * 255).astype(np.uint8)).save(image_bytes, format="JPEG")

    return image_bytes
