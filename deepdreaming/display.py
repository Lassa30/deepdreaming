from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from . import img
from .constants import TO_MODEL_SHAPE
from .utils import two_max_divisors


def display_two_img(init, out, figsize=(4, 4)):
    """Display two images side by side for before/after comparison.

    Args:
        init (np.ndarray): Initial image to display on the left.
        out (np.ndarray): Transformed image to display on the right.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (4, 4).

    Note:
        Images should be in RGB format with values in [0, 1] range.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes = axes.flatten()

    axes[0].imshow(init)
    axes[0].set_title("Initial image")
    axes[0].axis(False)

    axes[1].imshow(out)
    axes[1].set_title("After transformation")
    axes[1].axis(False)

    fig.tight_layout()


def classify_images(
    model: torch.nn.Module,
    sample_images_path: list[str],
    class_labels: list[str],
    device: Optional[str] = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Classify a batch of images and return images with their predicted labels.

    Args:
        model (torch.nn.Module): Pre-trained PyTorch model for classification.
        sample_images_path (list[str]): List of file paths to images to classify.
        class_labels (list[str]): List of class labels indexed by model output.
                                 E.g., ImageNet class names where index matches model prediction.
        device (str, optional): Device to run inference on ('cuda' or 'cpu').
                               Auto-detects if None.

    Returns:
        tuple[list[np.ndarray], list[str]]: Tuple containing:
            - List of preprocessed images as numpy arrays
            - List of predicted class labels as strings

    Note:
        Images are automatically resized to TO_MODEL_SHAPE and preprocessed for the model.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sample_images = []
    labels = []

    for img_path in sample_images_path:
        image = img.io.read_image(img_path, TO_MODEL_SHAPE)
        image = img.proc.pre_process_image(image)
        img_tensor = img.proc.to_tensor(image).to(device)

        with torch.no_grad():  # Disable gradient tracking
            output = model(img_tensor)
            output_class = class_labels[output.argmax().item()]

        sample_images.append(img.proc.discard_pre_processing(img))
        labels.append(output_class)

    return sample_images, labels


def display_images(sample_images: list[np.ndarray], labels: list[str]):
    """Display multiple images in a grid layout with labels as titles.

    Args:
        sample_images (list[np.ndarray]): List of images to display as numpy arrays.
        labels (list[str]): List of labels/titles for each image.

    Note:
        Grid dimensions are automatically calculated to best fit all images.
        Images should be in RGB format with values in [0, 1] range.
    """
    x, y = two_max_divisors(len(sample_images))
    fig, axes = plt.subplots(x, y, figsize=(3 * y, 3 * x))
    axes = axes.flatten()

    for idx, (sample_image, class_label) in enumerate(zip(sample_images, labels)):
        # show image with title = class_label
        axes[idx].axis(False)
        axes[idx].set_title(class_label)
        axes[idx].imshow(sample_image)

    fig.tight_layout()
