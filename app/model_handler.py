import torch
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights, inception_v3, Inception_V3_Weights
from typing import Tuple

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(model_name: str) -> torch.nn.Module:
    """Load and return the specified pre-trained model.

    Args:
        model_name (str): Name of the model to load (VGG16, ResNet50, Inception)

    Returns:
        torch.nn.Module: The loaded model in eval mode on the appropriate device
    """
    if model_name == "VGG16":
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
    elif model_name == "ResNet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    elif model_name == "Inception":
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).eval()
    else:
        # Default to VGG16 if unknown model name
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()

    # Move model to appropriate device
    return model.to(DEVICE)


def get_model_input_size(model_name: str) -> Tuple[int, int, int]:
    """Get the appropriate input image size for the specified model.

    Args:
        model_name (str): Name of the model

    Returns:
        tuple: (height, width, channels) for input images
    """
    if model_name == "Inception":
        return (299, 299, 3)
    else:
        return (224, 224, 3)


def get_model_info(model_name: str, num_layers: int) -> str:
    """Generate model info string for filenames and display.

    Args:
        model_name (str): Name of the model
        num_layers (int): Number of selected layers

    Returns:
        str: Formatted model information
    """
    return f"{model_name}_{num_layers}layers"
