from .img.proc import pre_process_image, to_tensor, discard_pre_processing
from .img.io import read_image
from .utils import two_max_divisors, read_image_net_classes
from .constants import TO_MODEL_SHAPE

import torch
import numpy as np
from matplotlib import pyplot as plt


def display_two_img(init, out, figsize=(4, 4)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes = axes.flatten()
    
    axes[0].imshow(init)
    axes[0].set_title("Initial image")
    axes[0].axis(False)
    
    axes[1].imshow(out)
    axes[1].set_title("After transformation")
    axes[1].axis(False)

    fig.tight_layout()


def classify_images(model, sample_images_path, class_labels) -> tuple[list[np.ndarray], list[str]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #############
    labels = []
    sample_images = []
    for img_path in sample_images_path:
        img = read_image(img_path, TO_MODEL_SHAPE)
        img = pre_process_image(img)
        img_tensor = to_tensor(img).to(device)

        model = model.to(device)
        output = model(img_tensor)
        output_class = class_labels[torch.argmax(output, dim=1).item()]

        sample_images.append(discard_pre_processing(img))
        labels.append(output_class)

    return sample_images, labels

def display_images(sample_images: list[np.ndarray], labels: list[str]):
    x, y = two_max_divisors(len(sample_images))
    fig, axes = plt.subplots(x, y, figsize=(3*y, 3*x))
    axes = axes.flatten()

    for idx, (sample_image, class_label) in enumerate(zip(sample_images, labels)):
        # show image with title = class_label
        axes[idx].axis(False)
        axes[idx].set_title(class_label)
        axes[idx].imshow(sample_image)

    fig.tight_layout()