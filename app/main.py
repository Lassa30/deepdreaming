import streamlit as st

# standard libraries
import os
import time
import sys
import tempfile

# Image processing
from PIL import Image
from io import BytesIO
import cv2 as cv
import os
import tempfile

import numpy as np
import torch

from deepdreaming.deepdream import DeepDream
from deepdreaming.config import DreamConfig, GradSmoothingMode
from deepdreaming.img import io as img_io
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights, inception_v3, Inception_V3_Weights

# Import the separated modules
from app.help_page import show_help
from app.layer_configs import get_layers_by_type, DEFAULT_SELECTED_LAYERS

# --- Constants ---
sys.path.append("../deepdreaming")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DeepDream modules
from deepdreaming.deepdream import DeepDream
from deepdreaming.config import DreamConfig, GradSmoothingMode
from deepdreaming.img import io as img_io
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights


# --- Functions ---
def generate_filename(model_info, config):
    """Generates a filename based on the model and DreamConfig."""
    timestamp = int(time.time())  # Get current timestamp in integer format
    filename = f"{model_info}_lr{config.learning_rate}_iter{config.num_iter}"
    if config.gradient_norm:
        filename += "_gradnorm"
    if config.grad_smoothing == GradSmoothingMode.GaussianSmoothing:
        filename += "_gauss"
    elif config.grad_smoothing == GradSmoothingMode.BoxSmoothing:
        filename += "_box"
    return f"{filename}_{timestamp}.jpg"


def process_image(image, config, model, layers):
    """Processes the image using DeepDream."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file.name)
        temp_file_path = temp_file.name

    # Load the image using img_io.read_image
    image_np = img_io.read_image(temp_file_path, (224, 224, 3))
    # Clean up the temporary file
    os.remove(temp_file_path)

    deep_dream = DeepDream(model, layers)
    dreamed_image = deep_dream.dream(image_np, config=config)
    return dreamed_image


def get_model(model_name):
    """Get the selected model based on user selection."""
    # Load the selected model
    if model_name == "VGG16":
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
    elif model_name == "ResNet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    elif model_name == "Inception":
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).eval()
    else:
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()

    # Move model to appropriate device
    return model.to(DEVICE)


def layer_selector_ui(model_name):
    """UI component for selecting layers."""
    selected_layers = []

    # Get all layers for this model
    all_layers = get_layers_by_type(model_name.lower())
    if not all_layers:
        st.warning(f"No layer information available for {model_name}")
        return DEFAULT_SELECTED_LAYERS.get(model_name.lower(), [])

    # Create expanders for ReLU and Conv2d layers
    with st.expander("🔍 ReLU Layers (Recommended)", expanded=True):
        st.info("ReLU layers typically produce more stable and visually appealing patterns.")

        # Group layers by their position in the network for more organized selection
        relu_layers = get_layers_by_type(model_name.lower(), "ReLU")

        # Create columns for better layout
        cols = st.columns(2)

        # Create checkboxes for each ReLU layer
        for i, (layer_id, info) in enumerate(relu_layers.items()):
            col_idx = i % 2
            default_checked = layer_id in DEFAULT_SELECTED_LAYERS.get(model_name.lower(), [])
            if cols[col_idx].checkbox(
                f"{layer_id}: {info['description']}", value=default_checked, key=f"relu_{layer_id}"
            ):
                selected_layers.append(layer_id)

    with st.expander("⚙️ Conv2d Layers (Optional)", expanded=False):
        st.warning("Conv layers can produce more detailed but sometimes chaotic patterns.")

        # Get all Conv2d layers
        conv_layers = get_layers_by_type(model_name.lower(), "Conv2d")

        # Create columns for better layout
        cols = st.columns(2)

        # Create checkboxes for each Conv2d layer
        for i, (layer_id, info) in enumerate(conv_layers.items()):
            col_idx = i % 2
            default_checked = layer_id in DEFAULT_SELECTED_LAYERS.get(model_name.lower(), [])
            if cols[col_idx].checkbox(
                f"{layer_id}: {info['description']}",
                value=False,  # Default to unchecked for Conv layers
                key=f"conv_{layer_id}",
            ):
                selected_layers.append(layer_id)

    # If no layers are selected, use defaults
    if not selected_layers:
        st.warning("No layers selected! Using default selection.")
        return DEFAULT_SELECTED_LAYERS.get(model_name.lower(), [])

    return selected_layers


def main():
    """Main function to run the Streamlit app."""
    st.title("✨🔮 DeepDreaming 🧠✨")

    # Add a help button in the top menu and a back button when in help mode
    show_help_page = st.sidebar.button("📘 Help / Parameter Guide")
    if show_help_page:
        show_help()
        if st.sidebar.button("🔙 Go Back & Start Dreaming"):
            show_help_page = False
            st.empty()  # Clear the help content
    else:
        # --- Image Input ---
        image = None  # Initialize image to None
        image_source = st.radio("Select Image Source:", ("Upload", "Webcam"))

        if image_source == "Upload":
            uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")
        else:
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                image = Image.open(camera_image)
                st.image(image, caption="Webcam Image")

        # --- Config Panel ---
        st.sidebar.header("DeepDream Configuration")

        # -- Model and Layer Selection --
        st.sidebar.subheader("🧠 Model & Layer Selection")

        # Select the model
        model_name = st.sidebar.radio(
            "Select Neural Network Model:",
            ["VGG16", "ResNet50"],
            help="Choose which neural network model to use for generating the DeepDream.",
        )

        # Get the model
        model = get_model(model_name)

        # Layer selection UI
        st.sidebar.markdown("### Select Layers:")
        st.sidebar.markdown("Choose which layers to target for feature visualization:")

        # Create the layer selection UI in the main area for more space
        with st.expander("🔍 Layer Selection", expanded=True):
            layers = layer_selector_ui(model_name)

            # Show the selected layers
            if layers:
                st.success(f"✅ {len(layers)} layers selected")
                with st.expander("Show selected layer IDs", expanded=False):
                    st.code(str(layers))

        # -- General --
        st.sidebar.subheader("⚙️ General")
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.2, 0.09, 0.01)
        num_iter = st.sidebar.slider("Number of Iterations", 5, 20, 10, 1)

        # -- Norm --
        st.sidebar.subheader("📊 Gradient Normalization")
        gradient_norm = st.sidebar.checkbox("Gradient Normalization", False)

        # -- Smoothing --
        st.sidebar.subheader("🌊 Smoothing")
        grad_smoothing = st.sidebar.selectbox(
            "Gradient Smoothing",
            (GradSmoothingMode.Disable, GradSmoothingMode.BoxSmoothing, GradSmoothingMode.GaussianSmoothing),
            index=2,
        )

        # params for both smoothers
        grad_smoothing_kernel_size = st.sidebar.slider("Smoothing Kernel Size", 3, 9, 3, 2)

        # gaussian ONLY
        if grad_smoothing == GradSmoothingMode.GaussianSmoothing:
            st.sidebar.markdown("**🌫️ Gaussian Smoothing Parameters**")
            grad_smoothing_gaussian_sigmas = tuple(
                st.sidebar.multiselect(
                    "Gaussian Sigmas",
                    [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 8],
                    [1],
                )
            )
            grad_smoothing_gaussian_blending_weights = tuple(
                st.sidebar.multiselect(
                    "Gaussian Blending Weights",
                    [1, 5, 10, 25, 50, 100, 200, 400, 500, 1000],
                    [100],
                )
            )
            if len(grad_smoothing_gaussian_blending_weights) == 0:
                grad_smoothing_gaussian_blending_weights = None
        else:
            grad_smoothing_gaussian_sigmas = 1
            grad_smoothing_gaussian_blending_weights = None

        # -- Other --
        st.sidebar.subheader("🔧 Other tricks...")
        pyramid_layers = st.sidebar.slider("Pyramid Layers", 1, 10, 5, 1)
        pyramid_ratio = st.sidebar.slider("Pyramid Ratio", 0.25, 0.95, 2 / 3, 0.05)
        shift_size = st.sidebar.slider("Shift Size", 0, 128, 32, 16)

        # --- Process Image ---
        if image and layers:  # Only proceed if image is not None and at least one layer is selected
            config = DreamConfig(
                learning_rate=learning_rate,
                num_iter=num_iter,
                gradient_norm=gradient_norm,
                grad_smoothing=grad_smoothing,
                grad_smoothing_kernel_size=grad_smoothing_kernel_size,
                grad_smoothing_gaussian_sigmas=(
                    grad_smoothing_gaussian_sigmas if grad_smoothing == GradSmoothingMode.GaussianSmoothing else 1
                ),
                grad_smoothing_gaussian_blending_weights=(
                    grad_smoothing_gaussian_blending_weights
                    if grad_smoothing == GradSmoothingMode.GaussianSmoothing
                    else None
                ),
                pyramid_layers=pyramid_layers,
                pyramid_ratio=pyramid_ratio,
                shift_size=shift_size,
            )

            if st.button("✨ Dream!"):
                with st.spinner(f"🧠 Dreaming with {model_name}... Using {len(layers)} layers"):
                    initial_size = image.size
                    dreamed_image = process_image(image, config, model, layers)
                dreamed_image = cv.resize(dreamed_image, initial_size)
                st.image(dreamed_image, caption=f"Dreamed Image ({model_name} with {len(layers)} layers)")

                # --- Save Image ---
                # Use model name and layer count in filename
                model_info = f"{model_name}_{len(layers)}layers"
                filename = generate_filename(model_info, config)
                image_bytes = BytesIO()
                Image.fromarray((dreamed_image * 255).astype(np.uint8)).save(image_bytes, format="JPEG")
                st.download_button(
                    label="💾 Save Image",
                    data=image_bytes.getvalue(),
                    file_name=filename,
                    mime="image/jpeg",
                )
        elif image and not layers:
            st.warning("⚠️ Please select at least one layer to proceed.")


if __name__ == "__main__":
    main()
