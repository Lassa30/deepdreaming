from typing import Optional, Tuple

import streamlit as st
from PIL import Image

from app.layer_configs import DEFAULT_SELECTED_LAYERS, get_layers_by_type
from deepdreaming.config import DreamConfig, GradSmoothingMode


def display_title():
    """Display the application title."""
    st.title("✨🔮 DeepDreaming 🧠✨")


def get_image_input() -> Tuple[Optional[Image.Image], str]:
    """Handle image input from upload or webcam.

    Returns:
        tuple: (image object, image source description)
    """
    image = None
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

    return image, image_source


def model_selection_ui() -> str:
    """UI for model selection.

    Returns:
        str: Selected model name
    """
    return st.sidebar.radio(
        "Select Neural Network Model:",
        ["VGG16", "ResNet50", "Inception"],
        help="Choose which neural network model to use for generating the DeepDream. Each model extracts different patterns from images.",
    )


def layer_selector_ui(model_name: str):
    """UI component for selecting layers.

    Args:
        model_name (str): Name of the selected model

    Returns:
        list: Selected layer identifiers
    """
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


def config_ui() -> DreamConfig:
    """UI for DeepDream configuration parameters.

    Returns:
        DreamConfig: Configuration object with user-selected parameters
    """
    # -- General --
    st.sidebar.subheader("⚙️ General")
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        0.01,
        0.2,
        0.09,
        0.01,
        help="Controls the intensity of the dream effect. Higher values create more dramatic, stylized results but may introduce artifacts.",
    )
    num_iter = st.sidebar.slider(
        "Number of Iterations",
        5,
        20,
        10,
        1,
        help="Sets how many times to enhance patterns. More iterations create stronger dream effects but take longer to process.",
    )

    # -- Norm --
    st.sidebar.subheader("📊 Gradient Normalization")
    gradient_norm = st.sidebar.checkbox(
        "Gradient Normalization",
        True,
        help="Stabilizes the dreaming process. When enabled, helps prevent extreme changes and produces more balanced results.",
    )

    # -- Smoothing --
    st.sidebar.subheader("🌊 Smoothing")
    grad_smoothing = st.sidebar.selectbox(
        "Gradient Smoothing",
        (GradSmoothingMode.Disable, GradSmoothingMode.BoxSmoothing, GradSmoothingMode.GaussianSmoothing),
        index=2,
        help="Determines how to smooth patterns. Gaussian creates more natural results, Box is simpler, None can be more chaotic.",
    )

    # params for both smoothers
    grad_smoothing_kernel_size = st.sidebar.slider(
        "Smoothing Kernel Size",
        3,
        9,
        3,
        2,
        help="Controls the blurring strength. Larger values produce smoother, more blended patterns with less fine detail.",
    )

    # gaussian ONLY
    if grad_smoothing == GradSmoothingMode.GaussianSmoothing:
        st.sidebar.markdown("**🌫️ Gaussian Smoothing Parameters**")
        grad_smoothing_gaussian_sigmas = tuple(
            st.sidebar.multiselect(
                "Gaussian Sigmas",
                [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 8],
                [1],
                help="Determines blurring intensity. Higher values create more blurring; multiple values blend different smoothing levels.",
            )
        )
        grad_smoothing_gaussian_blending_weights = tuple(
            st.sidebar.multiselect(
                "Gaussian Blending Weights",
                [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 500.0, 1000.0],
                [1],
                help="Controls how much each sigma contributes. Higher weights for a sigma increase its influence on the final result.",
            )
        )
    else:
        grad_smoothing_gaussian_sigmas = DreamConfig.grad_smoothing_gaussian_sigmas
        grad_smoothing_gaussian_blending_weights = tuple(
            1 / len(grad_smoothing_gaussian_sigmas) for _ in range(len(grad_smoothing_gaussian_sigmas))
        )

    # -- Other --
    st.sidebar.subheader("🔧 Other tricks...")
    pyramid_layers = st.sidebar.slider(
        "Pyramid Layers",
        1,
        10,
        5,
        1,
        help="Controls detail coherence. More layers create more detailed and coherent dream patterns but take longer to process.",
    )
    pyramid_ratio = st.sidebar.slider(
        "Pyramid Ratio",
        0.25,
        0.95,
        2 / 3,
        0.05,
        help="Controls scale changes between processing steps. Higher values preserve more details but may produce less dramatic effects.",
    )
    shift_size = st.sidebar.slider(
        "Shift Size",
        0,
        128,
        32,
        16,
        help="Adds variety to patterns. Larger values create more diverse patterns by shifting the image during processing.",
    )

    return DreamConfig(
        learning_rate=learning_rate,
        num_iter=num_iter,
        gradient_norm=gradient_norm,
        grad_smoothing=grad_smoothing,
        grad_smoothing_kernel_size=grad_smoothing_kernel_size,
        grad_smoothing_gaussian_sigmas=grad_smoothing_gaussian_sigmas,
        grad_smoothing_gaussian_blending_weights=grad_smoothing_gaussian_blending_weights,
        pyramid_layers=pyramid_layers,
        pyramid_ratio=pyramid_ratio,
        shift_size=shift_size,
    )


def display_result(dreamed_image, model_name, num_layers):
    """Display the processed image with appropriate caption.

    Args:
        dreamed_image (np.ndarray): Processed image array
        model_name (str): Name of the model used
        num_layers (int): Number of layers used
    """
    st.image(dreamed_image, caption=f"Dreamed Image ({model_name} with {num_layers} layers)")


def create_download_button(image_bytes, filename):
    """Create a download button for the processed image.

    Args:
        image_bytes (BytesIO): Image data
        filename (str): Filename to use for download
    """
    st.download_button(
        label="💾 Save Image",
        data=image_bytes.getvalue(),
        file_name=filename,
        mime="image/jpeg",
        help="Download the generated dream image to your device.",
    )
