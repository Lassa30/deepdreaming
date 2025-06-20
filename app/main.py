import streamlit as st
import sys
import cv2 as cv

sys.path.append("../deepdreaming")

from app.help_page import show_help
from app.ui import (
    display_title,
    get_image_input,
    model_selection_ui,
    layer_selector_ui,
    config_ui,
    display_result,
    create_download_button,
)
from app.model_handler import get_model, get_model_info
from app.processor import process_image, generate_filename, prepare_image_for_download


def main():
    """Main function to run the Streamlit app."""
    display_title()

    # Add a help button in the top menu and a back button when in help mode
    show_help_page = st.sidebar.button("📘 Help / Parameter Guide")
    if show_help_page:
        show_help()
        if st.sidebar.button("🔙 Go Back & Start Dreaming"):
            st.empty()  # Clear the help content
    else:
        # Get user input image
        image, _ = get_image_input()

        # Configure DeepDream parameters in sidebar
        st.sidebar.header("DeepDream Configuration")

        # Model and layer selection
        st.sidebar.subheader("🧠 Model & Layer Selection")
        model_name = model_selection_ui()
        model = get_model(model_name)

        # Layer selection UI directly in the sidebar
        st.sidebar.markdown("### 🧩 Select Layers: ✨")
        st.sidebar.markdown("Choose which layers to target for feature visualization:")

        with st.sidebar:
            layers = layer_selector_ui(model_name)

            # Show the selected layers count
            st.success(f"✅ {len(layers)} layers selected")
            with st.expander("Show selected layer IDs", expanded=False):
                st.code(str(layers))

        # Get configuration from UI
        config = config_ui()

        # Process image when requested
        if image and layers:  # Only proceed if image is not None and at least one layer is selected
            if st.button("✨ Dream!"):
                try:
                    with st.spinner(f"🧠 Dreaming with {model_name}... Using {len(layers)} layers"):
                        initial_size = image.size
                        dreamed_image = process_image(image, config, model, layers, model_name)
                        dreamed_image = cv.resize(dreamed_image, initial_size)

                    # Display result
                    display_result(dreamed_image, model_name, len(layers))

                    # Prepare download
                    model_info = get_model_info(model_name, len(layers))
                    filename = generate_filename(model_info, config)
                    image_bytes = prepare_image_for_download(dreamed_image, initial_size)

                    # Create download button
                    create_download_button(image_bytes, filename)

                except ValueError as e:
                    st.error(str(e))
        elif image and not layers:
            st.warning("⚠️ Please select at least one layer to proceed.")


if __name__ == "__main__":
    main()
