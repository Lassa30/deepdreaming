import streamlit as st


def show_help():
    """Shows help information about DeepDream parameters."""
    st.markdown("# DeepDream Parameter Guide")

    st.markdown("## 🧠 What is DeepDream?")
    st.markdown(
        """
    DeepDream is a computer vision algorithm that uses neural networks to generate dream-like images. 
    It works by enhancing patterns in images that the neural network has learned to recognize.
    
    The algorithm finds and enhances features in the image that activate specific neurons in the network, 
    creating surreal, psychedelic visualizations. 🎨🔮
    """
    )

    st.markdown("## 📸 Image Processing Note")
    st.markdown(
        """
    ⚠️ **Important**: All input images are automatically resized to (224, 224, 3) for processing to improve performance.
    After processing, the resulting dream image is upscaled back to the original dimensions.

    This resizing helps the neural network process images more efficiently, but some fine details may be lost. 
    For best results:
    - Use images with clear, prominent features
    - Consider that very detailed textures might be simplified
    - Final output quality will be better for smaller original images
    """
    )

    st.markdown("## ⚙️ General Parameters")
    st.markdown(
        """
    - **Learning Rate** 🚀: Controls the step size of optimization. Higher values lead to more dramatic changes but may cause instability.
      - *Range*: 0.01 to 0.2
      - *Default*: 0.09
      - *Effect*: Higher values create more dramatic, stylized results but may introduce artifacts.

    - **Number of Iterations** 🔄: How many optimization steps to perform per pyramid layer. More iterations give stronger effects.
      - *Range*: 5 to 20
      - *Default*: 10
      - *Effect*: More iterations create stronger dream effects but take longer to process.
    """
    )

    st.markdown("## 📊 Gradient Normalization")
    st.markdown(
        """
    - **Gradient Normalization** ⚖️: Normalizes the gradients to unit norm for more stable optimization.
      - *Default*: True
      - *Effect*: When enabled, helps prevent extreme changes and stabilizes the dreaming process.
    """
    )

    st.markdown("## 🌊 Smoothing Parameters")
    st.markdown(
        """
    - **Gradient Smoothing** 🧼: Method used to smooth gradients, which helps reduce high-frequency artifacts.
      - *Options*: No Smoothing, Box Smoothing, Gaussian Smoothing
      - *Default*: Gaussian Smoothing
      - *Effect*: Smoothing reduces noise and creates more coherent patterns.

    - **Smoothing Kernel Size** 📏: Size of the smoothing kernel. Must be an odd integer.
      - *Range*: 3 to 9
      - *Default*: 3
      - *Effect*: Larger sizes produce more blurring and smoother results.

    ### ☁️ Gaussian Smoothing Parameters
    These parameters only apply when Gaussian Smoothing is selected:

    - **Gaussian Sigmas** 🔍: Standard deviation of the Gaussian distribution. Controls the amount of blurring.
      - *Options*: 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 8
      - *Default*: 1
      - *Effect*: Higher values create more blurring. Multiple values can be selected to blend different levels of smoothing.

    - **Gaussian Blending Weights** ⚖️: Weights for blending multiple Gaussian kernels.
      - *Options*: 1, 5, 10, 25, 50, 100, 200, 400, 500, 1000
      - *Default*: 100
      - *Effect*: Controls how much each selected sigma contributes to the final result.
    """
    )

    st.markdown("## 🔧 Advanced Parameters")
    st.markdown(
        """
    - **Pyramid Layers** 🏔️: Number of pyramid layers for multi-scale processing.
      - *Range*: 1 to 10
      - *Default*: 5
      - *Effect*: More layers create more detailed and coherent dream patterns but increase processing time.

    - **Pyramid Ratio** 📉: Scale ratio between pyramid layers (0 < ratio < 1).
      - *Range*: 0.25 to 0.95
      - *Default*: 0.67
      - *Effect*: Controls how quickly the image size changes between pyramid layers.

    - **Shift Size** 🔄: Maximum pixel shift for random augmentation.
      - *Range*: 0 to 128
      - *Default*: 32
      - *Effect*: Larger values create more diverse patterns by shifting the image during processing.
    """
    )

    st.markdown("## 💡 Tips for Best Results")
    st.markdown(
        """
    1. 🌱 Start with default parameters and adjust gradually.
    2. 🖼️ Images with clear patterns and good contrast often produce the best results.
    3. 🔄 Try different layer combinations for different styles of dreams.
    4. 🌟 For subtle effects, use lower learning rates and fewer iterations.
    5. 🌈 For psychedelic effects, increase learning rate and iterations.
    """
    )

    # Add model and layer selection info
    st.markdown("## 🧩 Model and Layer Selection")
    st.markdown(
        """
    - **Model Selection** 🤖: Choose the neural network model for dream generation.
      - *VGG16* 🏛️: A classic architecture with 16 layers, good for consistent, recognizable patterns.
      - *ResNet50* 🏗️: A deeper network with 50 layers that can detect more complex features.
      - *Inception* 🌀: Google's Inception v3 architecture with mixed path blocks, great for capturing diverse features.

    - **Layer Selection** 📚: Individual layers can now be selected using checkboxes!
      - *ReLU layers* ✨: Typically produce more stable and aesthetically pleasing patterns.
      - *Conv layers* 🔄: Can produce more detailed but sometimes chaotic patterns.
      
    - **Layer Location Effects**:
      - *Early layers* (lower numbers): Focus on basic features like edges, textures, and colors.
      - *Middle layers*: Detect patterns like textures, parts of objects.
      - *Deeper layers* (higher numbers): Recognize complex objects and abstract concepts.

    For best results, try selecting:
    - Single layers to see their specific "style"
    - Multiple similar layers for a consistent effect
    - Mix of early, middle, and deep layers for diverse patterns
    """
    )
