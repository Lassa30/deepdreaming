## DeepDreaming

A PyTorch implementation of the [DeepDream algorithm](https://en.wikipedia.org/wiki/DeepDream).

<div style="display: flex; flex-direction: column; align-items: center; margin: 20px 0;">
  <div style="display: flex; flex-direction: row; align-items: center; justify-content: center; width: 100%;">
    <img src="assets/tree.png" width="400" height="400" style="margin-right: 10px;"/>
    <div style="display: flex; flex-direction: column; align-items: center; margin: 0 10px;">
      <div style="font-size: 18px; font-weight: bold; margin-bottom: 5px;">Dreaming...</div>
      <div style="font-size: 24px;">➡️</div>
    </div>
    <img src="assets/tree_deepdream.png" width="400" height="400" style="margin-left: 10px;"/>
  </div>
</div>

DeepDream is a fascinating computer vision algorithm that creates dreamlike hallucinogenic appearances in images. While there are many great implementations available, I wanted to create something more accessible and user-friendly.

This project offers two key features:
1. A simple web interface built with Streamlit where anyone can experiment with DeepDream without writing code
2. A cleaner implementation of layer access in PyTorch using forward hooks rather than hardcoded model redefinition. With this approach you could use any pretrained model not only the couple of hardcoded!

## Getting Started

There are two ways to experience `DeepDreaming`:
1. Play with the Streamlit demo
2. Dream deeper with Jupyter Notebooks (also available on Kaggle)

Each approach has detailed guides below. Remember that DeepDream doesn't have a one-size-fits-all configuration - you'll get the best results by experimenting with different settings for each image!

## Web Demo

[IMAGE PLACEHOLDER FOR DEMO INTERFACE]

**TASK: create a nice badge for this streamlit app link https://deepdreaming.streamlit.app/**


The Streamlit app provides an intuitive interface for experimenting with DeepDream. Simply upload an image, adjust the parameters, and watch the algorithm transform your image into a dream-like visualization.

**Note:** The web demo may run slowly, especially on shared servers. For performance reasons, your input images will be downscaled to:
- (224, 224) pixels for VGG and ResNet models
- (299, 299) pixels for Inception model

The default settings work well for most images, so you can get started right away. If you're curious about what each parameter does, just click the help button (❔) next to each control for a detailed explanation!

Help page is also present. Check this out to gain more intuition about parameters.

[HELP PAGE PLACEHOLDER]

## Notebooks

### GitHub Notebooks

The notebooks in this repository come pre-executed, making them the quickest way to see DeepDream in action. You can view all the outputs directly in your browser without running any code yourself. They're perfect for getting a quick understanding of what's possible with this implementation.

### Kaggle Notebooks

The Kaggle notebooks serve as interactive tutorials that walk you through the implementation step by step. They contain the same content as the GitHub notebooks but with more detailed explanations and no imported code. If you want to understand how everything works under the hood, these are your best starting point!

### Links

1. _[DeepDream Starter](deepdream-starter.ipynb)_ - No Image Pyramid or Gradient Smoothing. Gradient ascent is _all_ you need to start dreaming...

    **[STARTER IMAGES PLACEHOLDER]**

<div align="center">
  <a href="https://www.kaggle.com/code/vladislavlassa/deepdream-starter">
    <img src="https://img.shields.io/badge/Kaggle-DeepDream%20Starter-blue?logo=kaggle"
    alt="DeepDream Starter"/>
  </a>
  <a href="deepdream-starter.ipynb">
    <img src="https://img.shields.io/badge/GitHub-DeepDream%20Starter-green?logo=github"
    alt="DeepDream Starter GitHub"/>
  </a>
</div>

2. _[DeepDream Tricks](deepdream-tricks.ipynb)_ - Playing with different settings from gradient smoothing to image pyramid!

    **[TRICKY IMAGES PLACEHOLDER]**

<div align="center">
  <a href="https://www.kaggle.com/code/vladislavlassa/deepdream-tricks">
    <img src="https://img.shields.io/badge/Kaggle-DeepDream%20Tricks-blue?logo=kaggle"
    alt="DeepDream Tricks"/>
  </a>
  <a href="deepdream-tricks.ipynb">
    <img src="https://img.shields.io/badge/GitHub-DeepDream%20Tricks-green?logo=github"
    alt="DeepDream Tricks GitHub"/>
  </a>
</div>

3. _[DeepDream Guided](deepdream-guided.ipynb)_ - Discover how we can transfer features from one image to another.

    **[GUIDED IMAGES PLACEHOLDER]**

<div align="center">
  <a href="https://www.kaggle.com/code/vladislavlassa/deepdream-guided">
    <img src="https://img.shields.io/badge/Kaggle-DeepDream%20Guided-blue?logo=kaggle"
    alt="DeepDream Guided"/>
  </a>
  <a href="deepdream-guided.ipynb">
    <img src="https://img.shields.io/badge/GitHub-DeepDream%20Guided-green?logo=github"
    alt="DeepDream Guided GitHub"/>
  </a>
</div>

## Run Locally

Running the Streamlit app locally is recommended for better performance, especially if you have a GPU. Here's how to get started:

```bash
# 1. Clone the repository
git clone https://github.com/Lassa30/deepdreaming.git
cd deepdreaming

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app/main.py
```

This project has been tested with Python 3.12.3, but should work with other recent Python versions as well.

## Acknowledgements
- Google initial DeepDream implementation helped me a lot with guided dreaming: [GitHub](https://github.com/google/deepdream/tree/master)
- The AI Epiphany - is my source of inspiration for this project: [YouTube](https://www.youtube.com/@TheAIEpiphany), [GitHub](https://github.com/gordicaleksa/pytorch-deepdream)

## License
<div align="center">
    <a href="https://github.com/Lassa30/deepdreaming/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg">
    </a>
</div>