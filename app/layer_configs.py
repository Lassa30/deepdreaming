# Comprehensive layer configurations for each model
MODEL_LAYERS = {
    "vgg16": {
        # Conv layers
        "features[0]": {"type": "Conv2d", "description": "Initial conv 3×3 (RGB→64)"},
        "features[2]": {"type": "Conv2d", "description": "Conv 3×3 (64→64) in block1"},
        "features[5]": {"type": "Conv2d", "description": "Conv 3×3 (64→128) in block2"},
        "features[7]": {"type": "Conv2d", "description": "Conv 3×3 (128→128) in block2"},
        "features[10]": {"type": "Conv2d", "description": "Conv 3×3 (128→256) in block3"},
        "features[12]": {"type": "Conv2d", "description": "Conv 3×3 (256→256) in block3"},
        "features[14]": {"type": "Conv2d", "description": "Conv 3×3 (256→256) in block3"},
        "features[17]": {"type": "Conv2d", "description": "Conv 3×3 (256→512) in block4"},
        "features[19]": {"type": "Conv2d", "description": "Conv 3×3 (512→512) in block4"},
        "features[21]": {"type": "Conv2d", "description": "Conv 3×3 (512→512) in block4"},
        "features[24]": {"type": "Conv2d", "description": "Conv 3×3 (512→512) in block5"},
        "features[26]": {"type": "Conv2d", "description": "Conv 3×3 (512→512) in block5"},
        "features[28]": {"type": "Conv2d", "description": "Conv 3×3 (512→512) in block5"},
        # ReLU layers
        "features[1]": {"type": "ReLU", "description": "ReLU after first conv"},
        "features[3]": {"type": "ReLU", "description": "ReLU after block1 conv2"},
        "features[6]": {"type": "ReLU", "description": "ReLU after block2 conv1"},
        "features[8]": {"type": "ReLU", "description": "ReLU after block2 conv2"},
        "features[11]": {"type": "ReLU", "description": "ReLU after block3 conv1"},
        "features[13]": {"type": "ReLU", "description": "ReLU after block3 conv2"},
        "features[15]": {"type": "ReLU", "description": "ReLU after block3 conv3"},
        "features[18]": {"type": "ReLU", "description": "ReLU after block4 conv1"},
        "features[20]": {"type": "ReLU", "description": "ReLU after block4 conv2"},
        "features[22]": {"type": "ReLU", "description": "ReLU after block4 conv3"},
        "features[25]": {"type": "ReLU", "description": "ReLU after block5 conv1"},
        "features[27]": {"type": "ReLU", "description": "ReLU after block5 conv2"},
        "features[29]": {"type": "ReLU", "description": "ReLU after block5 conv3"},
    },
    "resnet50": {
        # Main ReLU
        "relu": {"type": "ReLU", "description": "Main ReLU after initial conv"},
        # Layer 1 blocks
        "layer1[0].relu": {"type": "ReLU", "description": "Block 1.0 main ReLU"},
        "layer1[0].conv1": {"type": "Conv2d", "description": "Block 1.0 1×1 conv (64→64)"},
        "layer1[0].conv2": {"type": "Conv2d", "description": "Block 1.0 3×3 conv (64→64)"},
        "layer1[0].conv3": {"type": "Conv2d", "description": "Block 1.0 1×1 conv (64→256)"},
        "layer1[1].relu": {"type": "ReLU", "description": "Block 1.1 main ReLU"},
        "layer1[1].conv1": {"type": "Conv2d", "description": "Block 1.1 1×1 conv (256→64)"},
        "layer1[1].conv2": {"type": "Conv2d", "description": "Block 1.1 3×3 conv (64→64)"},
        "layer1[1].conv3": {"type": "Conv2d", "description": "Block 1.1 1×1 conv (64→256)"},
        "layer1[2].relu": {"type": "ReLU", "description": "Block 1.2 main ReLU"},
        "layer1[2].conv1": {"type": "Conv2d", "description": "Block 1.2 1×1 conv (256→64)"},
        "layer1[2].conv2": {"type": "Conv2d", "description": "Block 1.2 3×3 conv (64→64)"},
        "layer1[2].conv3": {"type": "Conv2d", "description": "Block 1.2 1×1 conv (64→256)"},
        # Layer 2 blocks
        "layer2[0].relu": {"type": "ReLU", "description": "Block 2.0 main ReLU"},
        "layer2[0].conv1": {"type": "Conv2d", "description": "Block 2.0 1×1 conv (256→128)"},
        "layer2[0].conv2": {"type": "Conv2d", "description": "Block 2.0 3×3 conv (128→128)"},
        "layer2[0].conv3": {"type": "Conv2d", "description": "Block 2.0 1×1 conv (128→512)"},
        "layer2[1].relu": {"type": "ReLU", "description": "Block 2.1 main ReLU"},
        "layer2[1].conv1": {"type": "Conv2d", "description": "Block 2.1 1×1 conv (512→128)"},
        "layer2[1].conv2": {"type": "Conv2d", "description": "Block 2.1 3×3 conv (128→128)"},
        "layer2[1].conv3": {"type": "Conv2d", "description": "Block 2.1 1×1 conv (128→512)"},
        "layer2[2].relu": {"type": "ReLU", "description": "Block 2.2 main ReLU"},
        "layer2[2].conv1": {"type": "Conv2d", "description": "Block 2.2 1×1 conv (512→128)"},
        "layer2[2].conv2": {"type": "Conv2d", "description": "Block 2.2 3×3 conv (128→128)"},
        "layer2[2].conv3": {"type": "Conv2d", "description": "Block 2.2 1×1 conv (128→512)"},
        "layer2[3].relu": {"type": "ReLU", "description": "Block 2.3 main ReLU"},
        "layer2[3].conv1": {"type": "Conv2d", "description": "Block 2.3 1×1 conv (512→128)"},
        "layer2[3].conv2": {"type": "Conv2d", "description": "Block 2.3 3×3 conv (128→128)"},
        "layer2[3].conv3": {"type": "Conv2d", "description": "Block 2.3 1×1 conv (128→512)"},
        # Layer 3 blocks
        "layer3[0].relu": {"type": "ReLU", "description": "Block 3.0 main ReLU"},
        "layer3[0].conv1": {"type": "Conv2d", "description": "Block 3.0 1×1 conv (512→256)"},
        "layer3[0].conv2": {"type": "Conv2d", "description": "Block 3.0 3×3 conv (256→256)"},
        "layer3[0].conv3": {"type": "Conv2d", "description": "Block 3.0 1×1 conv (256→1024)"},
        "layer3[1].relu": {"type": "ReLU", "description": "Block 3.1 main ReLU"},
        "layer3[1].conv1": {"type": "Conv2d", "description": "Block 3.1 1×1 conv (1024→256)"},
        "layer3[1].conv2": {"type": "Conv2d", "description": "Block 3.1 3×3 conv (256→256)"},
        "layer3[1].conv3": {"type": "Conv2d", "description": "Block 3.1 1×1 conv (256→1024)"},
        "layer3[2].relu": {"type": "ReLU", "description": "Block 3.2 main ReLU"},
        "layer3[2].conv1": {"type": "Conv2d", "description": "Block 3.2 1×1 conv (1024→256)"},
        "layer3[2].conv2": {"type": "Conv2d", "description": "Block 3.2 3×3 conv (256→256)"},
        "layer3[2].conv3": {"type": "Conv2d", "description": "Block 3.2 1×1 conv (256→1024)"},
        "layer3[3].relu": {"type": "ReLU", "description": "Block 3.3 main ReLU"},
        "layer3[3].conv1": {"type": "Conv2d", "description": "Block 3.3 1×1 conv (1024→256)"},
        "layer3[3].conv2": {"type": "Conv2d", "description": "Block 3.3 3×3 conv (256→256)"},
        "layer3[3].conv3": {"type": "Conv2d", "description": "Block 3.3 1×1 conv (256→1024)"},
        "layer3[4].relu": {"type": "ReLU", "description": "Block 3.4 main ReLU"},
        "layer3[4].conv1": {"type": "Conv2d", "description": "Block 3.4 1×1 conv (1024→256)"},
        "layer3[4].conv2": {"type": "Conv2d", "description": "Block 3.4 3×3 conv (256→256)"},
        "layer3[4].conv3": {"type": "Conv2d", "description": "Block 3.4 1×1 conv (256→1024)"},
        "layer3[5].relu": {"type": "ReLU", "description": "Block 3.5 main ReLU"},
        "layer3[5].conv1": {"type": "Conv2d", "description": "Block 3.5 1×1 conv (1024→256)"},
        "layer3[5].conv2": {"type": "Conv2d", "description": "Block 3.5 3×3 conv (256→256)"},
        "layer3[5].conv3": {"type": "Conv2d", "description": "Block 3.5 1×1 conv (256→1024)"},
        # Layer 4 blocks
        "layer4[0].relu": {"type": "ReLU", "description": "Block 4.0 main ReLU"},
        "layer4[0].conv1": {"type": "Conv2d", "description": "Block 4.0 1×1 conv (1024→512)"},
        "layer4[0].conv2": {"type": "Conv2d", "description": "Block 4.0 3×3 conv (512→512)"},
        "layer4[0].conv3": {"type": "Conv2d", "description": "Block 4.0 1×1 conv (512→2048)"},
        "layer4[1].relu": {"type": "ReLU", "description": "Block 4.1 main ReLU"},
        "layer4[1].conv1": {"type": "Conv2d", "description": "Block 4.1 1×1 conv (2048→512)"},
        "layer4[1].conv2": {"type": "Conv2d", "description": "Block 4.1 3×3 conv (512→512)"},
        "layer4[1].conv3": {"type": "Conv2d", "description": "Block 4.1 1×1 conv (512→2048)"},
        "layer4[2].relu": {"type": "ReLU", "description": "Block 4.2 main ReLU"},
        "layer4[2].conv1": {"type": "Conv2d", "description": "Block 4.2 1×1 conv (2048→512)"},
        "layer4[2].conv2": {"type": "Conv2d", "description": "Block 4.2 3×3 conv (512→512)"},
        "layer4[2].conv3": {"type": "Conv2d", "description": "Block 4.2 1×1 conv (512→2048)"},
    },
    # Add Inception v3 layer definitions
    # Inception V3 model layers - focusing only on convolutional layers
    "inception": {
        # Early layers
        "Conv2d_1a_3x3.conv": {"type": "Conv2d", "description": "First 3×3 conv (RGB→32) stride 2"},
        "Conv2d_2a_3x3.conv": {"type": "Conv2d", "description": "Second 3×3 conv (32→32)"},
        "Conv2d_2b_3x3.conv": {"type": "Conv2d", "description": "Third 3×3 conv (32→64) padding 1"},
        "Conv2d_3b_1x1.conv": {"type": "Conv2d", "description": "1×1 conv (64→80)"},
        "Conv2d_4a_3x3.conv": {"type": "Conv2d", "description": "3×3 conv (80→192)"},
        # Mixed_5b block
        "Mixed_5b.branch1x1.conv": {"type": "Conv2d", "description": "Mixed_5b 1×1 conv (192→64)"},
        "Mixed_5b.branch5x5_1.conv": {"type": "Conv2d", "description": "Mixed_5b 1×1 conv (192→48)"},
        "Mixed_5b.branch5x5_2.conv": {"type": "Conv2d", "description": "Mixed_5b 5×5 conv (48→64) padding 2"},
        "Mixed_5b.branch3x3dbl_1.conv": {"type": "Conv2d", "description": "Mixed_5b 1×1 conv (192→64)"},
        "Mixed_5b.branch3x3dbl_2.conv": {"type": "Conv2d", "description": "Mixed_5b 3×3 conv (64→96) padding 1"},
        "Mixed_5b.branch3x3dbl_3.conv": {"type": "Conv2d", "description": "Mixed_5b 3×3 conv (96→96) padding 1"},
        "Mixed_5b.branch_pool.conv": {"type": "Conv2d", "description": "Mixed_5b 1×1 conv (192→32)"},
        # Mixed_5c block
        "Mixed_5c.branch1x1.conv": {"type": "Conv2d", "description": "Mixed_5c 1×1 conv (256→64)"},
        "Mixed_5c.branch5x5_1.conv": {"type": "Conv2d", "description": "Mixed_5c 1×1 conv (256→48)"},
        "Mixed_5c.branch5x5_2.conv": {"type": "Conv2d", "description": "Mixed_5c 5×5 conv (48→64) padding 2"},
        "Mixed_5c.branch3x3dbl_1.conv": {"type": "Conv2d", "description": "Mixed_5c 1×1 conv (256→64)"},
        "Mixed_5c.branch3x3dbl_2.conv": {"type": "Conv2d", "description": "Mixed_5c 3×3 conv (64→96) padding 1"},
        "Mixed_5c.branch3x3dbl_3.conv": {"type": "Conv2d", "description": "Mixed_5c 3×3 conv (96→96) padding 1"},
        "Mixed_5c.branch_pool.conv": {"type": "Conv2d", "description": "Mixed_5c 1×1 conv (256→64)"},
        # Mixed_5d block
        "Mixed_5d.branch1x1.conv": {"type": "Conv2d", "description": "Mixed_5d 1×1 conv (288→64)"},
        "Mixed_5d.branch5x5_1.conv": {"type": "Conv2d", "description": "Mixed_5d 1×1 conv (288→48)"},
        "Mixed_5d.branch5x5_2.conv": {"type": "Conv2d", "description": "Mixed_5d 5×5 conv (48→64) padding 2"},
        "Mixed_5d.branch3x3dbl_1.conv": {"type": "Conv2d", "description": "Mixed_5d 1×1 conv (288→64)"},
        "Mixed_5d.branch3x3dbl_2.conv": {"type": "Conv2d", "description": "Mixed_5d 3×3 conv (64→96) padding 1"},
        "Mixed_5d.branch3x3dbl_3.conv": {"type": "Conv2d", "description": "Mixed_5d 3×3 conv (96→96) padding 1"},
        "Mixed_5d.branch_pool.conv": {"type": "Conv2d", "description": "Mixed_5d 1×1 conv (288→64)"},
        # Mixed_6a block
        "Mixed_6a.branch3x3.conv": {"type": "Conv2d", "description": "Mixed_6a 3×3 conv (288→384) stride 2"},
        "Mixed_6a.branch3x3dbl_1.conv": {"type": "Conv2d", "description": "Mixed_6a 1×1 conv (288→64)"},
        "Mixed_6a.branch3x3dbl_2.conv": {"type": "Conv2d", "description": "Mixed_6a 3×3 conv (64→96) padding 1"},
        "Mixed_6a.branch3x3dbl_3.conv": {"type": "Conv2d", "description": "Mixed_6a 3×3 conv (96→96) stride 2"},
        # Mixed_6b block (selections)
        "Mixed_6b.branch1x1.conv": {"type": "Conv2d", "description": "Mixed_6b 1×1 conv (768→192)"},
        "Mixed_6b.branch7x7_1.conv": {"type": "Conv2d", "description": "Mixed_6b 1×1 conv (768→128)"},
        "Mixed_6b.branch7x7_2.conv": {"type": "Conv2d", "description": "Mixed_6b 1×7 conv (128→128) padding (0,3)"},
        "Mixed_6b.branch7x7_3.conv": {"type": "Conv2d", "description": "Mixed_6b 7×1 conv (128→192) padding (3,0)"},
        # Mixed_7a block (selections)
        "Mixed_7a.branch3x3_1.conv": {"type": "Conv2d", "description": "Mixed_7a 1×1 conv (768→192)"},
        "Mixed_7a.branch3x3_2.conv": {"type": "Conv2d", "description": "Mixed_7a 3×3 conv (192→320) stride 2"},
        # Mixed_7b block (selections)
        "Mixed_7b.branch1x1.conv": {"type": "Conv2d", "description": "Mixed_7b 1×1 conv (1280→320)"},
        "Mixed_7b.branch3x3_1.conv": {"type": "Conv2d", "description": "Mixed_7b 1×1 conv (1280→384)"},
        # Mixed_7c block (selections)
        "Mixed_7c.branch1x1.conv": {"type": "Conv2d", "description": "Mixed_7c 1×1 conv (2048→320)"},
        "Mixed_7c.branch3x3_1.conv": {"type": "Conv2d", "description": "Mixed_7c 1×1 conv (2048→384)"},
    },
}

# Default selected layers for each model (for convenience)
DEFAULT_SELECTED_LAYERS = {
    "vgg16": ["features[13]", "features[15]", "features[22]", "features[29]"],
    "resnet50": ["layer3[2].relu", "layer3[3].relu", "layer3[4].relu", "layer3[5].relu"],
    "inception": [
        "Mixed_5c.branch3x3dbl_1.conv",
        "Mixed_5c.branch3x3dbl_2.conv",
        "Mixed_5c.branch3x3dbl_3.conv",
        "Mixed_5c.branch_pool.conv",
        "Mixed_5d.branch1x1.conv",
        "Mixed_5d.branch5x5_1.conv",
        "Mixed_5d.branch5x5_2.conv",
        "Mixed_5d.branch3x3dbl_1.conv",
        "Mixed_5d.branch3x3dbl_2.conv",
        "Mixed_5d.branch3x3dbl_3.conv",
        "Mixed_5d.branch_pool.conv",
        "Mixed_6a.branch3x3.conv",
        "Mixed_6a.branch3x3dbl_1.conv",
        "Mixed_6a.branch3x3dbl_2.conv",
        "Mixed_6a.branch3x3dbl_3.conv",
        "Mixed_6b.branch1x1.conv",
        "Mixed_6b.branch7x7_1.conv",
        "Mixed_6b.branch7x7_2.conv",
        "Mixed_6b.branch7x7_3.conv",
        "Mixed_7a.branch3x3_1.conv",
        "Mixed_7a.branch3x3_2.conv",
        "Mixed_7b.branch1x1.conv",
        "Mixed_7b.branch3x3_1.conv",
        "Mixed_7c.branch1x1.conv",
        "Mixed_7c.branch3x3_1.conv",
    ],
}


# Helper function to get all layers of specific type for a model
def get_layers_by_type(model_name, layer_type=None):
    """Returns layers of a specific type for the given model.

    Args:
        model_name (str): Name of the model (e.g., "vgg16", "resnet50")
        layer_type (str, optional): Type of layer to return (e.g., "Conv2d", "ReLU").
                                    If None, returns all layers.

    Returns:
        dict: Dictionary of layer identifiers and their descriptions matching the type
    """
    if model_name.lower() not in MODEL_LAYERS:
        return {}

    if layer_type is None:
        return MODEL_LAYERS[model_name.lower()]

    return {layer_id: info for layer_id, info in MODEL_LAYERS[model_name.lower()].items() if info["type"] == layer_type}
