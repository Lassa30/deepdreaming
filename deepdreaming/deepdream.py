import re

import numpy as np
import torch

from deepdreaming import img, pyramid, shift, smoothing
from deepdreaming.config import DreamConfig, GradSmoothingMode


class DeepDream:
    """DeepDream algorithm implementation for generating dream-like visualizations.

    This class implements the DeepDream algorithm that maximizes activations in specified
    neural network layers, creating surreal, dream-like patterns in images. Supports both
    standard DeepDream (amplifying existing patterns) and guided DeepDream (using reference images).
    """

    def __init__(self, model: torch.nn.Module, layers: list[str]):
        """Initialize DeepDream with a neural network model and target layers.

        Args:
            model (torch.nn.Module): Pre-trained PyTorch model (e.g., VGG, ResNet) to use
                for feature extraction. Model should be in eval mode and better with gradients disabled.
            layers (list[str]): List of layer names to target for activation maximization.
                Layer names should follow PyTorch module naming convention.
                Examples: ["features[25]", "features[29]"] for VGG,
                          ["layer2[3].relu", "layer3[5].conv1"] for ResNet.

        Note:
            The model will be used to extract features at specified layers during the
            dreaming process. Hooks are registered on these layers to capture activations.
        """
        self.model = model
        self.layers = layers

        self.hook_handles: dict[str, torch.utils.hooks.RemovableHandle] = {}  # item = {layer, hook_handle}
        self.activations: list[torch.Tensor] = []

    def dream(
        self,
        input_image: np.ndarray,
        config: DreamConfig = DreamConfig(),
    ) -> np.ndarray:
        """Generate DeepDream visualization from input image.

        This method applies the standard DeepDream algorithm to transform an input
        image into a dream-like visualization. The process works by:
        1. Creating an image pyramid for multi-scale processing
        2. For each pyramid level, iteratively optimizing the image to maximize activations
        3. Applying random shifts and gradient processing for better results

        Args:
            input_image (np.ndarray): Input image to transform, shape (H, W, C) with values in [0, 1].
                                    Must be np.float32 type. Use img.io.read_image() for proper format.
            config (DreamConfig, optional): Configuration object containing all hyperparameters.
                                          Uses default settings if not provided.

        Returns:
            np.ndarray: Transformed dream image with same dimensions as input, values in [0, 1].
                       The output shows amplified patterns based on the target layers.

        Examples:
            >>> from torchvision.models import vgg16, VGG16_Weights
            >>> from deepdreaming import img, deepdream as dd
            >>> from deepdreaming.config import DreamConfig

            # Basic DeepDream
            >>> model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
            >>> deepdream = dd.DeepDream(model, ["features.25", "features.27"])
            >>> image = img.io.read_image("path/to/image.jpg", (224, 224, 3))
            >>> result = deepdream.dream(image)

            # With custom configuration
            >>> config = DreamConfig(
            ...     pyramid_layers=4,
            ...     num_iterations=15,
            ...     learning_rate=0.08,
            ...     gradient_norm=True
            ... )
            >>> result = deepdream.dream(image, config=config)

        Note:
            - Always use img.io.read_image() to load images in the correct format (np.float32, [0,1])
            - The algorithm processes images at multiple scales (pyramid) starting from small
              and working up to full resolution. This creates more coherent and detailed results.
            - For VGG models, use layer names like "features.25", "features.27"
            - For ResNet models, use layer names like "layer2[3].relu", "layer3[5].conv1"
        """
        self._register_hooks()

        input_img = img.proc.pre_process_image(input_image)
        random_shift = shift.RandomShift(config.shift_size)
        image_pyramid = pyramid.ImagePyramid(input_image.shape, config.pyramid_layers, config.pyramid_ratio)

        for new_shape in image_pyramid:
            input_img = img.proc.reshape_image(input_img, new_shape)
            input_tensor = img.proc.to_tensor(input_img).requires_grad_(True)

            optimizer = config.optimizer_class([input_tensor], lr=config.learning_rate, maximize=True)
            for _ in range(config.num_iter):
                input_tensor.data.copy_(random_shift.shift(input_tensor))

                # Standard DeepDream optimization step
                self._optimize_standard(input_tensor, optimizer, config)

                input_tensor.data.copy_(random_shift.shift_back(input_tensor))
                random_shift.update_random_shift()

            input_img = img.proc.to_image(input_tensor)

        # Clear and Return
        self._remove_hooks()
        output_img = DeepDream._prepare_output_image(input_tensor)
        return output_img

    def dream_guided(
        self,
        input_image: np.ndarray,
        reference_image: np.ndarray,
        config: DreamConfig = DreamConfig(),
    ) -> np.ndarray:
        """Generate guided DeepDream visualization from input image and reference image.

        This method applies the guided DeepDream algorithm to transform an input image using
        patterns from a reference image. The process works by:
        1. Creating an image pyramid for multi-scale processing
        2. For each pyramid level, iteratively optimizing the input image based on reference patterns
        3. Applying random shifts and gradient processing for better results

        Args:
            input_image (np.ndarray): Input image to transform, shape (H, W, C) with values in [0, 1].
                                    Must be np.float32 type. Use img.io.read_image() for proper format.
            reference_image (np.ndarray): Reference image for guided dreaming.
                                         The algorithm will try to transfer patterns
                                         from this image to the input image.
                                         Must be same format as input_image (np.float32, [0,1]).
            config (DreamConfig, optional): Configuration object containing all hyperparameters.
                                          Uses default settings if not provided.

        Returns:
            np.ndarray: Transformed dream image with same dimensions as input, values in [0, 1].
                       The output shows transferred patterns from the reference image.

        Examples:
            >>> from torchvision.models import vgg16, VGG16_Weights
            >>> from deepdreaming import img, deepdream as dd
            >>> from deepdreaming.config import DreamConfig

            # Guided DeepDream with reference image
            >>> model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).eval()
            >>> deepdream = dd.DeepDream(model, ["features.25", "features.27"])
            >>> image = img.io.read_image("path/to/image.jpg", (224, 224, 3))
            >>> reference = img.io.read_image("path/to/reference.jpg", (224, 224, 3))
            >>> result = deepdream.dream_guided(image, reference)

            # With custom configuration
            >>> config = DreamConfig(
            ...     pyramid_layers=4,
            ...     num_iterations=15,
            ...     learning_rate=0.08,
            ...     gradient_norm=True
            ... )
            >>> result = deepdream.dream_guided(image, reference, config=config)
        """
        self._register_hooks()

        input_img = img.proc.pre_process_image(input_image)
        reference_img = img.proc.pre_process_image(reference_image)

        random_shift = shift.RandomShift(config.shift_size)
        image_pyramid = pyramid.ImagePyramid(input_image.shape, config.pyramid_layers, config.pyramid_ratio)

        for new_shape in image_pyramid:
            input_img = img.proc.reshape_image(input_img, new_shape)
            input_tensor = img.proc.to_tensor(input_img).requires_grad_(True)

            reference_img = img.proc.reshape_image(reference_img, new_shape)
            reference_tensor = img.proc.to_tensor(reference_img)

            optimizer = config.optimizer_class([input_tensor], lr=config.learning_rate, maximize=True)
            for _ in range(config.num_iter):
                # Apply shift to both tensors
                DeepDream._shift_tensors(input_tensor, reference_tensor, random_shift.shift)

                # Guided optimization step
                self._optimize_guided(input_tensor, reference_tensor, optimizer, config)

                # Shift back and update random shift
                DeepDream._shift_tensors(input_tensor, reference_tensor, random_shift.shift_back)
                random_shift.update_random_shift()

            input_img = img.proc.to_image(input_tensor)

        # Clear and Return
        self._remove_hooks()
        output_img = DeepDream._prepare_output_image(input_tensor)
        return output_img

    def _optimize_standard(self, input_tensor: torch.Tensor, optimizer, config: DreamConfig) -> None:
        """Perform one standard gradient ascent step to maximize layer activations."""
        self.activations = []
        optimizer.zero_grad()

        # Forward pass and compute loss
        self.model(input_tensor)
        losses = [torch.norm(activation.flatten(), 2) for activation in self.activations]
        loss = torch.mean(torch.stack(losses))
        loss.backward()

        # Process gradients
        self._process_gradients(input_tensor, config)

        # Update tensor
        optimizer.step()

    def _optimize_guided(
        self, input_tensor: torch.Tensor, reference_tensor: torch.Tensor, optimizer, config: DreamConfig
    ) -> None:
        """Perform one guided gradient ascent step using reference image."""
        self.activations = []
        optimizer.zero_grad()

        # Get reference activations
        ref_activations = self._get_reference_image_activations(reference_tensor)

        # Forward pass with input image
        self.model(input_tensor)

        # Compute guided gradients
        gradients = DeepDream._objective_guide(self.activations, ref_activations)
        for act, grad in zip(self.activations, gradients):
            act.backward(grad, retain_graph=True)

        # Process gradients
        self._process_gradients(input_tensor, config)

        # Update tensor
        optimizer.step()

    def _process_gradients(self, input_tensor: torch.Tensor, config: DreamConfig) -> None:
        """Apply smoothing and normalization to gradients."""
        # Check `config.py` and `smoothing.py` to see how it works
        self._gradient_smoothing(input_tensor, config)
        self._gradient_normalization(input_tensor, config)

    @staticmethod
    def _objective_guide(current_acts, guide_acts) -> list[torch.Tensor]:
        """Compute gradients to guide current activations toward reference activations."""
        gradients = []
        for img_act, ref_act in zip(current_acts, guide_acts):
            x = img_act[0].flatten(1)
            y = ref_act[0].flatten(1)

            similarity = x.T @ y
            best_matches = y[:, similarity.argmax(dim=1)]

            gradients.append(best_matches.reshape_as(img_act))

        return gradients

    @staticmethod
    def _prepare_output_image(input_tensor: torch.Tensor) -> np.ndarray:
        """Convert optimized tensor to final output image with proper postprocessing."""
        out = img.proc.to_image(input_tensor.detach().clone())
        out = img.proc.discard_pre_processing(out)
        out = np.clip(out, 0.0, 1.0)
        return out

    @staticmethod
    def _shift_tensors(input_tensor, reference_tensor, shifter) -> None:
        """Apply shift function to input and optional reference tensors."""
        input_tensor.data.copy_(shifter(input_tensor))
        if reference_tensor is not None:
            reference_tensor.data.copy_(shifter(reference_tensor))

    def _register_hooks(self) -> None:
        """Register forward hooks on specified model layers to capture activations."""
        if self.layers is None:
            return
        if type(self.layers) is str:
            self.layers = [self.layers]

        for layer in self.layers:
            layer_string = layer
            normalized_layer = re.sub(r"\[(\d+)\]", r".\1", layer)

            model_layer: torch.nn.Module = self.model
            for layer_attr in normalized_layer.split("."):
                model_layer = getattr(model_layer, layer_attr)
            self.hook_handles[layer_string] = model_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """Hook function called during forward pass to store layer activations."""
        self.activations.append(output)

    def _remove_hooks(self):
        """Remove all registered hooks and clear stored activations."""
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()
        self.activations = []

    def _get_reference_image_activations(self, reference_tensor):
        """Extract activations from reference image without gradient tracking."""
        self.activations = []
        with torch.no_grad():
            self.model(reference_tensor)
            ref_activations = [act.detach() for act in self.activations]
        self.activations = []
        return ref_activations

    @staticmethod
    def _gradient_normalization(input_tensor: torch.Tensor, config: DreamConfig = DreamConfig()):
        """Normalize gradients to unit norm for more stable optimization."""
        assert input_tensor.grad is not None, "No gradients are provided for this tensor."
        with torch.no_grad():
            input_tensor.grad.data.copy_(torch.nn.functional.normalize(input_tensor.grad.data, p=2, dim=0))

    @staticmethod
    def _gradient_smoothing(input_tensor: torch.Tensor, config: DreamConfig = DreamConfig()):
        """Apply smoothing to gradients to reduce artifacts (placeholder implementation)."""
        assert input_tensor.grad is not None, "No gradients are provided for this tensor."
        with torch.no_grad():
            match config.grad_smoothing:
                case GradSmoothingMode.BoxSmoothing:
                    input_tensor.grad.data.copy_(
                        smoothing.box_smoothing(input_tensor.grad.data, *smoothing.get_box_smoothing_params(config))
                    )
                case GradSmoothingMode.GaussianSmoothing:
                    input_tensor.grad.data.copy_(
                        smoothing.gaussian_smoothing(
                            input_tensor.grad.data, *smoothing.get_gaussian_smoothing_params(config)
                        )
                    )
                case GradSmoothingMode.Disable:
                    pass
                case _:
                    assert False, f"Invalid `grad_smoothing` parameter is provided: {config.grad_smoothing}"
