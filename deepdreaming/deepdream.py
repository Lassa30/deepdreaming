from . import img
from . import pyramid
from . import shift
from .config import DreamConfig

import torch
import numpy as np
from typing import Optional
import re


class DeepDream:
    def __init__(self, model: torch.nn.Module, layers: list[str]):
        self.model = model
        self.layers = layers
        self.hook_handles = {}  # item = {layer, hook_handle}
        self.activations: list[torch.Tensor] = []

    def dream(
        self,
        input_image: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
        config: DreamConfig = DreamConfig(),
    ) -> np.ndarray:
        self._register_hooks()

        random_shift = shift.RandomShift(config.shift_size)
        input_img = img.proc.pre_process_image(input_image)
        reference_img = img.proc.pre_process_image(reference_image)

        for new_shape in pyramid.Pyramid(input_image.shape, config.pyramid_layers, config.pyramid_ratio):
            input_img = img.proc.reshape_image(input_img, new_shape)
            input_tensor = img.proc.to_tensor(input_img).requires_grad_(True)

            reference_img = img.proc.reshape_image(reference_image, new_shape)
            reference_tensor = img.proc.to_tensor(reference_img)

            optimizer = config.optimizer_class([input_tensor], lr=config.learning_rate, maximize=True)
            for _ in range(config.num_iterations):
                DeepDream._shift_tensors(input_tensor, reference_tensor, random_shift.shift)

                self._gradient_ascend_step(optimizer, input_tensor, reference_tensor, config)

                DeepDream._shift_tensors(input_tensor, reference_tensor, random_shift.shift_back)

                random_shift.update_random_shift()

            input_img = img.proc.to_image(input_tensor)

        # Clear and Return
        self._remove_hooks()
        output_img = DeepDream._prepare_output_image(input_tensor)
        return output_img

    def _gradient_ascend_step(
        self,
        optimizer,
        input_tensor: torch.Tensor,
        reference_tensor: Optional[torch.Tensor] = None,
        config: DreamConfig = DreamConfig(),
    ):
        self.activations = []
        optimizer.zero_grad()
        if reference_tensor is not None:
            ref_activations = self._get_reference_image_activations(reference_tensor)
            self.model(input_tensor)
            gradients = DeepDream._objective_guide(self.activations, ref_activations)
            for act, grad in zip(self.activations, gradients):
                act.backward(grad, retain_graph=True)
        else:
            self.model(input_tensor)
            losses = [torch.norm(activation.flatten(), 2) for activation in self.activations]
            loss = torch.mean(torch.stack(losses))
            loss.backward()

        if config.gradient_norm:
            gradient_normalization(input_tensor, config)
        if config.gradient_smooth:
            gradient_smoothing(input_tensor, config)

        optimizer.step()

    @staticmethod
    def _objective_guide(current_acts, guide_acts):
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
        out = img.proc.to_image(input_tensor.detach().clone())
        out = img.proc.discard_pre_processing(out)
        out = np.clip(out, 0.0, 1.0)
        return out

    @staticmethod
    def _shift_tensors(input_tensor, reference_tensor, shifter):
        shifter(input_tensor)
        if reference_tensor is not None:
            shifter(reference_tensor)

    def _register_hooks(self):
        if self.layers is None:
            return
        if type(self.layers) == str:
            self.layers = [self.layers]

        for layer in self.layers:
            layer_string = layer
            normalized_layer = re.sub(r"\[(\d+)\]", r".\1", layer)

            model_layer: torch.nn.Module = self.model
            for layer_attr in normalized_layer.split("."):
                model_layer = getattr(model_layer, layer_attr)
            self.hook_handles[layer_string] = model_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.activations.append(output)

    def _remove_hooks(self):
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()
        self.activations = []

    def _get_reference_image_activations(self, reference_tensor):
        self.activations = []
        with torch.no_grad():
            self.model(reference_tensor)
            ref_activations = [act.detach() for act in self.activations]
        self.activations = []
        return ref_activations


def gradient_normalization(input_tensor, config: DreamConfig = DreamConfig()):
    with torch.no_grad():
        input_tensor.grad.data.copy_(torch.nn.functional.normalize(input_tensor.grad.data, p=config.norm, dim=0))


def gradient_smoothing(input_tensor, config: DreamConfig = DreamConfig()):
    print("Smoothing placeholder")
