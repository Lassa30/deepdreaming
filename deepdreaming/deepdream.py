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

    def get_reference_image_activations(self, reference_tensor):
        self.activations = []
        with torch.no_grad():
            self.model(reference_tensor)
            ref_activations = [act.detach() for act in self.activations]
        self.activations = []
        return ref_activations

    @staticmethod
    def objective_guide(current_acts, guide_acts):
        gradients = []
        for img_act, ref_act in zip(current_acts, guide_acts):
            x = img_act[0].flatten(1)
            y = ref_act[0].flatten(1)

            similarity = x.T @ y
            best_matches = y[:, similarity.argmax(dim=1)]

            gradients.append(best_matches.reshape_as(img_act))

        return gradients

    def gradient_ascend_step(
        self, optimizer, input_tensor: torch.Tensor, reference_tensor: Optional[torch.Tensor] = None
    ):
        self.activations = []
        optimizer.zero_grad()
        if reference_tensor is not None:
            ref_activations = self.get_reference_image_activations(reference_tensor)
            self.model(input_tensor)
            gradients = DeepDream.objective_guide(self.activations, ref_activations)
            for act, grad in zip(self.activations, gradients):
                act.backward(grad, retain_graph=True)
        else:
            self.model(input_tensor)
            losses = [torch.norm(activation.flatten(), 2) for activation in self.activations]
            loss = torch.mean(torch.stack(losses))
            loss.backward()
        optimizer.step()

    def dream(
        self,
        input_image: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
        config: DreamConfig = DreamConfig(),
    ) -> np.ndarray:
        self.register_hooks()

        random_shift = shift.RandomShift(config.shift_size)
        input_img = img.proc.pre_process_image(input_image)
        reference_img = img.proc.pre_process_image(reference_image)

        for new_shape in pyramid.Pyramid(input_image.shape, config.pyramid_layers, config.pyramid_ratio):
            input_img = img.proc.reshape_image(input_img, new_shape)
            input_tensor = img.proc.to_tensor(input_img)
            input_tensor.requires_grad = True

            reference_img = img.proc.reshape_image(reference_image, new_shape)
            reference_tensor = img.proc.to_tensor(reference_img)

            optimizer = config.optimizer_class([input_tensor], lr=config.learning_rate, maximize=True)
            for _ in range(config.num_iterations):
                DeepDream.shift_tensors(input_tensor, reference_tensor, random_shift.shift)

                self.gradient_ascend_step(optimizer, input_tensor, reference_tensor)

                DeepDream.shift_tensors(input_tensor, reference_tensor, random_shift.shift_back)

                random_shift.generate()

            input_img = img.proc.to_image(input_tensor)

        # Clear and Return
        self.remove_hooks()
        out = img.proc.to_image(input_tensor.detach().clone())
        out = img.proc.discard_pre_processing(out)
        return np.clip(out, 0.0, 1.0)
    
    @staticmethod
    def shift_tensors(input_tensor, reference_tensor, shifter):
        shifter(input_tensor)
        if reference_tensor is not None:
            shifter(reference_tensor)

    def register_hooks(self):
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
            self.hook_handles[layer_string] = model_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations.append(output)

    def remove_hooks(self):
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()
        self.activations = []
