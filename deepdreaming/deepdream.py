from . import img
from . import pyramid
from . import shift

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_reference_image_activations(self, reference_tensor):
        self.activations = []
        with torch.no_grad():
            self.model(reference_tensor)
            ref_activations = [act.detach() for act in self.activations]
        self.activations = []
        return ref_activations

    def objective_guide(self, current_acts, guide_acts):
        gradients = []
        for img_act, ref_act in zip(current_acts, guide_acts):
            x = img_act[0].flatten(1)
            y = ref_act[0].flatten(1)

            similarity = x.T @ y
            best_matches = y[:, similarity.argmax(dim=1)]

            gradients.append(best_matches.reshape_as(img_act))

        return gradients

    def gradient_ascend(
        self,
        optimizer_class,
        input_tensor: torch.Tensor,
        ref_activations: Optional[list[torch.Tensor]],
        learning_rate: float,
        num_iterations: int,
    ):
        optimizer = optimizer_class([input_tensor], lr=learning_rate, maximize=True)
        for _ in range(num_iterations):
            optimizer.zero_grad()
            self.activations = []
            self.model(input_tensor)

            if ref_activations is not None:
                gradients = self.objective_guide(self.activations, ref_activations)
                for act, grad in zip(self.activations, gradients):
                    act.backward(grad, retain_graph=True)
            else:
                losses = [torch.norm(activation.flatten(), 2) for activation in self.activations]
                loss = torch.mean(torch.stack(losses))
                loss.backward()

            optimizer.step()

    def dream(
        self,
        input_image: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
        optimizer_class=torch.optim.Adam,
        learning_rate: float = 0.05,
        num_iterations: int = 30,
        image_pyramid_layers: int = 3,
        image_pyramid_ratio: float = 0.5,
        shift_size: int = 10,
    ) -> np.ndarray:

        self.register_hooks()

        random_shift = shift.RandomShift(shift_size)
        input_img = img.proc.pre_process_image(input_image)
        print("IMAGE PYRAMID RUNNING")
        for new_shape in pyramid.Pyramid(input_image.shape, image_pyramid_layers, image_pyramid_ratio):
            input_img = img.proc.reshape_image(input_img, new_shape)
            input_tensor = img.proc.to_tensor(input_img).to(self.device)

            input_tensor = random_shift.shift(input_tensor)

            #####################################################
            print("Input image shape")
            print(input_img.shape)
            #####################################################
            ref_activations = None
            reference_tensor = None
            if reference_image is not None:
                reference_img = img.proc.pre_process_image(reference_image)
                reference_img = img.proc.reshape_image(reference_image, new_shape)

                reference_tensor = img.proc.to_tensor(reference_img).to(self.device)

                reference_tensor = random_shift.shift(reference_tensor)

                ref_activations = self.get_reference_image_activations(reference_tensor)

            self.gradient_ascend(optimizer_class, input_tensor, ref_activations, learning_rate, num_iterations)

            input_tensor = random_shift.shift_back(input_tensor)
            input_img = img.proc.to_image(input_tensor)
            if reference_tensor is not None:
                reference_tensor = random_shift.shift_back(reference_tensor)
            random_shift.generate()
        print("IMAGE PYRAMID STOPPED")

        self.remove_hooks()

        # Post-processing
        output_tensor = input_tensor.detach().clone()
        out = img.proc.to_image(output_tensor)

        assert out is not None, "Output image is None somehow"

        out = img.proc.discard_pre_processing(out)
        out = np.clip(out, 0.0, 1.0)

        return out

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
