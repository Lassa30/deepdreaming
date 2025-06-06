from . import img
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

    def process_reference_image(self, reference_image):
        if reference_image is None:
            return None

        reference_img = img.proc.pre_process_image(reference_image)
        reference_tensor = img.proc.to_tensor(reference_img).to(self.device)
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

    def dream(
        self,
        input_image: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
        optimizer_class=torch.optim.Adam,
        learning_rate: float = 0.05,
        num_iterations: int = 30,
    ) -> np.ndarray:

        self.register_hooks()
        ref_activations = self.process_reference_image(
            reference_image
        )

        input_img = img.proc.pre_process_image(input_image)
        input_tensor = img.proc.to_tensor(input_img).to(self.device)
        input_tensor.requires_grad_(True)

        optimizer = optimizer_class([input_tensor], lr=learning_rate, maximize=True)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            self.activations = []
            self.model(input_tensor)  # Forward pass

            if ref_activations is not None:
                gradients = self.objective_guide(self.activations, ref_activations)
                for act, grad in zip(self.activations, gradients):
                    act.backward(grad, retain_graph=True)
            else:
                losses = [activation.norm() for activation in self.activations]
                loss = torch.mean(torch.stack(losses))
                loss.backward()

            optimizer.step()

        self.remove_hooks()

        # Post-processing
        output_tensor = torch.sigmoid(input_tensor.detach().clone())
        out = img.proc.to_image(output_tensor)
        out = img.proc.discard_pre_processing(out)

        # Normalize per channel
        for c in range(out.shape[-1]):
            channel = out[..., c]
            min_val = channel.min()
            max_val = channel.max()
            out[..., c] = (channel - min_val) / (max_val - min_val + 1e-8)

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
            self.hook_handles[layer_string] = \
                model_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations.append(output)

    def remove_hooks(self):
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()
        self.activations = []
