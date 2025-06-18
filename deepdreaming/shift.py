import torch
import numpy as np


class RandomShift:
    def __init__(self, shift_size):
        self.shift_size = shift_size
        self.horizontal, self.vertical = -shift_size, shift_size
        self.update_random_shift()

    def update_random_shift(self):
        new_horizontal, new_vertical = np.random.randint(-self.shift_size, self.shift_size + 1, 2)
        self.horizontal, self.vertical = new_horizontal, new_vertical

    def reverse(self):
        self.horizontal *= -1
        self.vertical *= -1

    def shift(self, shifted: torch.Tensor):
        with torch.no_grad():
            shifted = torch.roll(shifted, shifts=(self.horizontal, self.vertical), dims=(2, 3))
            return shifted

    def shift_back(self, tensor: torch.Tensor):
        self.reverse()
        shifted_back = self.shift(tensor)
        self.reverse()
        return shifted_back
