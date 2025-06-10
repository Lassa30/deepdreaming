import torch
import numpy as np


class RandomShift:
    def __init__(self, shift_size):
        self.shift_size = shift_size
        self.horizontal, self.vertical = self.generate()
    
    def generate(self):
        return np.random.randint(-self.shift_size, self.shift_size + 1, 2)

    def reverse(self):
        self.horizontal *= -1
        self.vertical *= -1

    def shift(self, tensor: torch.Tensor):
        with torch.no_grad():
            tensor = torch.roll(tensor, shifts=(self.horizontal, self.vertical), dims=(2, 3))

    def shift_back(self, tensor: torch.Tensor):
        self.reverse()
        self.shift(tensor)
        self.reverse()
