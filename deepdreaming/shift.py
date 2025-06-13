import torch
import numpy as np


class RandomShift:
    """Applies random shifts to tensors for DeepDream augmentation.
    
    Used to reduce edge artifacts and create more natural patterns by
    randomly shifting the input tensor during optimization.
    """

    def __init__(self, shift_size):
        """Initialize with maximum shift distance in pixels.
        
        Args:
            shift_size (int): Maximum number of pixels to shift in any direction.
                             Actual shifts will be random values in [-shift_size, shift_size].
        """
        self.shift_size = shift_size
        self.horizontal, self.vertical = -shift_size, shift_size
        self.update_random_shift()

    def update_random_shift(self):
        """Generate new random horizontal and vertical shift values.
        
        Creates new shift values uniformly distributed in [-shift_size, shift_size]
        for both horizontal and vertical directions.
        """
        new_horizontal, new_vertical = np.random.randint(-self.shift_size, self.shift_size + 1, 2)
        self.horizontal, self.vertical = new_horizontal, new_vertical

    def reverse(self):
        """Reverse current shift direction by negating shift values.
        
        Used internally to implement shift_back functionality.
        """
        self.horizontal *= -1
        self.vertical *= -1

    def shift(self, tensor: torch.Tensor):
        """Apply current shift to tensor in-place using torch.roll.
        
        Args:
            tensor (torch.Tensor): Input tensor of shape (batch, channels, height, width).
                                  Modified in-place by rolling along spatial dimensions.
        
        Note:
            Uses torch.roll which wraps pixels around edges (circular shift).
        """
        with torch.no_grad():
            tensor = torch.roll(tensor, shifts=(self.horizontal, self.vertical), dims=(2, 3))

    def shift_back(self, tensor: torch.Tensor):
        """Apply reverse shift to tensor, restoring original position.
        
        Args:
            tensor (torch.Tensor): Tensor to shift back to original position.
                                  Must be the same tensor that was previously shifted.
        
        Note:
            This undoes the last shift operation by applying the negative shift.
        """
        self.reverse()
        self.shift(tensor)
        self.reverse()
