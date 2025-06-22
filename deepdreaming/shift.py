import numpy as np
import torch


class RandomShift:
    """Applies a random shift to the input tensor.

    This class is used to randomly shift a tensor. It maintains an internal
    state of the shift and can reverse the shift.

    Args:
        shift_size (int): The maximum shift value in pixels.
    """

    def __init__(self, shift_size):
        """Initializes the RandomShift class.

        Args:
            shift_size (int): The maximum shift value in pixels.
        """
        self.shift_size = shift_size
        self.horizontal, self.vertical = -shift_size, shift_size
        self.update_random_shift()

    def update_random_shift(self):
        """Randomly updates the current shift.

        Note:
            This function modifies the internal state `horizontal` and `vertical`.
        """
        new_horizontal, new_vertical = np.random.randint(-self.shift_size, self.shift_size + 1, 2)
        self.horizontal, self.vertical = new_horizontal, new_vertical

    def reverse(self):
        """Reverses the current shift.

        Note:
            This function modifies the internal state `horizontal` and `vertical`.
        """
        self.horizontal *= -1
        self.vertical *= -1

    def shift(self, shifted: torch.Tensor) -> torch.Tensor:
        """Applies the current shift to the input tensor.

        Args:
            shifted (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The shifted tensor.
        """
        with torch.no_grad():
            shifted = torch.roll(shifted, shifts=(self.horizontal, self.vertical), dims=(2, 3))
            return shifted

    def shift_back(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the reverse of the current shift to the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: The shifted back tensor.
        """
        self.reverse()
        shifted_back = self.shift(tensor)
        self.reverse()
        return shifted_back
