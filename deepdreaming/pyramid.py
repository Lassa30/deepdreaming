import numpy as np


class ImagePyramid:
    """Image pyramid iterator that generates progressively smaller shapes for multi-scale processing.

    Used in DeepDream to process images at multiple resolutions, starting from smallest
    and working up to full size. This creates more coherent and detailed dream patterns.

    Note: This iterator returns shapes (height, width) tuples, not actual resized images.
    """

    MIN_SIZE = 32  # the least possible height or width possible for the image

    def __init__(self, shape, layers, ratio):
        """Initialize pyramid with image shape and scaling parameters.

        Args:
            shape (tuple): Initial image shape as (height, width) or (height, width, channels).
                          Only first two dimensions (h, w) are used for pyramid calculation.
            layers (int): Number of pyramid layers to generate. Must be >= 1.
            ratio (float): Scale ratio between consecutive layers, where 0 < ratio < 1.
                          Each layer is ratio times smaller than the previous.
                          E.g., ratio=0.75 means each layer is 75% the size of the previous.

        Raises:
            ValueError: If shape contains non-integer values, layers < 1, or ratio not in (0,1).
        """
        self.shape = shape
        self.ratio = ratio
        self.exponent = layers - 1

        if any(x != int(x) for x in self.shape):
            raise ValueError("Shapes should be integer values")
        if self.exponent < 0:
            raise ValueError("Number of layers of the pyramid should be >= 1.")
        if self.ratio >= 1 or self.ratio <= 0:
            raise ValueError("Ratio of the image pyramid should be 0 < ratio < 1")

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self):
        """Generate next pyramid layer shape.

        Returns:
            np.ndarray: Shape array of format [height, width] for the next pyramid layer.
                       Shapes are generated from smallest to largest (ascending order).

        Raises:
            RuntimeError: If calculated shape would be smaller than MIN_SIZE pixels.
            StopIteration: When all pyramid layers have been generated.

        Note:
            Returns shape tuples only - actual image resizing must be done separately.
        """
        while self.exponent >= 0:
            next_shape = np.round(np.float32(self.shape[:2]) * np.power(self.ratio, self.exponent)).astype(np.int32)
            if next_shape.min() < ImagePyramid.MIN_SIZE:
                raise RuntimeError(
                    """
                    The smallest size of the pyramid is exceeded. 
                    Consider using less layers or upscaling the image.
                    """
                )
            self.exponent -= 1
            return next_shape
        raise StopIteration
