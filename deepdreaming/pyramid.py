import numpy as np


class Pyramid:
    MIN_SIZE = 28  # the least possible height or width possible for the image

    def __init__(self, shape, layers, ratio):
        """
        shape: (h, w)
            initial shape of the image
        layers: int
            number of layers of the Image Pyramid
        ratio: float
            ratio = p_i / p_{i+1}, where p = {p_1, p_2, ... , p_l}, shapes of the pyramid layers
            0 < ratio < 1 -- should be satisfied i.e. next layer should be smaller then the current
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
        return self

    def __next__(self):
        while self.exponent >= 0:
            next_shape = np.round(np.float32(self.shape[:2]) * self.ratio**self.exponent).astype(np.int32)
            if next_shape.min() < Pyramid.MIN_SIZE:
                raise RuntimeError(
                    """
                    The smallest size of the pyramid is exceeded. 
                    Consider using less layers or upscaling the image.
                    """
                )
            self.exponent -= 1
            return next_shape
        raise StopIteration
