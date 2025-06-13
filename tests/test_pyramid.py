import unittest
from deepdreaming.pyramid import ImagePyramid
import numpy as np


class TestPyramid(unittest.TestCase):
    def setUp(self):
        """Initialize test fixtures"""
        self.shape = (224, 224)
        self.ratio = 0.5

    def test_one_layer(self):
        new_shape = next(ImagePyramid(self.shape, 1, self.ratio))
        is_equal = all(new_shape == np.array([224, 224]))
        self.assertTrue(is_equal)

    def test_too_many_layers(self):
        should_raise = lambda: [shape for shape in ImagePyramid(self.shape, 5, self.ratio)]
        self.assertRaises(RuntimeError, should_raise)

    def test_invalid_ratio(self):
        ratio_zero = lambda: next(ImagePyramid(self.shape, 1, 0))
        self.assertRaises(ValueError, ratio_zero)

        ratio_negative = lambda: next(ImagePyramid(self.shape, 1, -1))
        self.assertRaises(ValueError, ratio_negative)

        ratio_greater_than_one = lambda: next(ImagePyramid(self.shape, 1, 5))
        self.assertRaises(ValueError, ratio_greater_than_one)

    def test_float_shape(self):
        float_shape = lambda: next(ImagePyramid((50.1, 50.0), 1, self.ratio))
        self.assertRaises(ValueError, float_shape)


if __name__ == "__main__":
    unittest.main()
