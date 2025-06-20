from deepdreaming.pyramid import ImagePyramid
import pytest


class TestImagePyramid:
    def test_zero_layers(self):
        with pytest.raises(ValueError):
            ImagePyramid((224, 224), 0, 0.5)

    def test_invalid_ratio(self):
        # equal to 1
        with pytest.raises(ValueError):
            ImagePyramid((224, 224), 2, 1)
        # equal to 0
        with pytest.raises(ValueError):
            ImagePyramid((224, 224), 2, 0)
        # less than 0
        with pytest.raises(ValueError):
            ImagePyramid((224, 224), 0, -1)
        # greater then 1
        with pytest.raises(ValueError):
            ImagePyramid((224, 224), 0, 5)

    def test_more_layers_than_allowed(self):
        shapes = [(224, 224), (512, 512)]
        raising_layers = [4, 6]
        ratio = 0.5
        for shape, layers in zip(shapes, raising_layers):
            with pytest.raises(RuntimeError):
                next(ImagePyramid(shape, layers, ratio))

    @pytest.mark.parametrize(
        ("ratio", "layers", "expected"),
        (
            (0.5, 1, (512, 512)),
            (0.5, 2, (256, 256)),
            (0.5, 4, (64, 64)),
            (0.5, 5, (32, 32)),
        ),
    )
    def test_upper_layer(self, ratio, layers, expected):
        upper_layer_shape = next(ImagePyramid((512, 512), layers, ratio))
        assert tuple(upper_layer_shape) == expected

    def test_smoke(self):
        shape, ratio, layers = (224, 224), 0.5, 3
        pyramid = ImagePyramid(shape=shape, layers=layers, ratio=ratio)
        shapes = []
        for shape in pyramid:
            shapes.append(shape)
        assert len(shapes) == layers
        assert all(d1 == d2 for d1, d2 in zip(shape, sorted(shape, reverse=True)))
        for idx in range(len(shapes) - 1):
            curr, prev = shapes[idx], shapes[idx + 1]
            assert prev[0] == 2 * curr[0]
            assert prev[1] == 2 * curr[1]
