import pytest
import torch

from deepdreaming.shift import RandomShift


@pytest.mark.parametrize(("shift_size"), (32, 64, 128))
def test_reverse(shift_size):
    r_shift = RandomShift(shift_size)

    old_h, old_v = r_shift.horizontal, r_shift.vertical
    r_shift.reverse()
    new_h, new_v = r_shift.horizontal, r_shift.vertical

    assert (old_h == new_h * -1) and (old_v == new_v * -1)


@pytest.mark.parametrize(("shift_size"), (32, 64, 128))
def test_smoke(shift_size):
    r_shift = RandomShift(shift_size)
    rand_tensor = torch.rand((1, 3, 224, 224))
    new_tensor = r_shift.shift(rand_tensor)
    assert new_tensor.shape == rand_tensor.shape
    assert id(new_tensor) != id(rand_tensor)

    new_tensor = r_shift.shift_back(new_tensor)
    assert torch.all(rand_tensor == new_tensor)
