import unittest
import numpy as np
from src.dataset import h_flip, shadow
from test_models import get_input


def get_target(size=1):

    return np.random.rand(size)


class Tester(unittest.TestCase):

    def test_hflip(self):

        # RGB image
        input_t = get_input()
        target = get_target()

        out, t = h_flip(input_t[None, ...], target)

    def test_shadow(self):

        # RGB image
        input_t = get_input()
        target = get_target()
        out, t = shadow(input_t[None, ...], target)


if __name__ == '__main__':
    unittest.main()
