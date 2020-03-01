import unittest
import numpy as np
from src.dataset import h_flip, shadow
from test_models import get_input


def get_target(size=1):

    return np.random.rand(size)


class Tester(unittest.TestCase):

    def test_hflip(self):

        # RGB image
        input_t = get_input()[None, ...]
        input_id = id(input_t)
        target = get_target()
        target_id = id(target)

        out, t = h_flip(input_t, target)
        self.assertEqual(id(out), input_id)
        self.assertEqual(id(t), target_id)

    def test_shadow(self):

        # RGB image
        input_t = get_input()
        target = get_target()
        t_val = target.copy()
        target_id = id(target)

        out, t = shadow(input_t[None, ...], target)
        self.assertEqual(id(t), target_id)
        self.assertTrue(np.equal(t, t_val))


if __name__ == '__main__':
    unittest.main()
