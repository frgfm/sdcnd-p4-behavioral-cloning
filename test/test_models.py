import unittest
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from src import models


def get_input(input_shape=(64, 64, 3), dtype=np.float32):

    return (255 * np.random.rand(*input_shape).astype(dtype)).round()


class Tester(unittest.TestCase):

    def test_normalize(self):

        input_t = get_input()
        init_id = id(input_t)
        out = models.normalize_tensor(input_t)

        self.assertTrue(np.all(out <= 0.5))
        self.assertTrue(np.all(out >= -0.5))
        # Check inplace op
        self.assertEqual(id(out), init_id)

    def _test_model(self, arch, input_shape=(64, 64, 3), dtype=np.float32):

        model = models.__dict__[arch](input_shape=input_shape)
        self.assertIsInstance(model, Sequential)

        input_t = get_input(input_shape, dtype=dtype)
        angle = model.predict(input_t[None, ...], batch_size=1)[0]
        self.assertEqual(angle.shape, (1,))
        self.assertEqual(angle.dtype, dtype)


tested_models = ['lenet5', 'nvidia', 'babypilot']

for arch in tested_models:

    def do_test(self, arch=arch):
        self._test_model(arch)

    setattr(Tester, f"test_{arch}", do_test)


if __name__ == '__main__':
    unittest.main()
