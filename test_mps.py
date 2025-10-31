import unittest
import numpy as np
from quimb.tensor.tensor_1d import MatrixProductState
from quimb_tebd import mps_to_vector

class TestMPSToVector(unittest.TestCase):

    def test_cb_state(self):
        psi = np.zeros(4)
        psi[0] = 1.
        mps = MatrixProductState.from_dense(psi)
        psi_from_mps = mps_to_vector(mps)
        self.assertTrue(np.allclose(psi_from_mps, psi))

    def test_random_state(self):
        psi = np.random.rand(2 ** 3)
        mps = MatrixProductState.from_dense(psi)
        psi_from_mps = mps_to_vector(mps)
        self.assertTrue(np.allclose(psi_from_mps, psi))


if __name__ == "__main__":
    unittest.main()