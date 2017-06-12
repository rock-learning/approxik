import os
import numpy as np
from approxik import ApproxInvKin
from numpy.testing import assert_array_less


CURRENT_PATH = os.sep.join(__file__.split(os.sep)[:-1])
if CURRENT_PATH:
    DATA_PATH = os.path.join(CURRENT_PATH, "..", "data")


def test_bound_violation():
    urdf_path = os.path.join(DATA_PATH, "kuka_lbr.urdf")
    aik = ApproxInvKin(
        urdf_path, "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
        1.0, 0.001, maxiter=100000, verbose=0)

    q = np.array([-0.383964, 1.13772, 0.639001, 0.632084, -0.836381, 2.07997, -3.02979])
    p = np.array([0.506597, -0.39263, 0.781173, 0.00687452, 0.868186, -0.463353, 0.177511])
    aik.cart_to_jnt(p, q)

    qhi = np.array([2.94, 2.08, 2.94, 2.08, 2.94, 2.08, 3.03])
    qlo = -qhi
    eps = 0.001
    assert_array_less(q, qhi + eps)
    assert_array_less(qlo - eps, q)
