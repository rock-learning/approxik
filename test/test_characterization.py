"""Characterization tests to ensure stability when dependencies change."""
import os
import numpy as np
from approxik import ApproxInvKin, ApproxLocalInvKin, ExactInvKin
from numpy.testing import assert_array_almost_equal
from nose.tools import (assert_false, assert_less, assert_less_equal,
                        assert_raises)


CURRENT_PATH = os.sep.join(__file__.split(os.sep)[:-1])
if CURRENT_PATH:
    DATA_PATH = os.path.join(CURRENT_PATH, "..", "data")


def test_no_jump():
    urdf_path = os.path.join(DATA_PATH, "kuka_lbr.urdf")
    base_link = "kuka_lbr_l_link_0"
    ee_link = "kuka_lbr_l_link_7"

    aik = ApproxLocalInvKin(urdf_path, base_link, ee_link, max_jump=0.1,
                            verbose=0)
    aik.reset()
    n_joints = aik.get_n_joints()

    q = np.zeros(n_joints)
    p = np.empty(7)
    aik.jnt_to_cart(q, p)
    p_new = np.copy(p)
    p_new[:3] += np.array([0.1, 0.1, -0.1])
    q_new = np.copy(q)
    aik.cart_to_jnt(p, q_new)
    for _ in range(10):
        q_new = np.copy(q_new)
        aik.cart_to_jnt(p_new, q_new)
        for i in range(n_joints):
            assert_less_equal(abs(q[i] - q_new[i]), 0.10001)
        q = q_new


def test_reachable_path():
    urdf_path = os.path.join(DATA_PATH, "kuka_lbr.urdf")
    base_link = "kuka_lbr_l_link_0"
    ee_link = "kuka_lbr_l_link_7"

    aik = ApproxInvKin(urdf_path, base_link, ee_link, 1.0, 1.0, verbose=0)
    eik = ExactInvKin(urdf_path, base_link, ee_link, 1e-10, 1000, verbose=0)

    n_joints = aik.get_n_joints()
    n_steps = 101
    Q = np.empty((n_steps, n_joints))
    for i in range(n_joints):
        Q[:, i] = np.linspace(-0.5, 0.5, n_steps)

    p1 = np.empty(7)
    p1[:] = np.nan
    p2 = np.empty(7)
    p2[:] = np.nan
    qe = np.empty(n_joints)
    qe[:] = Q[0]
    qa = np.empty(n_joints)
    qa[:] = Q[0]
    for t in range(n_steps):
        if t > 0:
            qe[:] = Q[t - 1]
            qa[:] = Q[t - 1]

        # Exact IK seems to be closer in Cartesian space
        eik.jnt_to_cart(Q[t], p1)
        eik.cart_to_jnt(p1, qe)
        eik.jnt_to_cart(qe, p2)
        assert_array_almost_equal(p1, p2, decimal=6)
        assert_array_almost_equal(qe, Q[t], decimal=0)

        # Approximate IK seems to be closer in joint space
        aik.jnt_to_cart(Q[t], p1)
        aik.cart_to_jnt(p1, qa)
        aik.jnt_to_cart(qa, p2)
        assert_array_almost_equal(p1, p2, decimal=3)
        assert_array_almost_equal(qa, Q[t], decimal=2)


def test_unreachable_pose():
    urdf_path = os.path.join(DATA_PATH, "kuka_lbr.urdf")
    base_link = "kuka_lbr_l_link_0"
    ee_link = "kuka_lbr_l_link_7"

    aik = ApproxInvKin(urdf_path, base_link, ee_link, 1.0, 1.0, verbose=0)
    eik = ExactInvKin(urdf_path, base_link, ee_link, 1e-6, 1000, verbose=0)

    n_joints = aik.get_n_joints()
    q = 0.3 * np.ones(n_joints)
    p = np.empty(7)
    p[:] = np.nan
    eik.jnt_to_cart(q, p)
    p[2] += 0.1

    q_result = np.zeros(n_joints)
    p_result_exact = np.zeros(7)
    success = eik.cart_to_jnt(p, q_result)
    assert_false(success)
    eik.jnt_to_cart(q_result, p_result_exact)

    q_result = np.zeros(n_joints)
    p_result_approx = np.zeros(7)
    success = aik.cart_to_jnt(p, q_result)
    assert_false(success)
    aik.jnt_to_cart(q_result, p_result_approx)

    exact_dist = np.linalg.norm(p - p_result_exact)
    approx_dist = np.linalg.norm(p - p_result_approx)
    assert_less(approx_dist, exact_dist)


def test_missing_urdf():
    urdf_path = os.path.join(DATA_PATH, "missing.urdf")
    assert_raises(IOError, ApproxInvKin, urdf_path, "base", "ee", 1.0, 1.0)

