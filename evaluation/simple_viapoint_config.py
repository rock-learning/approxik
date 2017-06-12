import numpy as np
from approxik import ApproxInvKin, ExactInvKin
from ik_cartesian_dmp_behavior import IKCartesianDMPBehavior
from dmp_behavior import DMPBehavior


qhi = np.array([2.94, 2.08, 2.94, 2.08, 2.94, 2.08, 3.03])
qlo = -qhi
x0 = np.array([-0.2, -0.5, 0.8, 0.0, 1.0, 0.0, 0.0])
g = np.array([0.2, -0.5, 0.8, 0.0, 1.0, 0.0, 0.0])
execution_time = 1.0
dt = 0.01
via_points = np.array([
    [0.5, 0.0, -0.5, 0.5],
    ])
penalty_vel = 0.0
penalty_acc = 0.0
penalty_via_point = 100.0

variance = {
    "approxik": 200.0 ** 2,
    "exactik": 200.0 ** 2,
    "joint": 500.0 ** 2
}

cart_mp_keys = ["x0", "q0", "g", "qg"]
cart_mp_values = [x0[:3], x0[3:], g[:3], g[3:]]


def make_approx_cart_dmp(x0, g, execution_time, dt):
    ik = ApproxInvKin(
        "../data/kuka_lbr.urdf", "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
        1.0, 0.001, maxiter=100000, verbose=0)
    beh = IKCartesianDMPBehavior(ik, execution_time, dt)
    return ik, beh, cart_mp_keys, cart_mp_values


def make_exact_cart_dmp(x0, g, execution_time, dt):
    ik = ExactInvKin(
        "../data/kuka_lbr.urdf", "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
        1e-5, 150)
    beh = IKCartesianDMPBehavior(ik, execution_time, dt)
    return ik, beh, cart_mp_keys, cart_mp_values


def make_joint_dmp(x0, g, execution_time, dt):
    ik = ExactInvKin(
        "../data/kuka_lbr.urdf", "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
        1e-5, 150)
    beh = DMPBehavior(execution_time, dt)
    mp_keys = ["x0", "g"]
    joints_start = np.zeros(ik.get_n_joints())
    ik.cart_to_jnt(x0, joints_start)
    joints_goal = joints_start.copy()
    ik.cart_to_jnt(g, joints_goal)
    mp_values = [joints_start, joints_goal]
    return ik, beh, mp_keys, mp_values
