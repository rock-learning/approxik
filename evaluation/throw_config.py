import numpy as np
from approxik import ApproxInvKin, ExactInvKin
from ik_cartesian_dmp_behavior import (IKCartesianDMPBehavior,
                                       IKCartesianDMPBehaviorWithGoalParams)
from dmp_behavior import DMPBehavior, DMPBehaviorWithGoalParams
from bolero.representation import CartesianDMPBehavior


x0j = np.array([0.0, -0.5, -0.5, 0.0, -0.5, 0.0])
gj = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.0])

ik = ExactInvKin("../data/compi.urdf",
                 "linkmount", "linkkelle", 1e-5, 150)

x0 = np.zeros(7)
ik.jnt_to_cart(x0j, x0)
g = np.zeros(7)
ik.jnt_to_cart(gj, g)
execution_time = 0.5
dt = 0.001

cart_mp_keys = ["x0", "q0", "g", "qg"]
cart_mp_values = [x0[:3], x0[3:], g[:3], g[3:]]


def _initial_cart_params(ik):
    joint_dmp = DMPBehavior(execution_time, dt)
    n_joints = ik.get_n_joints()
    joint_dmp.init(n_joints, n_joints)
    joint_dmp.set_meta_parameters(["x0", "g"], [x0j, gj])
    Q = joint_dmp.trajectory()[0]
    P = np.empty((Q.shape[0], 7))
    for t in range(Q.shape[0]):
        ik.jnt_to_cart(Q[t], P[t])
    cart_dmp = CartesianDMPBehavior(execution_time, dt)
    cart_dmp.init(7, 7)
    cart_dmp.imitate(P.T[:, :, np.newaxis])
    return cart_dmp.get_params()


def make_approx_cart_dmp(x0, g, execution_time, dt):
    ik = ApproxInvKin(
        "../data/compi.urdf", "linkmount", "linkkelle",
        0.1, 1.0, maxiter=100000, verbose=0)
    beh = IKCartesianDMPBehaviorWithGoalParams(g, ik, execution_time, dt)
    beh.csdmp.set_params(_initial_cart_params(ik))
    return ik, beh, cart_mp_keys, cart_mp_values, g, 0.05 ** 2


def make_exact_cart_dmp(x0, g, execution_time, dt):
    beh = IKCartesianDMPBehaviorWithGoalParams(g, ik, execution_time, dt)
    beh.csdmp.set_params(_initial_cart_params(ik))
    return ik, beh, cart_mp_keys, cart_mp_values, g, 0.05 ** 2


def make_joint_dmp(x0, g, execution_time, dt):
    beh = DMPBehaviorWithGoalParams(gj, execution_time, dt)
    mp_keys = ["x0", "g"]
    mp_values = [x0j, gj]
    return ik, beh, mp_keys, mp_values, gj, (0.1 * np.pi) ** 2
