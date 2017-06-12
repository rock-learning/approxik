import numpy as np
from approxik import ApproxInvKin, ExactInvKin
from bolero.representation import (BlackBoxBehavior, DMPBehavior,
                                   CartesianDMPBehavior)


class IKCartesianDMPBehavior(BlackBoxBehavior):
    """Cartesian Space Dynamical Movement Primitive using Inverse Kinematics.

    This is a cartesian space dmp which uses the invesrse kinematics to
    position an end-effector.

    Parameters
    ----------
    ik : object
        Inverse kinematics solver

    execution_time : float, optional (default: 1)
        Execution time of the DMP in seconds.

    dt : float, optional (default: 0.01)
        Time between successive steps in seconds.

    n_features : int, optional (default: 50)
        Number of RBF features for each dimension of the DMP.

    configuration_file : string, optional (default: None)
        Name of a configuration file that should be used to initialize the DMP.
        If it is set all other arguments will be ignored.
    """
    def __init__(self, ik, execution_time=1.0, dt=0.01, n_features=50,
                 configuration_file=None):
        self.ik = ik
        self.csdmp = CartesianDMPBehavior(execution_time, dt, n_features,
                                          configuration_file)
        self.csdmp.init(7, 7)

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        self.n_joints = self.ik.get_n_joints()
        if self.n_joints * 3 != n_inputs:
            raise ValueError("Expected %d inputs, got %d"
                             % (self.n_joints * 3, n_inputs))
        if self.n_joints * 3 != n_outputs:
            raise ValueError("Expected %d inputs, got %d"
                             % (self.n_joints * 3, n_inputs))

        self.q = np.empty(self.n_joints)
        self.q[:] = np.nan
        self.p = np.empty(7)
        self.p[:] = np.nan
        self.success = None

    def reset(self):
        self.csdmp.reset()
        self.q[:] = 0.0

    def set_inputs(self, inputs):
        self.q[:] = inputs[:self.n_joints]

    def can_step(self):
        return self.csdmp.can_step()

    def step(self):
        """Compute forward kin., execute DMP step, perform inverse kin."""
        self.ik.jnt_to_cart(self.q, self.p)
        self.csdmp.set_inputs(self.p)
        self.csdmp.step()
        self.csdmp.get_outputs(self.p)
        self.success = self.ik.cart_to_jnt(self.p, self.q)

    def get_outputs(self, outputs):
        if self.success is not None and not self.success:
            outputs[:self.n_joints] = np.nan
        else:
            outputs[:self.n_joints] = self.q[:]

    def get_n_params(self):
        return self.csdmp.get_n_params()

    def get_params(self):
        return self.csdmp.get_params()

    def set_params(self, params):
        self.csdmp.set_params(params)

    def set_meta_parameters(self, keys, values):
        self.csdmp.set_meta_parameters(keys, values)

    def trajectory(self):
        return self.csdmp.trajectory()


qhi = np.array([2.94, 2.08, 2.94, 2.08, 2.94, 2.08, 3.03])
qlo = -qhi
x0 = np.zeros(7)
g = np.array([np.pi / 8, np.pi / 3, 0, -np.pi / 3, 0, np.pi / 3, 0])
execution_time = 1.0
dt = 0.01

cart_mp_keys = ["x0", "q0", "g", "qg"]


def _initial_cart_params(ik):
    joint_dmp = DMPBehavior(execution_time, dt)
    n_joints = ik.get_n_joints()
    joint_dmp.init(n_joints * 3, n_joints * 3)
    joint_dmp.set_meta_parameters(["x0", "g"], [x0, g])
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
        "../data/kuka_lbr.urdf", "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
        1.0, 0.001, maxiter=100000, verbose=0)

    p = np.empty(7)
    ik.jnt_to_cart(x0, p)
    x0_pos = p[:3].copy()
    x0_rot = p[3:].copy()
    ik.jnt_to_cart(g, p)
    g_pos = p[:3].copy()
    g_rot = p[3:].copy()
    cart_mp_values = [x0_pos, x0_rot, g_pos, g_rot]

    beh = IKCartesianDMPBehavior(ik, execution_time, dt)
    beh.csdmp.set_params(_initial_cart_params(ik))
    return ik, beh, cart_mp_keys, cart_mp_values


def make_exact_cart_dmp(x0, g, execution_time, dt):
    ik = ExactInvKin(
        "../data/kuka_lbr.urdf", "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
        1e-5, 150)

    p = np.empty(7)
    ik.jnt_to_cart(x0, p)
    x0_pos = p[:3].copy()
    x0_rot = p[3:].copy()
    ik.jnt_to_cart(g, p)
    g_pos = p[:3].copy()
    g_rot = p[3:].copy()
    cart_mp_values = [x0_pos, x0_rot, g_pos, g_rot]

    beh = IKCartesianDMPBehavior(ik, execution_time, dt)
    beh.csdmp.set_params(_initial_cart_params(ik))
    return ik, beh, cart_mp_keys, cart_mp_values


def make_joint_dmp(x0, g, execution_time, dt):
    ik = ExactInvKin(
        "../data/kuka_lbr.urdf", "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
        1e-5, 150)
    beh = DMPBehavior(execution_time, dt)
    mp_keys = ["x0", "g"]
    mp_values = [x0, g]
    return ik, beh, mp_keys, mp_values
