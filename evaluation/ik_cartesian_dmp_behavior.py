# Authors: Manuel Meder <Manuel.Meder@dfki.de>
#          Alexander Fabisch <Alexander.Fabisch@dfki.de>

import numpy as np
from bolero.representation import BlackBoxBehavior, CartesianDMPBehavior


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
        if self.n_joints != n_inputs:
            raise ValueError("Expected %d inputs, got %d"
                             % (self.n_joints, n_inputs))
        if self.n_joints != n_outputs:
            raise ValueError("Expected %d inputs, got %d"
                             % (self.n_joints, n_inputs))

        self.q = np.empty(self.n_joints)
        self.q[:] = np.nan
        self.p = np.empty(7)
        self.p[:] = np.nan
        self.success = None

    def reset(self):
        self.csdmp.reset()
        self.q[:] = 0.0

    def set_inputs(self, inputs):
        self.q[:] = inputs[:]

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
            outputs[:] = np.nan
        else:
            outputs[:] = self.q[:]

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


class IKCartesianDMPBehaviorWithGoalParams(IKCartesianDMPBehavior):
    def __init__(self, goal, ik, execution_time=1.0, dt=0.01, n_features=50,
                 configuration_file=None):
        super(IKCartesianDMPBehaviorWithGoalParams, self).__init__(
            ik, execution_time, dt, n_features, configuration_file)
        self.params = np.copy(goal)

    def set_meta_parameters(self, keys, values):
        self.csdmp.set_meta_parameters(keys, values)
        self.set_params(self.params)

    def get_n_params(self):
        return len(self.params)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params[:] = params
        g = params[:3]
        qg = params[3:]
        qg = qg / np.linalg.norm(qg)
        self.csdmp.set_meta_parameters(["g", "qg"], [g, qg])
