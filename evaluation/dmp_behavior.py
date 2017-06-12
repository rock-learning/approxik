# Author: Alexander Fabisch <Alexander.Fabisch@dfki.de>

import numpy as np
from bolero.representation import BlackBoxBehavior
from bolero.representation import DMPBehavior as DMPBehaviorImpl


class DMPBehavior(BlackBoxBehavior):
    """Dynamical Movement Primitive.

    Parameters
    ----------
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
    def __init__(self, execution_time=1.0, dt=0.01, n_features=50,
                 configuration_file=None):
        self.dmp = DMPBehaviorImpl(execution_time, dt, n_features,
                                   configuration_file)

    def init(self, n_inputs, n_outputs):
        """Initialize the behavior.

        Parameters
        ----------
        n_inputs : int
            number of inputs

        n_outputs : int
            number of outputs
        """
        self.dmp.init(3 * n_inputs, 3 * n_outputs)
        self.n_joints = n_inputs
        self.x = np.empty(3 * self.n_joints)
        self.x[:] = np.nan

    def reset(self):
        self.dmp.reset()
        self.x[:] = 0.0

    def set_inputs(self, inputs):
        self.x[:self.n_joints] = inputs[:]

    def can_step(self):
        return self.dmp.can_step()

    def step(self):
        self.dmp.set_inputs(self.x)
        self.dmp.step()
        self.dmp.get_outputs(self.x)

    def get_outputs(self, outputs):
        outputs[:] = self.x[:self.n_joints]

    def get_n_params(self):
        return self.dmp.get_n_params()

    def get_params(self):
        return self.dmp.get_params()

    def set_params(self, params):
        self.dmp.set_params(params)

    def set_meta_parameters(self, keys, values):
        self.dmp.set_meta_parameters(keys, values)

    def trajectory(self):
        return self.dmp.trajectory()


class DMPBehaviorWithGoalParams(DMPBehavior):
    def __init__(self, goal, execution_time=1.0, dt=0.01, n_features=50,
                 configuration_file=None):
        super(DMPBehaviorWithGoalParams, self).__init__(
            execution_time, dt, n_features, configuration_file)
        self.params = np.copy(goal)

    def set_meta_parameters(self, keys, values):
        self.dmp.set_meta_parameters(keys, values)
        self.set_params(self.params)

    def get_n_params(self):
        return len(self.params)

    def get_params(self):
        return self.params

    def set_params(self, params):
        self.params[:] = params
        self.dmp.set_meta_parameters(["g"], [self.params])
