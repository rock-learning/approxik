# Authors: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import cdist
from bolero.environment import Environment
from bolero.utils.log import get_logger
from pytransform.urdf import UrdfTransformManager


JOINTNAMES = ["kuka_lbr_l_joint_%d" % i for i in range(1, 8)]
LINKNAMES = ["kuka_lbr_l_link_%d" % i for i in range(8)]


def forward(tm, q, tips=["kuka_lbr_l_link_%d" % i for i in range(8)]):
    for i, angle in enumerate(q):
        tm.set_joint(JOINTNAMES[i], angle)
    positions = np.empty((len(tips), 3))
    for i, tip in enumerate(tips):
        positions[i] = tm.get_transform(tip, LINKNAMES[0])[:3, 3]
    return positions


def displacement(mass_ee, mass_sphere, radius_pendulum, vel):
    # approximation: ignore z component
    vel = vel[:2]

    # approximation: elastic collision
    speed = np.linalg.norm(vel)
    sphere_speed = 2 * (mass_ee * speed) / (mass_ee + mass_sphere)
    height = 0.5 * sphere_speed ** 2 / 9.81
    height = np.clip(height, 0, 2 * radius_pendulum)
    displacement_xyplane = np.sin(np.arccos(
        (radius_pendulum - height) / radius_pendulum)) * radius_pendulum

    # approximation: direction of motion is directly transmitted
    swing_angle = np.arctan2(vel[1], vel[0])
    pos_xyplane = displacement_xyplane * np.array([np.cos(swing_angle),
                                                   np.sin(swing_angle)])

    return np.hstack((pos_xyplane, (height,)))


class Pendulum(Environment):
    """Optimize a trajectory according to some criteria.

    Parameters
    ----------
    x0 : array-like, shape = (n_task_dims,), optional (default: [0, 0])
        Initial position.

    g : array-like, shape = (n_task_dims,), optional (default: [1, 1])
        Goal position.

    execution_time : float, optional (default: 1.0)
        Execution time in seconds

    dt : float, optional (default: 0.01)
        Time between successive steps in seconds.

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output
    """
    def __init__(self,
                 x0=np.zeros(7),
                 g=np.zeros(7),
                 execution_time=1.0,
                 dt=0.01,
                 log_to_file=False,
                 log_to_stdout=False):
        self.x0 = x0
        self.g = g
        self.execution_time = execution_time
        self.dt = dt
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout

    def init(self):
        """Initialize environment."""
        self.x0 = np.asarray(self.x0)
        self.g = np.asarray(self.g)
        self.n_task_dims = self.x0.shape[0]
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

        n_steps = 1 + int(self.execution_time / self.dt)
        self.X = np.empty((n_steps, self.n_task_dims))
        self.xd = np.empty(self.n_task_dims)
        self.xdd = np.empty(self.n_task_dims)

        self.tm = UrdfTransformManager()
        self.tm.load_urdf(open("../data/kuka_lbr.urdf", "r"))

        self.sphere_pos = np.array([0.6, 0.0, 1.0])
        self.pendulum_radius = 0.5
        self.sphere_radius = 0.1
        # We set the target so that the trajectory has to be modified
        # significantly. It has to hit the pendulum from another direction.
        self.target_pos = np.array([
            0.6 + self.pendulum_radius * np.cos(0.6 * np.pi),
            self.pendulum_radius * np.sin(0.6 * np.pi),
            1.5])
        self.pendulum_mass = 1.0
        self.ee_mass = 30.0

    def reset(self):
        """Reset state of the environment."""
        self.t = 0

    def get_num_inputs(self):
        """Get number of environment inputs.

        Returns
        -------
        n : int
            number of environment inputs
        """
        return 3 * self.n_task_dims

    def get_num_outputs(self):
        """Get number of environment outputs.

        Returns
        -------
        n : int
            number of environment outputs
        """
        return 3 * self.n_task_dims

    def get_outputs(self, values):
        """Get environment outputs.

        Parameters
        ----------
        values : array
            Outputs of the environment: positions, velocities and accelerations
            in that order, e.g. for n_task_dims=2 the order would be xxvvaa
        """
        if self.t == 0:
            values[:self.n_task_dims] = np.copy(self.x0)
            values[self.n_task_dims:-self.n_task_dims] = np.zeros(
                self.n_task_dims)
            values[-self.n_task_dims:] = np.zeros(self.n_task_dims)
        else:
            values[:self.n_task_dims] = self.X[self.t - 1]
            values[self.n_task_dims:-self.n_task_dims] = self.xd
            values[-self.n_task_dims:] = self.xdd

    def set_inputs(self, values):
        """Set environment inputs, e.g. next action.

        Parameters
        ----------
        values : array,
            Inputs for the environment: positions, velocities and accelerations
            in that order, e.g. for n_task_dims=2 the order would be xxvvaa
        """
        if np.all(np.isfinite(values[:self.n_task_dims])):
            self.X[self.t, :] = values[:self.n_task_dims]
            self.xd[:] = values[self.n_task_dims:-self.n_task_dims]
            self.xdd[:] = values[-self.n_task_dims:]
        else:
            self.X[self.t, :] = self.X[self.t - 1, :]

    def step_action(self):
        """Execute step perfectly."""
        self.t += 1

    def is_evaluation_done(self):
        """Check if the time is over.

        Returns
        -------
        finished : bool
            Is the episode finished?
        """
        return self.t * self.dt > self.execution_time

    def get_feedback(self):
        """Get reward per timestamp based on weighted criteria (penalties)

        Returns
        -------
        rewards : array-like, shape (n_steps,)
            reward for every timestamp; non-positive values
        """
        colliding = False
        collision_velocity = None
        collision_happened = False

        for t in range(len(self.X)):
            p = forward(self.tm, self.X[t], tips=["kuka_lbr_l_link_7"])[0]
            if colliding:
                collision_velocity -= p
                collision_velocity /= -self.dt
                collision_happened = True
                break
            elif np.linalg.norm(p - self.sphere_pos) < 2 * self.sphere_radius:
                colliding = True
                collision_velocity = p.copy()

        sphere_displacement = self.sphere_pos.copy()
        if collision_happened:
            d = displacement(self.ee_mass, self.pendulum_mass,
                             self.pendulum_radius, collision_velocity)
            sphere_displacement += d
        distance = np.linalg.norm(sphere_displacement - self.target_pos)

        return np.array([-distance ** 2])

    def is_behavior_learning_done(self):
        """Check if the behavior learning is finished.

        Returns
        -------
        finished : bool
            Always false
        """
        return False

    def get_maximum_feedback(self):
        """Returns the maximum sum of feedbacks obtainable."""
        return 0.0

    def plot(self, ax=None):
        """Plot a two-dimensional environment.

        Parameters
        ----------
        ax : Axis, optional
            Matplotlib axis
        """
        if ax is None:
            ax = plt.subplot(111, projection="3d")

        ax.scatter(self.sphere_pos[0], self.sphere_pos[1], self.sphere_pos[2],
                   c="gray", s=200)
        ax.scatter(self.sphere_pos[0], self.sphere_pos[1],
                   self.sphere_pos[2] + self.pendulum_radius, c="k", s=100)
        ax.plot((self.sphere_pos[0], self.sphere_pos[0]),
                (self.sphere_pos[1], self.sphere_pos[1]),
                (self.sphere_pos[2] + self.pendulum_radius, self.sphere_pos[2]),
                c="gray")
        ax.scatter(self.target_pos[0], self.target_pos[1], self.target_pos[2],
                   c="orange", s=100)

        initial_positions = forward(self.tm, self.x0)
        for p in initial_positions:
            ax.scatter(p[0], p[1], p[2], c="r", s=100)
        final_positions = forward(self.tm, self.g)
        for p in final_positions:
            ax.scatter(p[0], p[1], p[2], c="g", s=100)

        P = []
        colliding = False
        collision_velocity = None
        collision_happened = 0

        for t in range(len(self.X)):
            p = forward(self.tm, self.X[t])
            P.append(p)
            if colliding:
                collision_velocity -= p[-1]
                collision_velocity /= -self.dt
                colliding = False
            elif not collision_happened and np.linalg.norm(p[-1] - self.sphere_pos) < 0.1:
                colliding = True
                collision_velocity = p[-1].copy()
                collision_happened = t

        P = np.array(P)
        for i in range(len(LINKNAMES)):
            ax.plot(P[:, i, 0], P[:, i, 1], P[:, i, 2], c="k", ls="--")

        if collision_happened:
            v = collision_velocity
            v_norm = 0.3 * v / np.linalg.norm(v)
            ax.plot(
                [P[collision_happened, -1, 0],
                 P[collision_happened, -1, 0] + v_norm[0]],
                [P[collision_happened, -1, 1],
                 P[collision_happened, -1, 1] + v_norm[1]],
                [P[collision_happened, -1, 2],
                 P[collision_happened, -1, 2] + v_norm[2]]
            )
            d = displacement(self.ee_mass, self.pendulum_mass,
                             self.pendulum_radius, v)
            sphere_displacement = self.sphere_pos + d
            ax.scatter(sphere_displacement[0],
                       sphere_displacement[1],
                       sphere_displacement[2], c="k", s=200)
            ax.plot((self.sphere_pos[0], sphere_displacement[0]),
                    (self.sphere_pos[1], sphere_displacement[1]),
                    (self.sphere_pos[2] + self.pendulum_radius,
                     sphere_displacement[2]),
                    c="k")

        ax.set_xlim((-1.0, 1.5))
        ax.set_ylim((-1.25, 1.25))
        ax.set_zlim((-0.25, 2.25))

        return ax
