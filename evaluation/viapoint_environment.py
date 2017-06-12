# Authors: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from scipy.spatial.distance import cdist
from bolero.environment import Environment
from bolero.utils.log import get_logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from pytransform.rotations import matrix_from_quaternion


class ViaPointEnvironment(Environment):
    """Optimize a trajectory that passes several viapoints.

    Parameters
    ----------
    ik : object
        Inverse kinematics solver

    x0 : array-like, shape = (7,)
        Initial position.

    via_points : array-like, shape = (n_via_points, 4)
        Via points: (t, x, y, z)

    execution_time : float
        Execution time in seconds

    dt : float
        Time between successive steps in seconds.

    qlo : array-like, shape (n_joints,)
        Lower joint limits

    qhi : array-like, shape (n_joints,)
        Upper joint limits

    penalty_vel : float, optional (default: 0)
        Penalty weight for velocities

    penalty_acc : float, optional (default: 0)
        Penalty weight for accelerations

    penalty_via_point : float, optional (default: 0)
        Penalty weight for distance to via points

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output
    """
    def __init__(self, ik, x0, via_points, execution_time, dt, qlo, qhi,
                 penalty_vel=0.0, penalty_acc=0.0, penalty_via_point=0.0,
                 log_to_file=False, log_to_stdout=False):
        self.ik = ik
        self.x0 = x0
        self.via_points = via_points
        self.execution_time = execution_time
        self.dt = dt
        self.qlo = qlo
        self.qhi = qhi
        self.penalty_vel = penalty_vel
        self.penalty_acc = penalty_acc
        self.penalty_via_point = penalty_via_point
        self.log_to_file = log_to_file
        self.log_to_stdout = log_to_stdout

    def init(self):
        self.x0 = np.asarray(self.x0)
        self.logger = get_logger(self, self.log_to_file, self.log_to_stdout)

        self.n_steps = 1 + int(self.execution_time / self.dt)
        self.n_joints = self.ik.get_n_joints()
        self.P = np.empty((self.n_steps, 7))
        self.Q = np.empty((self.n_steps, self.n_joints))
        self.p = np.empty(7)
        self.q = np.empty(self.n_joints)

    def reset(self):
        self.p[:] = self.x0.copy()
        self.q[:] = 0.0
        self.ik.cart_to_jnt(self.p, self.q)
        self.t = 0

    def get_num_inputs(self):
        return self.n_joints

    def get_num_outputs(self):
        return self.n_joints

    def get_outputs(self, values):
        values[:] = self.q[:]

    def set_inputs(self, values):
        if np.isfinite(values).all():
            self.q[:] = np.clip(values[:], self.qlo, self.qhi)

    def step_action(self):
        self.ik.jnt_to_cart(self.q, self.p)
        self.P[self.t, :] = self.p[:]
        self.Q[self.t, :] = self.q[:]
        self.t += 1

    def is_evaluation_done(self):
        return self.t >= self.n_steps

    def get_speed(self):
        Qd = np.vstack((np.zeros(self.n_joints), np.diff(self.Q, axis=0) / self.dt))
        speed = np.sqrt(np.sum(Qd ** 2, axis=1))
        self.logger.info("Speed: %g" % speed.sum())
        return speed

    def get_acceleration(self):
        Qd = np.vstack((np.zeros(self.n_joints), np.diff(self.Q, axis=0) / self.dt))
        Qdd = np.vstack((np.zeros(self.n_joints), np.diff(Qd, axis=0) / self.dt))
        acceleration = np.sqrt(np.sum(Qdd ** 2, axis=1))
        self.logger.info("Accelerations: %g" % acceleration.sum())
        return acceleration

    def get_via_point_dist(self):
        """Get list of collisions with obstacles during the performed movement.

        Returns
        -------
        min_dist : array-like, shape (n_via_points,)
            Minimum distances to all via points
        """
        dists = np.empty(len(self.via_points))
        for i, via_point in enumerate(self.via_points):
            t = int(via_point[0] / self.dt)
            p = via_point[1:]
            dists[i] = np.linalg.norm(p - self.P[t, :3])
        self.logger.info("Distances: %r" % dists)
        return dists

    def get_feedback(self):
        rewards = np.zeros(self.t)
        if self.penalty_vel > 0.0:
            rewards -= self.get_speed() * self.penalty_vel
        if self.penalty_acc > 0.0:
            rewards -= self.get_acceleration() * self.penalty_acc
        if self.penalty_via_point > 0.0:
            rewards[-1] -= self.penalty_via_point * self.get_via_point_dist().sum()
        return rewards

    def is_behavior_learning_done(self):
        return False

    def get_maximum_feedback(self):
        return 0.0

    def plot(self):
        plt.figure()
        ax = plt.subplot(111, projection="3d", aspect="equal")

        #ax.scatter(self.x0[0], self.x0[1], self.x0[2], c="r", s=100)
        for viapoint in self.via_points:
            ax.scatter(viapoint[1], viapoint[2], viapoint[3], c="k", s=100)
        # TODO sort by time
        ax.plot(self.via_points[:, 1], self.via_points[:, 2],
                self.via_points[:, 3], c="k", alpha=0.5)

        #plot_trajectory(ax=ax, P=self.P, s=0.05, lw=2, c="k",
        #                show_direction=False)
        ax.plot(self.P[:, 0], self.P[:, 1], self.P[:, 2], c="k")
        key_frames = np.linspace(0, self.P.shape[0] - 1, 10).astype(np.int)
        s = 0.1
        for p in self.P[key_frames]:
            R = matrix_from_quaternion(p[3:])
            for d in range(3):
                ax.plot([p[0], p[0] + s * R[0, d]],
                        [p[1], p[1] + s * R[1, d]],
                        [p[2], p[2] + s * R[2, d]],
                        color="k")

        ax.set_xlim((-0.4, 0.4))
        ax.set_ylim((-0.9, 0.1))
        ax.set_zlim((0.2, 1.0))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        return ax
