# Authors: Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import numpy as np
from scipy.spatial.distance import cdist
from bolero.environment import Environment
from bolero.utils.log import get_logger
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from pytransform.rotations import matrix_from_quaternion


class ObstacleEnvironment(Environment):
    """Optimize a trajectory that avoids several obstacles.

    Parameters
    ----------
    ik : object
        Inverse kinematics solver

    x0 : array-like, shape = (7,)
        Initial position.

    g : array-like, shape = (n_task_dims,), optional (default: [1, 1])
        Goal position.

    obstacles : array-like, shape = (n_obstacles, 3)
        Via points: (t, x, y, z)

    execution_time : float
        Execution time in seconds

    dt : float
        Time between successive steps in seconds.

    qlo : array-like, shape (n_joints,)
        Lower joint limits

    qhi : array-like, shape (n_joints,)
        Upper joint limits

    penalty_goal_dist : float, optional (default: 0)
        Penalty weight for distance to goal at the end

    penalty_vel : float, optional (default: 0)
        Penalty weight for velocities

    penalty_acc : float, optional (default: 0)
        Penalty weight for accelerations

    obstacle_dist : float, optional (default: 0.1)
        Distance that should be kept to the obstacles (penalty is zero outside
        of this area)

    penalty_obstacle : float, optional (default: 0)
        Penalty weight for obstacle avoidance

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output
    """
    def __init__(self, ik, x0, g, obstacles, execution_time, dt, qlo, qhi,
                 penalty_goal_dist=0.0, penalty_vel=0.0, penalty_acc=0.0,
                 obstacle_dist=0.1, penalty_obstacle=0.0,
                 log_to_file=False, log_to_stdout=False):
        self.ik = ik
        self.x0 = x0
        self.g = g
        self.obstacles = obstacles
        self.execution_time = execution_time
        self.dt = dt
        self.qlo = qlo
        self.qhi = qhi
        self.penalty_goal_dist = penalty_goal_dist
        self.penalty_vel = penalty_vel
        self.penalty_acc = penalty_acc
        self.obstacle_dist = obstacle_dist
        self.penalty_obstacle = penalty_obstacle
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

    def _get_collision_penalties(self, obstacle_filter=None):
        if obstacle_filter is None:
            obstacles = self.obstacles
        else:
            obstacles = np.asarray(self.obstacles)[obstacle_filter, :]
        distances = cdist(self.P[:, :3], obstacles)
        collision_penalties = np.maximum(0., 1.0 - distances /
                                         self.obstacle_dist)
        return collision_penalties

    def get_obstacle_is_colliding(self):
        if self.obstacles is None:
            return np.zeros(len(self.obstacles))
        collision_penalties = self._get_collision_penalties()
        return collision_penalties.max(axis=0) > 0.0

    def get_collision(self, obstacle_filter=None):
        if self.obstacles is None:
            return np.zeros(self.t)
        collision_penalties = self._get_collision_penalties(obstacle_filter)
        #self.logger.info("Distances to obstacles: %r" % distances.min(axis=0))
        self.logger.info("Distances penalties: %r"
                         % np.round(collision_penalties.max(axis=0), 2))
        collisions = collision_penalties.sum(axis=1)
        return collisions

    def get_num_obstacles(self):
        if self.obstacles is None:
            return 0
        return self.obstacles.shape[0]

    def get_feedback(self):
        rewards = np.zeros(self.t)
        if self.penalty_goal_dist > 0.0:
            goal_dist = np.linalg.norm(self.g[:3] - self.P[-1, :3])
            self.logger.info("Distance to goal: %.3f (* %.2f)"
                             % (goal_dist, self.penalty_goal_dist))
            self.logger.info("Goal: %s" % self.g)
            self.logger.info("last position: %s" % self.P[-1])
            rewards[-1] -= goal_dist * self.penalty_goal_dist
        if self.penalty_vel > 0.0:
            rewards -= self.get_speed() * self.penalty_vel
        if self.penalty_acc > 0.0:
            rewards -= self.get_acceleration() * self.penalty_acc
        if self.obstacles is not None and self.penalty_obstacle > 0.0:
            rewards -= self.penalty_obstacle * self.get_collision()
        return rewards

    def is_behavior_learning_done(self):
        return False

    def get_maximum_feedback(self):
        return 0.0

    def plot(self):
        plt.figure()
        ax = plt.subplot(111, projection="3d", aspect="equal")

        ax.scatter(self.x0[0], self.x0[1], self.x0[2], marker="^", c="k", s=100)
        ax.scatter(self.g[0], self.g[1], self.g[2], c="k", s=100)
        is_colliding = self.get_obstacle_is_colliding()
        for i, obstacle in enumerate(self.obstacles):
            phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
            x = obstacle[0] + self.obstacle_dist * np.sin(phi) * np.cos(theta)
            y = obstacle[1] + self.obstacle_dist * np.sin(phi) * np.sin(theta)
            z = obstacle[2] + self.obstacle_dist * np.cos(phi)
            color = "r" if is_colliding[i] else "k"
            ax.plot_surface(x, y, z,  rstride=5, cstride=5, color=color,
                            alpha=0.1, linewidth=0)
            ax.plot_wireframe(x, y, z,  rstride=20, cstride=20, color=color,
                              alpha=0.1)

        #plot_trajectory(ax=ax, P=self.P, s=0.1, lw=2, c="k",
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
        ax.set_ylim((-0.9, -0.1))
        ax.set_zlim((0.4, 1.2))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        return ax
