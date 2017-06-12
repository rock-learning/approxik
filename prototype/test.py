from kuka_ik_kdl import KukaLRBIK
import numpy as np
import pytransform.rotations as pyrot
from scipy.optimize import fmin_l_bfgs_b


def approxInvKin(w, R, p, q, limits, fk):
    quat = pyrot.quaternion_from_matrix(R)
    def objective(q):
        Rm, pm = fk.forward(q)
        quatm = pyrot.quaternion_from_matrix(Rm)
        qd = pyrot.quaternion_dist(quat, quatm)
        return np.sum(w[:3] * (p - pm) ** 2) + w[3] * qd

    q_new, e, _ = fmin_l_bfgs_b(objective, q, approx_grad=True, bounds=limits)
    return q_new, e


def approx_trajectory(P, w, fk, q_bounds):
    Q = np.empty((P.shape[0], 7))
    q = np.zeros(7)
    for t in range(P.shape[0]):
        R = pyrot.matrix_from_quaternion(P[t, 3:])
        q, _ = approxInvKin(w, R, P[t, :3], q, q_bounds, fk)
        Q[t, :] = q
    return Q


def exact_trajectory(P, ik):
    Q = np.empty((P.shape[0], 7))
    ik.q = np.zeros(7)
    for t in range(P.shape[0]):
        R = pyrot.matrix_from_quaternion(P[t, 3:])
        ik.set_point(R=R, p=P[t, :3])
        if ik.check_reachable():
            Q[t, :] = ik.get_q()
        else:
            Q[t, :] = np.nan
    return Q


def fk_trajectory(Q, fk):
    P = np.empty((Q.shape[0], 7))
    for t in range(Q.shape[0]):
        R, p = fk.forward(Q[t])
        P[t, :3] = p
        P[t, 3:] = pyrot.quaternion_from_matrix(R)
    return P


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):  # http://stackoverflow.com/a/11156353/915743
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def _make_3d_axis(ax, plot_handler, window_offset=True):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        #if window_offset:
        #    plt.get_current_fig_manager().window.wm_geometry("+1200+40")
        #plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        ax = fig.add_subplot(111, projection="3d", aspect="equal")
        plot_handler(ax)
    return ax


def plot_handler(ax):
    ax.set_xlim((-0.25, 0.5))
    ax.set_ylim((-0.25, 0.5))
    ax.set_zlim((-0.0, 0.75))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_pose_trajectory(P_original, P, ax=None, show_plot=False, color="k", direction=True,
                         alpha=1.0, plot_handler=plot_handler, window_offset=True):
    if len(P) == 0:
        raise ValueError("Trajectory does not contain any elements.")

    ax = _make_3d_axis(ax, plot_handler, window_offset)

    P[np.logical_not(np.isfinite(P))] = 0.0

    ax.plot(P_original[:, 0], P_original[:, 1], P_original[:, 2], lw=3, color=color, alpha=0.3)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=3, color=color, alpha=alpha)
    for t in range(P.shape[0]):
        #print np.round(P[t], 2), np.round(P_original[t], 2)
        ax.plot([P[t, 0], P_original[t, 0]], [P[t, 1], P_original[t, 1]],
                [P[t, 2], P_original[t, 2]], color="r", ls="o")
    for p in P[::(P.shape[0] - 1) / 10]:
        pyrot.plot_basis(ax, pyrot.matrix_from_quaternion(p[3:]), p[:3], s=0.03,
                         alpha=alpha)

    if direction:
        s = P[0, :3]
        g = P[-1, :3]
        start = s + 0.2 * (g - s)
        goal = g - 0.2 * (g - s)
        ax.add_artist(Arrow3D([start[0], goal[0]],
                              [start[1], goal[1]],
                              [start[2], goal[2]],
                              mutation_scale=20, lw=1, arrowstyle="-|>",
                              color="k"))

    if show_plot:
        plt.show()

    return ax


def plot_joint_trajectories(
    Qs, labels=None, fig=None,
    lo=np.array([-2.96706, -2.094395, -2.96706, -2.094395, -2.96706, -2.094395, -3.054326]),
    hi=np.array([2.96706, 2.094395, 2.96706, 2.094395, 2.96706, 2.094395, 3.054326])):

    if fig is None:
        fig = plt.figure()

    n_trajectories = len(Qs)
    if n_trajectories > 0:
        n_steps, n_joints = Qs[0].shape

        for j in range(n_trajectories):
            for i in range(n_joints):
                ax = fig.add_subplot(n_joints, 1, i + 1)
                if labels is None:
                    ax.plot(Qs[j][:, i])
                else:
                    ax.plot(Qs[j][:, i],
                            label="%s, joint #%d" % (labels[j], (i + 1)))
                ax.set_xlim((0, n_steps))
                ax.set_ylim((lo[i], hi[i]))
                ax.legend(prop={"size": 8})

    plt.subplots_adjust(wspace=0.2, hspace=0.5)

    return fig


if __name__ == "__main__":
    q_bounds = np.array([
        [-2.94, 2.94],
        [-2.08, 2.08],
        [-2.94, 2.94],
        [-2.08, 2.08],
        [-2.94, 2.94],
        [-2.08, 2.08],
        [-3.03, 3.03]])
    fk = KukaLRBIK()
    q = np.zeros(7)

    P = np.zeros((100, 7))
    P[:, 0] = 0.5
    P[:, 2] = np.linspace(0.5, 1.0, P.shape[0])
    P[:, 3] = 1.0
    w = np.array([1.0, 1.0, 1.0, 1e-3])

    Qa = approx_trajectory(P, w, fk, q_bounds)
    Pa = fk_trajectory(Qa, fk)
    Qe = exact_trajectory(P, fk)
    Pe = fk_trajectory(Qe, fk)

    plot_pose_trajectory(P, Pa)
    plot_pose_trajectory(P, Pe)

    plot_joint_trajectories([Qa, Qe])
    plt.show()
