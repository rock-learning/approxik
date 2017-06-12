import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import pytransform.rotations as pyrot


class Arrow3D(FancyArrowPatch):  # http://stackoverflow.com/a/11156353/915743
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]), (xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def _make_3d_axis(ax, plot_handler):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d", aspect="equal")
        plot_handler(ax)
    return ax


def plot_handler(ax):
    ax.set_xlim((-0.75, 0.75))
    ax.set_ylim((-0.75, 0.75))
    ax.set_zlim((-0.5, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=15, azim=-45)


def plot_pose_trajectory(
        P_original, P, ax=None, show_plot=False, color="k", direction=True,
        alpha=1.0, plot_handler=plot_handler):
    if len(P) == 0:
        raise ValueError("Trajectory does not contain any elements.")

    ax = _make_3d_axis(ax, plot_handler)

    r = np.all(np.isfinite(P), axis=1)  # reachable

    ax.plot(P_original[:, 0],
            P_original[:, 1],
            P_original[:, 2],
            lw=5, color=color, alpha=0.3)
    P_original_not_reachable = np.copy(P_original)
    P_original_not_reachable[r] = np.nan
    ax.plot(P_original_not_reachable[:, 0],
            P_original_not_reachable[:, 1],
            P_original_not_reachable[:, 2],
            lw=5, color="r", alpha=0.3)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=5, color=color, alpha=alpha)

    for t in range(P.shape[0]):
        ax.plot([P[t, 0], P_original[t, 0]], [P[t, 1], P_original[t, 1]],
                [P[t, 2], P_original[t, 2]], color="r")

    step = max(1, (P.shape[0] - 1) / 10)
    frame_indices = np.arange(0, P.shape[0], step)
    for t in frame_indices:
        if not r[t]:
            continue
        p = P[t]
        pyrot.plot_basis(ax, pyrot.matrix_from_quaternion(p[3:]), p[:3],
                         s=0.1, alpha=alpha)

    if direction and r[0] and r[-1]:
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


def mark_origin(ax):
    X = np.arange(-0.8, 0.8, 0.1)
    Y = np.arange(-0.8, 0.8, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, lw=0, antialiased=False,
        color=colorConverter.to_rgba("b", alpha=0.2))
    ax.scatter(0, 0, 0, s=200, c="k")


def plot_joint_trajectories(
    Qs, labels=None, fig=None,
    lo=np.array([-2.96706, -2.094395, -2.96706, -2.094395, -2.96706, -2.094395, -3.054326]),
    hi=np.array([2.96706, 2.094395, 2.96706, 2.094395, 2.96706, 2.094395, 3.054326]),
    jump_threshold=0.2):

    if fig is None:
        fig = plt.figure()

    n_trajectories = len(Qs)
    if n_trajectories > 0:
        n_steps, n_joints = Qs[0].shape

        for i in range(n_joints):
            ax = fig.add_subplot(n_joints, 1, i + 1)
            for j in range(n_trajectories):
                if labels is None:
                    ax.plot(Qs[j][:, i])
                else:
                    ax.plot(Qs[j][:, i], label="%s" % labels[j])
                d = np.abs(np.diff(Qs[j][:, i]))
                jump_indices = np.where(d > jump_threshold)
                ax.scatter(jump_indices, Qs[j][jump_indices, i], c="r")
                ax.set_xlim((0, n_steps))
                ax.set_ylim((lo[i], hi[i]))
                ax.set_yticks([-2, 0, 2])
            if i < n_joints - 1:
                ax.set_xticks(())
            else:
                ax.legend(loc="lower left", prop={"size": 8})

    plt.subplots_adjust(hspace=0.1)

    return fig
