import pytransform.rotations as pyrot
from visualization import *
from utils import parse_args
from convert import trajectory_ik, trajectory_fk, timing_report, reaching_report
from approxik import ApproxInvKin, ExactInvKin


def line1(P, turns=5, max_radius=1.0, height=0.8):
    n_steps = P.shape[0]
    x = np.linspace(-np.pi * turns, np.pi, n_steps)
    r = np.linspace(0, max_radius, n_steps)
    P[:, 0] = np.sin(x) * r
    P[:, 1] = np.cos(x) * r
    P[:, 2] = height
    P[:, 3] = 1.0


if __name__ == "__main__":
    filename, base_link, ee_link = parse_args()

    aik = ApproxInvKin(filename, base_link, ee_link, 1.0, 0.001, verbose=0)
    eik = ExactInvKin(filename, base_link, ee_link, 1e-4, 200, verbose=0)

    P = np.zeros((1000, 7))
    line1(P, max_radius=0.6)

    Qa, timings = trajectory_ik(P, aik)
    reaching_report(P, Qa, aik, label="Approximate IK")
    timing_report(timings, "Approximate IK")
    Pa = trajectory_fk(Qa, aik)
    Qe, timings, reachable = trajectory_ik(P, eik, return_reachable=True)
    timing_report(timings, "Exact IK")
    reaching_report(P, Qe, eik, label="Exact IK")
    Pe = trajectory_fk(Qe, eik)
    Pe[np.logical_not(reachable)] = np.nan
    Qe[np.logical_not(reachable)] = np.nan

    ax = plot_pose_trajectory(P, Pa)
    ax = plot_pose_trajectory(P, Pe)

    plot_joint_trajectories([Qa, Qe], labels=["Approximation", "Exact"])

    plt.show()
