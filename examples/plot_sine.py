import pytransform.rotations as pyrot
from visualization import *
from utils import parse_args
from convert import trajectory_ik, trajectory_fk, timing_report, reaching_report
from approxik import ApproxInvKin, ExactInvKin


def line1(P):
    P[:, 0] = 0.5
    P[:, 1] = np.linspace(-1.0, 1.0, P.shape[0])
    P[:, 2] = 0.5 + 0.8 * np.sin(1.5 * np.pi * P[:, 1])
    P[:, 3] = 1.0


if __name__ == "__main__":
    filename, base_link, ee_link = parse_args()

    aik = ApproxInvKin(filename, base_link, ee_link, 1.0, 0.001, verbose=0)
    eik = ExactInvKin(filename, base_link, ee_link, 1e-4, 200, verbose=0)

    P = np.zeros((1000, 7))
    line1(P)

    Qa, timings = trajectory_ik(P, aik)
    timing_report(timings, "Approximate IK")
    reaching_report(P, Qa, aik, label="Approximate IK")
    Pa = trajectory_fk(Qa, aik)
    Qe, timings, reachable = trajectory_ik(P, eik, return_reachable=True)
    reaching_report(P, Qe, eik, label="Exact IK")
    timing_report(timings, "Exact IK")
    Pe = trajectory_fk(Qe, eik)
    Pe[np.logical_not(reachable)] = np.nan
    Qe[np.logical_not(reachable)] = np.nan

    ax = plot_pose_trajectory(P, Pa, direction=False)
    mark_origin(ax)
    plt.savefig("sine_approximation.png")
    ax = plot_pose_trajectory(P, Pe, direction=False)
    mark_origin(ax)
    plt.savefig("sine_exact.png")

    plot_joint_trajectories([Qa, Qe], labels=["Approximation", "Exact"])

    plt.show()
