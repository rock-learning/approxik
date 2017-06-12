import pytransform.rotations as pyrot
from visualization import *
from utils import parse_args
from convert import trajectory_ik, trajectory_fk, timing_report
from approxik import ApproxInvKin, ExactInvKin


if __name__ == "__main__":
    filename, base_link, ee_link = parse_args()

    aik = ApproxInvKin(filename, base_link, ee_link, 1.0, 0.001, verbose=2)
    eik = ExactInvKin(filename, base_link, ee_link, 1e-4, 200, verbose=2)

    P = np.array([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]])

    Qa, timings = trajectory_ik(P, aik)
    timing_report(timings, "Approximate IK")
    Pa = trajectory_fk(Qa, aik)
    Qe, timings, reachable = trajectory_ik(P, eik, return_reachable=True)
    timing_report(timings, "Exact IK")
    Pe = trajectory_fk(Qe, eik)
    Pe[np.logical_not(reachable)] = np.nan
    Qe[np.logical_not(reachable)] = np.nan

    ax = plot_pose_trajectory(P, Pa)
    ax = plot_pose_trajectory(P, P, ax=ax)

    plt.show()
