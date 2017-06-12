import sys
import pytransform.rotations as pyrot
from visualization import *
from utils import parse_args
from convert import trajectory_ik, trajectory_fk, timing_report, reaching_report
from approxik import ApproxInvKin, ExactInvKin


def print_usage():
    print("Usage: <script> <filename> [<base_link> <endeffector_link>] "
          "trajectory <file>")


if __name__ == "__main__":
    try:
        i = sys.argv.index("trajectory")
    except ValueError:
        print_usage()
        exit(1)
    if len(sys.argv) < i + 2:
        print_usage()
        exit(1)

    trajectory_filename = sys.argv[i + 1]
    print("Using trajectory from file '%s'" % trajectory_filename)

    filename, base_link, ee_link = parse_args()

    aik = ApproxInvKin(filename, base_link, ee_link, 1.0, 0.001, verbose=0)
    eik = ExactInvKin(filename, base_link, ee_link, 1e-4, 200, verbose=0)

    P = np.loadtxt(trajectory_filename)

    Qa, timings = trajectory_ik(P, aik)
    reaching_report(P, Qa, aik, label="Approximate IK")
    timing_report(timings, "Approximate IK")
    Pa = trajectory_fk(Qa, aik)
    Qe, timings, reachable = trajectory_ik(P, eik, return_reachable=True)
    reaching_report(P, Qe, eik, label="Exact IK")
    timing_report(timings, "Exact IK")
    Pe = trajectory_fk(Qe, eik)
    Pe[np.logical_not(reachable)] = np.nan
    Qe[np.logical_not(reachable)] = np.nan

    ax = plot_pose_trajectory(P, Pa)
    ax.set_xlim((0.2, 0.7))
    ax.set_ylim((-0.25, 0.25))
    ax.set_zlim((0.5, 1.0))
    ax = plot_pose_trajectory(P, Pe)
    ax.set_xlim((0.2, 0.7))
    ax.set_ylim((-0.25, 0.25))
    ax.set_zlim((0.5, 1.0))

    plot_joint_trajectories([Qa, Qe], labels=["Approximation", "Exact"])

    plt.show()
