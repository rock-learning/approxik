import pytransform.rotations as pyrot
from visualization import *
from utils import parse_args
from convert import trajectory_ik, trajectory_fk, timing_report
from approxik import ApproxInvKin


if __name__ == "__main__":
    filename, base_link, ee_link = parse_args(base_link="link_0", ee_link="tcp")

    # Try setting the rotation penalty to 1.0
    aik = ApproxInvKin(filename, base_link, ee_link, 1.0, 0.001, verbose=0)

    n_steps = 1000
    Q = np.zeros((n_steps, aik.get_n_joints()))
    Q[:, 0] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_steps)
    P = trajectory_fk(Q, aik)

    P_offset = P.copy()
    # Offset in (unchangeable) x-direction
    P_offset[:, 0] += 0.1
    # Fix rotation
    P_offset[:, 3] = 1.0
    P_offset[:, 4:] = 0.0

    Qa, timings = trajectory_ik(P_offset, aik)
    Pa = trajectory_fk(Qa, aik)

    timing_report(timings)

    ax = plot_pose_trajectory(P_offset, Pa)

    plot_joint_trajectories([Q, Qa], labels=["Original", "Approximated Offset"])

    plt.show()
