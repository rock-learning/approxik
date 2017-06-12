import sys
import pytransform.rotations as pyrot
from visualization import *
from convert import trajectory_ik, trajectory_fk, check_ik
from approxik import ApproxInvKin, ExactInvKin


if __name__ == "__main__":
    filename = sys.argv[1]

    print("= Approximate Inverse Kinematics =")
    aik = ApproxInvKin(filename, "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
                       1.0, 0.001, verbose=0)
    check_ik(aik)
    print("= Inverse Kinematics =")
    eik = ExactInvKin(filename, "kuka_lbr_l_link_0", "kuka_lbr_l_link_7",
                       1e-4, 1000, verbose=0)
    check_ik(eik)
