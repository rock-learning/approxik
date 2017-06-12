import numpy as np
import pytransform.rotations as pyrot
from utils import parse_args
from convert import trajectory_fk, point_ik, timing_report, reaching_report
from approxik import ApproxInvKin, ExactInvKin


if __name__ == "__main__":
    filename, base_link, ee_link = parse_args()

    # Try setting the rotation penalty to 1.0
    aik = ApproxInvKin(filename, base_link, ee_link, 1.0, 0.001, verbose=0)
    eik = ExactInvKin(filename, base_link, ee_link, 1e-4, 200, verbose=0)

    lo = np.array([-2.96706, -2.094395, -2.96706, -2.094395, -2.96706, -2.094395, -3.054326])
    hi = np.array([2.96706, 2.094395, 2.96706, 2.094395, 2.96706, 2.094395, 3.054326])
    rg = hi - lo

    n_steps = 1000
    random_state = np.random.RandomState(0)
    Q = random_state.rand(n_steps, aik.get_n_joints())
    Q = rg.reshape(1, -1) * Q + lo.reshape(1, -1)
    P = trajectory_fk(Q, aik)

    Qa, timings = point_ik(P, aik)
    reaching_report(P, Qa, aik, label="Approximate")
    timing_report(timings, "Approximate")
    Qe, timings = point_ik(P, eik)
    reaching_report(P, Qe, eik, label="Exact")
    timing_report(timings, "Exact")
