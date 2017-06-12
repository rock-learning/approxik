import numpy as np


def trajectory_ik(P, ik, return_reachable=False, q_init=None):
    Q = np.empty((P.shape[0], ik.get_n_joints()))
    reachable = np.empty(P.shape[0])
    lastq = np.zeros(ik.get_n_joints())
    q = np.zeros(ik.get_n_joints())
    if q_init is not None:
        q[:] = q_init[:]
    p = np.empty(7)
    timings = []
    for t in range(P.shape[0]):
        p[:] = P[t]
        reachable[t] = ik.cart_to_jnt(p, q)
        timings.append(ik.get_last_timing())
        if return_reachable and not reachable[t]:
            Q[t, :] = lastq
            q[:] = lastq
        else:
            Q[t, :] = q
            lastq[:] = q[:]
    if return_reachable:
        return Q, timings, reachable
    else:
        return Q, timings


def point_ik(P, ik):
    Q = np.empty((P.shape[0], ik.get_n_joints()))
    q = np.empty(ik.get_n_joints())
    p = np.empty(7)
    timings = []
    for t in range(P.shape[0]):
        q[:] = 0.0
        p[:] = P[t]
        ik.cart_to_jnt(p, q)
        timings.append(ik.get_last_timing())
        Q[t] = q[:]
    return Q, timings


def trajectory_fk(Q, ik):
    P = np.empty((Q.shape[0], 7))
    q = np.empty(Q.shape[1])
    p = np.empty(7)
    for t in range(Q.shape[0]):
        q[:] = Q[t]
        ik.jnt_to_cart(q, p)
        P[t, :] = p
    return P


def check_ik(ik):
    q = np.ones(ik.get_n_joints())
    p = np.empty(7)
    ik.jnt_to_cart(q, p)
    print("q => p")
    print("  q = %s =>\n  p = %s" % (np.round(q, 2), np.round(p, 2)))
    ik.cart_to_jnt(p, q)
    print("p => q")
    print("  p = %s =>\n  q = %s" % (np.round(p, 2), np.round(q, 2)))
    ik.jnt_to_cart(q, p)
    print("q => p")
    print("  q = %s =>\n  p = %s" % (np.round(q, 2), np.round(p, 2)))


def reaching_report(P, Q, ik, precision=1e-2, label=None):
    print("=== Reaching report (with precision %g) ===" % precision)
    if label is not None:
        print(label)
    P_reached = trajectory_fk(Q, ik)
    pos_errors = np.sqrt(np.sum((P[:, :3] - P_reached[:, :3]) ** 2, axis=1))
    n_pos_reached = np.count_nonzero(pos_errors <= precision)
    print("  %d positions reached" % n_pos_reached)
    # TODO


def timing_report(timings, label=None):
    timings = np.array(timings)
    print("=== Timing report ===")
    if label is not None:
        print(label)
    print("  Calls:   %d" % timings.shape[0])
    print("  Time   : %g s" % sum(timings))
    print("  Average: %g ms" % (1000.0 * np.mean(timings)))
    print("  Minimum: %g ms" % (1000.0 * np.min(timings)))
    print("  Maximum: %g ms" % (1000.0 * np.max(timings)))
    print("  Median:  %g ms" % (1000.0 * np.median(timings)))
