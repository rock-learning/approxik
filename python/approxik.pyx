cimport approxik
from libcpp cimport bool
from libcpp.string cimport string
cimport numpy as np
import numpy as np


cdef class ApproxInvKin:
    cdef ApproximateInverseKinematics *thisptr
    def __cinit__(ApproxInvKin self, str file, str root, str tip,
                  double position_weight, double rotation_weight,
                  int maxiter=100000, int verbose=0):
        cdef bytes file_arg = file.encode("utf-8")
        cdef bytes root_arg = root.encode("utf-8")
        cdef bytes tip_arg = tip.encode("utf-8")
        self.thisptr = new ApproximateInverseKinematics(
            file_arg, root_arg, tip_arg, position_weight, rotation_weight,
            maxiter, verbose)

    def __del__(self):
        del self.thisptr

    def get_n_joints(self):
        return self.thisptr.getNumJoints()

    def jnt_to_cart(self, np.ndarray[double, ndim=1] q,
                    np.ndarray[double, ndim=1] pose):
        self.thisptr.JntToCart(&q[0], &pose[0])

    def cart_to_jnt(self, np.ndarray[double, ndim=1] pose,
                    np.ndarray[double, ndim=1] q):
        self.thisptr.CartToJnt(&pose[0], &q[0])

    def get_last_timing(self):
        return self.thisptr.getLastTiming()


cdef class ApproxLocalInvKin:
    cdef ApproximateLocalInverseKinematics *thisptr
    def __cinit__(ApproxLocalInvKin self, str file, str root, str tip,
                  double position_weight=1.0, double rotation_weight=1.0,
                  double max_jump=1.0, int maxiter=100000, int verbose=0):
        cdef bytes file_arg = file.encode("utf-8")
        cdef bytes root_arg = root.encode("utf-8")
        cdef bytes tip_arg = tip.encode("utf-8")
        self.thisptr = new ApproximateLocalInverseKinematics(
            file_arg, root_arg, tip_arg, position_weight, rotation_weight,
            max_jump, maxiter, verbose)

    def __del__(self):
        del self.thisptr

    def reset(ApproxLocalInvKin self):
        self.thisptr.reset()

    def get_n_joints(ApproxLocalInvKin self):
        return self.thisptr.getNumJoints()

    def jnt_to_cart(ApproxLocalInvKin self, np.ndarray[double, ndim=1] q,
                    np.ndarray[double, ndim=1] pose):
        self.thisptr.JntToCart(&q[0], &pose[0])

    def cart_to_jnt(ApproxLocalInvKin self, np.ndarray[double, ndim=1] pose,
                    np.ndarray[double, ndim=1] q):
        self.thisptr.CartToJnt(&pose[0], &q[0])

    def get_last_timing(ApproxLocalInvKin self):
        return self.thisptr.getLastTiming()


cdef class ExactInvKin:
    cdef ExactInverseKinematics *thisptr
    def __cinit__(ExactInvKin self, str file, str root, str tip,
                  double eps, int maxiter, int verbose=0):
        cdef bytes file_arg = file.encode("utf-8")
        cdef bytes root_arg = root.encode("utf-8")
        cdef bytes tip_arg = tip.encode("utf-8")
        self.thisptr = new ExactInverseKinematics(
            file_arg, root_arg, tip_arg, eps, maxiter, verbose)

    def __del__(self):
        del self.thisptr

    def get_n_joints(self):
        return self.thisptr.getNumJoints()

    def jnt_to_cart(self, np.ndarray[double, ndim=1] q,
                    np.ndarray[double, ndim=1] pose):
        self.thisptr.JntToCart(&q[0], &pose[0])

    def cart_to_jnt(self, np.ndarray[double, ndim=1] pose,
                    np.ndarray[double, ndim=1] q):
        return self.thisptr.CartToJnt(&pose[0], &q[0])

    def get_last_timing(self):
        return self.thisptr.getLastTiming()
