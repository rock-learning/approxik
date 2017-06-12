from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "approxik.hpp":
    cdef cppclass ApproximateInverseKinematics:
        ApproximateInverseKinematics(
            string file, string root, string tip, double positionWeight,
            double rotationWeight, unsigned int maxiter, int verbose) except +
        unsigned int getNumJoints()
        void JntToCart(double* q, double* pose)
        void CartToJnt(double* pose, double* q)
        double getLastTiming()


cdef extern from "approxik.hpp":
    cdef cppclass ApproximateLocalInverseKinematics:
        ApproximateLocalInverseKinematics(
            string file, string root, string tip, double positionWeight,
            double rotationWeight, double maxJump, unsigned int maxiter,
            int verbose) except +
        void reset()
        unsigned int getNumJoints()
        void JntToCart(double* q, double* pose)
        void CartToJnt(double* pose, double* q)
        double getLastTiming()


cdef extern from "approxik.hpp":
    cdef cppclass ExactInverseKinematics:
        ExactInverseKinematics(
            string file, string root, string tip, double eps, int maxiter,
            int verbose) except +
        unsigned int getNumJoints()
        void JntToCart(double* q, double* pose)
        bool CartToJnt(double* pose, double* q)
        double getLastTiming()
