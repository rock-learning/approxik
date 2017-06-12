import numpy as np
from pytransform.urdf import UrdfTransformManager


# A script that helps to determine the optimum ratio
# between step-sizes in Cartesian and joint space.
# We sample joint angles with variance 1 and measure
# the resulting standard deviation in Cartesian space.

tm = UrdfTransformManager()
tm.load_urdf(open("../data/kuka_lbr.urdf", "r"))

random_state = np.random.RandomState(42)
n_samples = 1000

joint_limits = np.array(
    [tm.get_joint_limits("kuka_lbr_l_joint_%d" % (j + 1))
     for j in range(7)]
)
#joint_angles = (random_state.rand(n_samples, 7) *
#                (joint_limits[:, 1] - joint_limits[:, 0]) +
#                joint_limits[:, 0])
joint_angles = random_state.randn(n_samples, 7)
positions = np.empty((n_samples, 3))
for n in range(n_samples):
    for j in range(7):
        tm.set_joint("kuka_lbr_l_joint_%d" % (j + 1), joint_angles[n, j])
    positions[n] = tm.get_transform("kuka_lbr_l_link_7", "kuka_lbr_l_link_0")[:3, 3]
print(np.std(joint_angles, axis=0))
print(np.mean(positions, axis=0))
print(np.std(positions, axis=0))
