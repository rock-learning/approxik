# Dependencies

The library depends on the following packages:

* [L-BFGS-B](https://github.com/PatWie/CppNumericalSolvers)
* [URDFDOM](https://github.com/ros/urdfdom)
* [URDFDOM Headers](https://github.com/ros/urdfdom_headers)
* [ROS Console Bridge](https://github.com/ros/console_bridge)
* [KDL](https://github.com/orocos/orocos_kinematics_dynamics)
* [KDL Parser](https://github.com/AlexanderFabisch/robot_model)

You can install all dependencies with

    ./get_dependencies.sh

Dependencies that can be installed from system packages are

* boost-date-time, boost-system, boost-thread, boost-test
* tinyxml
* Python (headers)
* NumPy
* Cython
* Eigen 3

For Ubuntu 14.04 you can use these commands:

    sudo apt-get install python-dev python-numpy libeigen3-dev \
       libboost-date-time-dev libtinyxml-dev \
       libboost-system-dev libboost-thread-dev libboost-test-dev \
       cmake python-pip
    sudo pip install cython
