# Approximate Inverse Kinematics

Implementation of an approximate inverse kinematics for robots described by
URDF files in C++ with Python bindings. The approximate IK allows you to set
a weight for the position and the orientation to define how important it is
to reach them. Typically it is more important to reach the position exactly
than the orientation. Sometimes there are situations where it is not even
possible to reach the position at all while ignoring the orientation. In these
cases we want to find the IK solution that is closest to the desired goal.

We achieve these features by defining a cost function that has a weight for
the position and the orientation constraint respectively and lets the choice
of the weight for the user. We solve the IK problem by optimizing the cost
function with respect to the joint angles with L-BFGS-B, a local optimizer
that respects boundaries (i.e. joint limits). The required derivatives are
calculated numerically which is usually fast enough.

## Installation

### With rock

The package `control/approxik` is defined in
[this package set](https://git.hb.dfki.de/afabisch/package_set).

### Standalone

Install the [dependencies](dependencies.md) and install the C++ header:

    mkdir build
    cd build
    cmake ..
    make install

Now you can use

    #include <approxik/approxik.hpp>

in your project.

## Python Bindings

    ./get_dependencies.sh
    mkdir build
    cd build
    cmake ..
    cd ../python
    sudo python setup.py install

Note that dependencies will be installed in the subfolders `deps/install`.
Source the `env.sh` to set the correct environment variables.

## Folders

* approxik: C++ implementation (in one header file)
* cmake: additional CMake modules
* data: contains data files (e.g. URDFs, trajectories)
* deps: dependencies
* evaluation: scripts to generate plots for the paper
* examples: several scripts for visualization of the approximate IK approach
* paper: summary of the method, experiments, etc.
* prototype: first prototype of the approximate IK
* python: Python wrapper for C++ implementation
* src: C++ example program
* test: unit tests
