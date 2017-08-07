if [ ! -d "deps" ]; then
  mkdir deps
fi

cd deps
mkdir install

DIR=CppNumericalSolvers
echo ===========================================================================
echo $DIR
echo ===========================================================================
if [ ! -d "$DIR" ]; then
  git clone https://github.com/PatWie/CppNumericalSolvers.git
  cd $DIR
  # TODO there is a bug in future versions that prevents setting stopping
  # criteria
  git checkout d21022ff8a76b2c33714110374d2393fb4430f93
  cd ..
fi

DIR=console_bridge
echo ===========================================================================
echo $DIR
echo ===========================================================================
if [ ! -d "$DIR" ]; then
  git clone https://github.com/ros/console_bridge.git
  cd $DIR
  #git checkout 34f5c7d0fa4cb11abb7e794a86fa251432e8827a
  git checkout 0.2.5
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=../../install ..
  make -j2
  make install
  cd ../..
fi

DIR=urdfdom_headers
echo ===========================================================================
echo $DIR
echo ===========================================================================
if [ ! -d "$DIR" ]; then
  git clone https://github.com/rock-learning/urdfdom_headers.git
  cd $DIR
  git checkout fork
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=../../install ..
  make -j2
  make install
  cd ../..
fi

DIR=urdfdom
echo ===========================================================================
echo $DIR
echo ===========================================================================
if [ ! -d "$DIR" ]; then
  git clone https://github.com/rock-learning/urdfdom.git
  cd $DIR
  git checkout fork
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=../../install -DCMAKE_MODULE_PATH=../../install/share ..
  make -j2
  make install
  cd ../..
fi

DIR=orocos_kinematics_dynamics
echo ===========================================================================
echo $DIR
echo ===========================================================================
if [ ! -d "$DIR" ]; then
  git clone https://github.com/orocos/orocos_kinematics_dynamics.git
  cd $DIR/orocos_kdl
  git checkout 2aa76640f0a1c5ac57946c20e844372578b55743
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=../../../install ..
  make -j2
  make install
  cd ../../..
fi

DIR=robot_model
echo ===========================================================================
echo $DIR
echo ===========================================================================
if [ ! -d "$DIR" ]; then
  git clone https://github.com/AlexanderFabisch/robot_model.git
  cd $DIR/kdl_parser
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=../../../install ..
  make -j2
  make install
  cd ../../..
fi

cd ..
