#include <approxik/approxik.hpp>


int main(int argc, char** argv)
{
  if (argc < 2){
    std::cerr << "Expect xml file to parse" << std::endl;
    return -1;
  }
  std::string file(argv[1]);

  Eigen::VectorXd p(7);

  ApproximateInverseKinematics approxik(
      file, "kuka_lbr_l_link_0", "kuka_lbr_l_link_7", 1.0, 0.001, 2);

  ExactInverseKinematics exactik(
      file, "kuka_lbr_l_link_0", "kuka_lbr_l_link_7", 1e-5, 10000, 2);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(approxik.getNumJoints());
  approxik.JntToCart(q, p);
  std::cout << "pose = " << p.transpose() << std::endl;

  q = Eigen::VectorXd::Random(approxik.getNumJoints());
  std::cout << "q = " << q.transpose() << std::endl;

  Eigen::VectorXd desiredPose(7);
  desiredPose << 0.0, 0.0, 1.2, 0.0, 0.0, 1.0, 0.0;
  std::cout << "==========" << std::endl;
  approxik.CartToJnt(desiredPose, q);
  std::cout << "==========" << std::endl;
  exactik.CartToJnt(desiredPose, q);

  // FK example: http://www.orocos.org/kdl/examples
  // KDL API doc: http://docs.ros.org/indigo/api/orocos_kdl/html/index.html
}
