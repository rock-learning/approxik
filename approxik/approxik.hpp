#include <string>
#include <iostream>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <limits>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <Eigen/Core>
// TODO make dependency for timing optional
#include <boost/date_time/posix_time/posix_time.hpp>
// TODO correct paths
#include "cppoptlib/problem.h"
#include "cppoptlib/solver/lbfgsbsolver.h"


void chainFromUrdf(const std::string& file, const std::string& root,
                   const std::string& tip, KDL::Chain& chain,
                   Eigen::VectorXd& qlo, Eigen::VectorXd& qhi, bool verbose=0)
{
  std::ifstream f(file.c_str());
  if(!f.good())
    throw std::ios_base::failure("Cannot read from file '" + file + "'.");

  boost::shared_ptr<urdf::ModelInterface> robot_model =
      urdf::parseURDFFile(file);

  KDL::Tree tree;
#ifdef ROCK // uses an old version of kdl_parser
  if(!kdl_parser::treeFromUrdfModel(*robot_model, tree))
#else
  if(!kdl_parser::treeFromUrdfModel(robot_model, tree))
#endif
    throw std::runtime_error("Could not extract kdl tree");
  if(verbose >= 1)
    std::cout << "[IK] KDL::Tree (" << tree.getNrOfJoints() << " joints, "
        << tree.getNrOfSegments() << " segments)" << std::endl;

  tree.getChain(root, tip, chain);
  if(verbose >= 1)
    std::cout << "[IK] KDL::Chain (" << chain.getNrOfJoints() << " joints, "
        << chain.getNrOfSegments() << " segments)" << std::endl;

  std::list<double> lo, hi;
  for(unsigned int i = 0; i < chain.getNrOfSegments(); i++)
  {
    KDL::Segment segment = chain.getSegment(i);
    KDL::Joint joint = segment.getJoint();
    if(joint.getType() != KDL::Joint::None)
    {
      std::string name = joint.getName();
      const double lower = robot_model->getJoint(name)->limits->lower;
      const double upper = robot_model->getJoint(name)->limits->upper;
      if(verbose >= 1)
      {
        std::cout << "  joint #" << lo.size() << ": '" << name << "'" << std::endl
            << "    limits: [" << lower << ", " << upper << "]" << std::endl;
      }
      lo.push_back(lower);
      hi.push_back(upper);
    }
  }

  const unsigned int numJoints = chain.getNrOfJoints();
  if(numJoints != lo.size() || numJoints != hi.size())
    throw std::runtime_error(
        "Number of joints in chain and limits is not equal!");

  qlo.resize(numJoints);
  qhi.resize(numJoints);

  std::list<double>::iterator loIt = lo.begin(), hiIt = hi.begin();
  for(unsigned int i = 0; i < numJoints; i++, loIt++, hiIt++)
  {
    qlo(i) = *loIt;
    qhi(i) = *hiIt;
  }
}


template <typename Derived>
KDL::JntArray jntEigen2KDL(const Eigen::MatrixBase<Derived>& qin)
{
  KDL::JntArray qout;
  qout.data = qin;
  return qout;
}


template <typename Derived>
void poseKDL2Eigen(const KDL::Frame& frame, Eigen::MatrixBase<Derived>& pose)
{
  for(unsigned int i = 0; i < 3; i++)
    pose(i) = frame.p(i);
  frame.M.GetQuaternion(pose(4), pose(5), pose(6), pose(3)); // x, y, z, w
}


template <typename Derived>
void poseEigen2KDL(const Eigen::MatrixBase<Derived>& pose, KDL::Frame& frame)
{
  for(unsigned int i = 0; i < 3; i++)
    frame.p(i) = pose(i);
  frame.M = KDL::Rotation::Quaternion(pose(4), pose(5), pose(6), pose(3));
}


class ForwardKinematics
{
  const int verbose;
  unsigned int numJoints;
  KDL::JntArray q;
  KDL::Frame p;
public:
  KDL::Chain chain;
  Eigen::VectorXd qlo, qhi;
  KDL::ChainFkSolverPos_recursive* fk;

  ForwardKinematics(const std::string& file, const std::string& root,
                    const std::string& tip, int verbose=0)
    : verbose(verbose)
  {
    chainFromUrdf(file, root, tip, chain, qlo, qhi, verbose);
    numJoints = chain.getNrOfJoints();
    fk = new KDL::ChainFkSolverPos_recursive(chain);
  }

  ~ForwardKinematics()
  {
    delete fk;
  }

  const unsigned int getNumJoints()
  {
    return numJoints;
  }

  template <typename Derived1, typename Derived2>
  bool JntToCart(const Eigen::MatrixBase<Derived1>& q,
                 Eigen::MatrixBase<Derived2>& p)
  {
    this->q.data = q;
    const bool result = fk->JntToCart(this->q, this->p) >= 0;
    poseKDL2Eigen(this->p, p);
    return result;
  }
};


class IKProblem : public cppoptlib::Problem<double>
{
  ForwardKinematics& fk;
  const Eigen::VectorXd* desiredPose;
  Eigen::VectorXd pose;
  const double positionWeight;
  const double rotationWeight;
public:
  IKProblem(ForwardKinematics& fk, double positionWeight=1.0,
            double rotationWeight=1.0)
    : fk(fk), desiredPose(0), pose(7),
      positionWeight(positionWeight), rotationWeight(rotationWeight)
  {
  }

  void setDesiredPose(const Eigen::VectorXd& newPose)
  {
    desiredPose = &newPose;
  }

  double value(const cppoptlib::Vector<double>& q)
  {
    const bool success = fk.JntToCart(q, pose);
    if(success)
    {
      const double sqPositionDist = (pose.head<3>() - desiredPose->head<3>()).squaredNorm();
      const Eigen::Quaterniond q1(pose.tail<4>().data());
      const Eigen::Quaterniond q2(desiredPose->tail<4>());
      const double angle = 2.0 * std::acos((q1 * q2.conjugate()).w());
      const double angularDist = std::min(angle, 2.0 * M_PI - angle);
      //const double angularDist = q1.angularDistance(q2);
      const double sqAngularDist = angularDist * angularDist;

      return positionWeight * sqPositionDist + rotationWeight * sqAngularDist;
    }
    else
    {
      return std::numeric_limits<double>::max();
    }
  }
};


class ApproximateInverseKinematics
{
protected:
  ForwardKinematics fk;
  IKProblem objective;
  unsigned int maxiter;
  const int verbose;
  // TODO make optional
  boost::posix_time::ptime start_time;
  boost::posix_time::ptime stop_time;
public:
  ApproximateInverseKinematics(
      const std::string& file, const std::string& root,
      const std::string& tip, double positionWeight=1.0,
      double rotationWeight=1.0, unsigned int maxiter=100000,
      int verbose=0)
    : fk(file, root, tip, verbose),
      objective(fk, positionWeight, rotationWeight), maxiter(maxiter),
      verbose(verbose)
  {
  }

  virtual ~ApproximateInverseKinematics()
  {
  }

  const unsigned int getNumJoints()
  {
    return fk.getNumJoints();
  }

  template <typename Derived1, typename Derived2>
  void JntToCart(const Eigen::MatrixBase<Derived1>& q,
                 Eigen::MatrixBase<Derived2>& pose)
  {
    fk.JntToCart(q, pose);
  }

  void JntToCart(double* q, double* pose)
  {
    Eigen::Map<Eigen::VectorXd> qvec(q, fk.getNumJoints());
    Eigen::Map<Eigen::VectorXd> pvec(pose, 7);
    fk.JntToCart(qvec, pvec);
  }

  template <typename Derived1, typename Derived2>
  bool CartToJnt(const Eigen::MatrixBase<Derived1>& pose,
                 Eigen::MatrixBase<Derived2>& currentQ)
  {
    // TODO ctor args
    cppoptlib::ISolver<double, 1>::Info stopControl;
    stopControl.rate = 0.0;
    stopControl.iterations = maxiter;
    stopControl.gradNorm = 0.0;
    stopControl.m = 10;
    // TODO check whether we can make the solver a member
    cppoptlib::LbfgsbSolver<double> solver(stopControl);

    // We cannot pass currentQ directly to solver.minimize() because the
    // interface is not generic enough
    Eigen::VectorXd q = currentQ;
    setProblemBounds(&q);  // must be done before any clipping is performed
    clipToLimits(q);

    Eigen::VectorXd p = pose;
    objective.setDesiredPose(p);

    if(verbose >= 2)
    {
      std::cout << "[IK] Limits:" << std::endl
          << "  " << objective.lowerBound().transpose() << std::endl
          << "  " << objective.upperBound().transpose() << std::endl;
    }

    start_time = boost::posix_time::microsec_clock::local_time(); // TODO optional
    solver.minimize(objective, q);

    // Workaround for a serious problem: for some poses we are completely stuck
    // in a local minimum
    if(q == currentQ)
    {
      if(verbose >= 2)
        std::cout << "[IK] Workaround is active" << std::endl;
      p(0) += 0.001;
      p(1) -= 0.001;
      p(2) += 0.001;
      objective.setDesiredPose(p);
      solver.minimize(objective, q);
      p(0) -= 0.001;
      p(1) += 0.001;
      p(2) -= 0.001;
      objective.setDesiredPose(p);
      solver.minimize(objective, q);
    }

    stop_time = boost::posix_time::microsec_clock::local_time(); // TODO optional
    // TODO check why limits are sometimes violated
    clipToLimits(q);

    currentQ = q;
    if(verbose >= 1)
    {
      std::cout << "[IK] Optimized joints:" << std::endl
          << "  " << currentQ.transpose() << std::endl
          << "  Objective: " << objective(currentQ) << std::endl;
      if(verbose >= 2)
      {
        Eigen::VectorXd actualPose(7);
        fk.JntToCart(currentQ, actualPose);
        std::cout << "[IK] Desired pose:" << std::endl
            << "  " << pose.transpose() << std::endl;
        std::cout << "[IK] Actual pose:" << std::endl
            << "  " << actualPose.transpose() << std::endl;
      }
    }

    return true;
  }

  void CartToJnt(double* pose, double* q)
  {
    Eigen::Map<Eigen::VectorXd> pvec(pose, 7);
    Eigen::Map<Eigen::VectorXd> qvec(q, fk.getNumJoints());
    CartToJnt(pvec, qvec);
  }

  // TODO optional
  double getLastTiming() // in seconds
  {
    return (stop_time - start_time).total_nanoseconds() / 1000000000.0;
  }

protected:
  virtual void setProblemBounds(Eigen::VectorXd* currentQ = 0)
  {
    objective.setBoxConstraint(fk.qlo, fk.qhi);
  }

  template <typename Derived>
  void clipToLimits(Eigen::MatrixBase<Derived>& q)
  {
    q = q.cwiseMax(objective.lowerBound()).cwiseMin(objective.upperBound());
  }
};


class ApproximateLocalInverseKinematics : public ApproximateInverseKinematics
{
  bool initialized;
  double maxJump;
public:
  ApproximateLocalInverseKinematics(
      const std::string& file, const std::string& root,
      const std::string& tip, double positionWeight=1.0,
      double rotationWeight=1.0, double maxJump=1.0,
      unsigned int maxiter=100000, int verbose=0)
    : ApproximateInverseKinematics(file, root, tip, positionWeight,
                                   rotationWeight, maxiter, verbose),
      initialized(false), maxJump(maxJump)
  {
  }

  void reset()
  {
    initialized = false;
  }
protected:
  virtual void setProblemBounds(Eigen::VectorXd* currentQ)
  {
    if(!initialized)
    {
      ApproximateInverseKinematics::setProblemBounds();
      initialized = true;
      return;
    }

    Eigen::VectorXd lo = (currentQ->array() - maxJump).matrix().cwiseMax(fk.qlo);
    Eigen::VectorXd hi = (currentQ->array() + maxJump).matrix().cwiseMin(fk.qhi);
    objective.setBoxConstraint(lo, hi);
  }
};


class ExactInverseKinematics
{
  ForwardKinematics fk;
  KDL::ChainIkSolverVel_pinv velIk;
  KDL::ChainIkSolverPos_NR_JL posIk;
  const int verbose;
  KDL::JntArray qout;
  KDL::Frame frame;
  // TODO make optional
  boost::posix_time::ptime start_time;
  boost::posix_time::ptime stop_time;
public:
  ExactInverseKinematics(
      const std::string& file, const std::string& root,
      const std::string& tip, double eps=1e-5, int maxiter=150,
      int verbose=0)
    : fk(file, root, tip, verbose), velIk(fk.chain, eps, maxiter),
      posIk(fk.chain, jntEigen2KDL(fk.qlo), jntEigen2KDL(fk.qhi), *fk.fk,
            velIk, maxiter, eps), verbose(verbose)
  {
    qout.resize(fk.getNumJoints());
  }

  const unsigned int getNumJoints()
  {
    return fk.getNumJoints();
  }

  template <typename Derived1, typename Derived2>
  void JntToCart(const Eigen::MatrixBase<Derived1>& q,
                 Eigen::MatrixBase<Derived2>& pose)
  {
    fk.JntToCart(q, pose);
  }

  void JntToCart(double* q, double* pose)
  {
    Eigen::Map<Eigen::VectorXd> qvec(q, fk.getNumJoints());
    Eigen::Map<Eigen::VectorXd> pvec(pose, 7);
    fk.JntToCart(qvec, pvec);
  }

  template <typename Derived1, typename Derived2>
  bool CartToJnt(const Eigen::MatrixBase<Derived1>& pose,
                 Eigen::MatrixBase<Derived2>& currentQ)
  {
    poseEigen2KDL(pose, frame);
    start_time = boost::posix_time::microsec_clock::local_time(); // TODO optional
    const int result = posIk.CartToJnt(jntEigen2KDL(currentQ), frame, qout);
    stop_time = boost::posix_time::microsec_clock::local_time(); // TODO optional
    currentQ = qout.data;

    if(verbose >= 1)
    {
      if(result == -3)
        std::cout << "[IK] FAILED: Maximum number of iterations reached!"
            << std::endl;
      std::cout << "[IK] Optimized joints:" << std::endl
          << "  " << currentQ.transpose() << std::endl;
      if(verbose >= 2)
      {
        Eigen::VectorXd actualPose(7);
        fk.JntToCart(currentQ, actualPose);
        std::cout << "[IK] Desired pose:" << std::endl
            << "  " << pose.transpose() << std::endl;
        std::cout << "[IK] Actual pose:" << std::endl
            << "  " << actualPose.transpose() << std::endl;
      }
    }
    return result >= 0;
  }

  bool CartToJnt(double* pose, double* q)
  {
    Eigen::Map<Eigen::VectorXd> pvec(pose, 7);
    Eigen::Map<Eigen::VectorXd> qvec(q, fk.getNumJoints());
    return CartToJnt(pvec, qvec);
  }

  // TODO optional
  double getLastTiming() // in seconds
  {
    return (stop_time - start_time).total_nanoseconds() / 1000000000.0;
  }
};
