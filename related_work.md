# Related Work

## Approximate IK

Alex Henning (2014):
Approximate Inverse Kinematics Using a Database.
[PDF](https://www.wpi.edu/Pubs/E-project/Available/E-project-120814-181143/unrestricted/adhenning-mqp-final-20141208.pdf)

Morten Pol Engell-Nørregård, Sarah Maria Niebe, Morten Bo Bonding (2007):
Inverse Kinematics With Constraints.
[PDF](http://www.home.icandy.dk/kinematics/videos/rapport.pdf)

## Optimization for IK

Jianmin Zhao, Norman I. Badler (1994):
Inverse Kinematics Positioning Using Nonlinear Programming for Highly Articulated Figures.
[PDF](http://ai.stanford.edu/~latombe/cs99k/2000/badler.pdf)

Kwan W. Chin (1996):
Closed-form and Generalized Inverse Kinematics Solutions for Animating the Human Articulated Structure.
[PDF](http://www.uow.edu.au/~kwanwu/Honours-Thesis.pdf)

S. Kumar, N. Sukavanam, R. Balasubramanian (2010):
An optimization approach to solve the inverse kinematics of redundant manipulator
[PDF](http://www.math.ualberta.ca/ijiss/SS-Volume-6-2010/No-4-10/SS-10-04-07.pdf)

Maurice Fallon, Scott Kuindersma, Sisir Karumanchi, Matthew Antone, ... (2015):
An Architecture for Online Affordance-based Perception and Whole-body Planning
[PDF](http://groups.csail.mit.edu/robotics-center/public_papers/Fallon14.pdf)

Patrick Beeson, Barrett Ames (2015):
TRAC-IK: An Open-Source Library for Improved Solving of Generic Inverse Kinematics.
[PDF](https://personal.traclabs.com/~pbeeson/papers/Beeson-humanoids-15.pdf)

## IK Surveys

Rickard Nilsson (2009):
Inverse Kinematics
[PDF](http://epubl.ltu.se/1402-1617/2009/142/LTU-EX-09142-SE.pdf)

    The Pseudoinverse method reaches the target positions with the least number
    of iterations when they were in range. But when the target positions are
    out of range, the lack of a dampening factor causes the Jacobian matrix to
    get near singular and the resulting joint scalars became too large such that
    the model started to move irradically.
    The Pseudoinverse method which were using null space to try stabilize the
    joint scalars when target positions were out of range needed most time for
    each iteration. This method did not produce any results when the target
    positions were out of range which part can be blamed on the implementation
    and the other part can be blamed on the method it self.
    The transpose method produced results in all tests except when it used four
    Jacobians and the targets were out of range. The modified transpose method
    only outperformed the other methods when using multiple Jacobians with
    multiple end-effectors. When using a merged single Jacobian the number
    of iterations and time generally decreased.
    ...
    If I were to use any method to solve the inverse kinematic problem I would
    use the damped least square method. Looking at the results it can handle
    both target positions that are in range and out of range without generating
    any jerky movement to the model.