import os
import numpy as np
import matplotlib.pyplot as plt
from obstacle_environment import ObstacleEnvironment
from bolero.behavior_search import BlackBoxSearch
from bolero.optimizer import CMAESOptimizer
from bolero.controller import Controller
import obstacle_config as cfg


ik, beh, mp_keys, mp_values = cfg.make_approx_cart_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)
#ik, beh, mp_keys, mp_values = cfg.make_exact_cart_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)
#ik, beh, mp_keys, mp_values = cfg.make_joint_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)


env = ObstacleEnvironment(
    ik, cfg.x0, cfg.g, cfg.obstacles, cfg.execution_time, cfg.dt,
    cfg.qlo, cfg.qhi,
    penalty_goal_dist=cfg.penalty_goal_dist, penalty_vel=cfg.penalty_vel,
    penalty_acc=cfg.penalty_acc,
    obstacle_dist=cfg.obstacle_dist, penalty_obstacle=cfg.penalty_obstacle,
    log_to_stdout=True)

if os.path.exists("initial_params.txt"):
    initial_params = np.loadtxt("initial_params.txt")
else:
    initial_params = None
opt = CMAESOptimizer(initial_params=initial_params,
                     variance=cfg.variance["approxik"],
                     random_state=0)
bs = BlackBoxSearch(beh, opt)
controller = Controller(environment=env, behavior_search=bs, n_episodes=800,
                        verbose=2)
rewards = controller.learn(mp_keys, mp_values)

best = bs.get_best_behavior()
best_params = best.get_params()
np.save("best_params_obstacle_approxik.npy", best_params)
reward = controller.episode_with(best)
print(reward.sum())

plt.plot(rewards)

ax = env.plot()
ax.view_init(azim=-90, elev=0)
ax.set_xticks((-0.3, 0.0, 0.3))
ax.set_yticks(())
ax.set_zticks((0.5, 0.8, 1.1))
plt.savefig("obstacles1.pdf")

ax = env.plot()
ax.view_init(azim=175, elev=30)
ax.set_xticks((-0.3, 0.0, 0.3))
ax.set_yticks((-0.2, -0.5, -0.8))
ax.set_zticks((0.5, 0.8, 1.1))
plt.savefig("obstacles2.pdf")

plt.show()
