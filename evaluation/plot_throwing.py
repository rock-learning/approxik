import os
import numpy as np
import matplotlib.pyplot as plt
from throw_environment import ThrowEnvironment
from bolero.behavior_search import BlackBoxSearch
from bolero.optimizer import CMAESOptimizer
from bolero.controller import Controller
import throw_config as cfg


ik, beh, mp_keys, mp_values, initial_params, var = cfg.make_approx_cart_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)
#ik, beh, mp_keys, mp_values, initial_params, var = cfg.make_exact_cart_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)
#ik, beh, mp_keys, mp_values, initial_params, var = cfg.make_joint_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)


env = ThrowEnvironment(start=cfg.x0j, random_state=0, verbose=1)
opt = CMAESOptimizer(initial_params=initial_params, variance=var,
                     random_state=0)
bs = BlackBoxSearch(beh, opt)
controller = Controller(environment=env, behavior_search=bs, n_episodes=800,
                        verbose=2)
rewards = controller.learn(mp_keys, mp_values)

best = bs.get_best_behavior()
reward = controller.episode_with(best)
print(reward.sum())

plt.plot(rewards)
plt.show()
