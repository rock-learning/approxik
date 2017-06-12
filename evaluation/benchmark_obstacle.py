import os
import numpy as np
import matplotlib.pyplot as plt
from obstacle_environment import ObstacleEnvironment
from bolero.behavior_search import BlackBoxSearch
from bolero.optimizer import CMAESOptimizer
from bolero.controller import Controller
import obstacle_config as cfg
from joblib import Parallel, delayed
import pickle


n_runs = 30
n_episodes = 2000
setup_funs = [("approxik", -1, cfg.make_approx_cart_dmp),
              ("exactik", -1, cfg.make_exact_cart_dmp),
              ("joint", -1, cfg.make_joint_dmp)
              ]


def learn(name, setup_fun, run):
    ik, beh, mp_keys, mp_values = setup_fun(
            cfg.x0, cfg.g, cfg.execution_time, cfg.dt)
    env = ObstacleEnvironment(
        ik, cfg.x0, cfg.g, cfg.obstacles, cfg.execution_time, cfg.dt,
        cfg.qlo, cfg.qhi,
        penalty_goal_dist=cfg.penalty_goal_dist, penalty_vel=cfg.penalty_vel,
        penalty_acc=cfg.penalty_acc,
        obstacle_dist=cfg.obstacle_dist, penalty_obstacle=cfg.penalty_obstacle)

    opt = CMAESOptimizer(variance=cfg.variance[name], random_state=run)
    bs = BlackBoxSearch(beh, opt)
    controller = Controller(environment=env, behavior_search=bs,
                            n_episodes=n_episodes, verbose=0)
    rewards = controller.learn(mp_keys, mp_values)

    best = bs.get_best_behavior()
    reward = controller.episode_with(best)

    return name, rewards, reward.sum()


for name, n_jobs, setup_fun in setup_funs:
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch="all", verbose=10)
    results = parallel(delayed(learn)(name, setup_fun, run)
                       for run in range(n_runs))
    pickle.dump(results, open("results_benchmark_obstacle_%s.pickle"
                              % name, "w"))
