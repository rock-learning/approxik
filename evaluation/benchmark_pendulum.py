import numpy as np
from pendulum import Pendulum
from bolero.behavior_search import BlackBoxSearch
from bolero.representation import DMPBehavior
from bolero.optimizer import CMAESOptimizer
from bolero.controller import Controller
from joblib import Parallel, delayed
import pendulum_config as cfg
import pickle


n_runs = 30
n_episodes = 600
setup_funs = [("approxik", 15, cfg.make_approx_cart_dmp, 500.0 ** 2),
              ("exactik", 15, cfg.make_exact_cart_dmp, 500.0 ** 2),
              ("joint", 15, cfg.make_joint_dmp, 1250.0 ** 2)
              ]


def learn(name, run, setup_fun, variance):
    ik, beh, mp_keys, mp_values = setup_fun(
            cfg.x0, cfg.g, cfg.execution_time, cfg.dt)

    env = Pendulum(
        x0=cfg.x0, g=cfg.g,
        execution_time=cfg.execution_time, dt=cfg.dt
    )

    opt = CMAESOptimizer(variance=variance, random_state=run)
    bs = BlackBoxSearch(beh, opt)
    controller = Controller(environment=env, behavior_search=bs,
                            n_episodes=n_episodes, verbose=2)
    rewards = controller.learn(mp_keys, mp_values)

    best = bs.get_best_behavior()
    reward = controller.episode_with(best)

    return name, rewards, reward.sum()


for name, n_jobs, setup_fun, variance in setup_funs:
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch="all", verbose=10)
    results = parallel(delayed(learn)(name, run, setup_fun, variance)
                       for run in range(n_runs))
    pickle.dump(results, open("results_benchmark_pendulum_%s.pickle"
                              % name, "w"))
