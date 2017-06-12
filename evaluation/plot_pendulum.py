import matplotlib.pyplot as plt
import numpy as np
from pendulum import Pendulum
from bolero.behavior_search import BlackBoxSearch
from bolero.representation import DMPBehavior
from bolero.optimizer import CMAESOptimizer
from bolero.controller import Controller
import pendulum_config as cfg


n_episodes = 1000


def learn(setup_fun, variance):
    #ik, beh, mp_keys, mp_values = cfg.make_approx_cart_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)
    #ik, beh, mp_keys, mp_values = cfg.make_exact_cart_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)
    ik, beh, mp_keys, mp_values = cfg.make_joint_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt)

    env = Pendulum(
        x0=cfg.x0, g=cfg.g,
        execution_time=cfg.execution_time, dt=cfg.dt
    )

    opt = CMAESOptimizer(variance=variance, random_state=0)
    bs = BlackBoxSearch(beh, opt)
    controller = Controller(environment=env, behavior_search=bs,
                            n_episodes=n_episodes, verbose=2)
    rewards = controller.learn(mp_keys, mp_values)

    best = bs.get_best_behavior()
    best_params = best.get_params()
    np.save("best_params_pendulum_joint.npy", best_params)
    reward = controller.episode_with(best)

    ax = env.plot()
    plt.show()


learn(cfg.make_approx_cart_dmp, 500.0 ** 2)
