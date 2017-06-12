import numpy as np
import matplotlib.pyplot as plt
from viapoint_environment import ViaPointEnvironment
from bolero.controller import Controller
import viapoint_config as cfg


modified_parameter = 50
param_range = 6000.0
n_steps = 101


configs = {
    "approxik" : cfg.make_approx_cart_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt),
    "exactik" : cfg.make_exact_cart_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt),
    "joint" : cfg.make_joint_dmp(cfg.x0, cfg.g, cfg.execution_time, cfg.dt),
}
linestyles = {
    "approxik" : "-",
    "exactik" : "--",
    "joint" : ":",
}
labels = {
    "approxik" : "Approximate IK",
    "exactik" : "Exact IK",
    "joint" : "Joint space",
}

plt.figure(figsize=(4, 3), dpi=300)
plt.subplots_adjust(left=0.2, right=0.93, bottom=0.16, top=0.96)
for name in sorted(configs.keys()):
    ik, beh, mp_keys, mp_values = configs[name]
    env = ViaPointEnvironment(
        ik, cfg.x0, cfg.via_points, cfg.execution_time, cfg.dt, cfg.qlo, cfg.qhi,
        penalty_vel=cfg.penalty_vel, penalty_acc=cfg.penalty_acc,
        penalty_via_point=cfg.penalty_via_point, log_to_stdout=True)
    env.init()

    beh.init(env.get_num_outputs(), env.get_num_inputs())
    best_params = np.load("best_params_viapoint_%s.npy" % name)
    beh.set_params(best_params)

    controller = Controller(environment=env, verbose=2)

    initial_param_value = best_params[modified_parameter]
    offset = np.linspace(-param_range / 2.0, param_range / 2.0, n_steps)

    returns = np.empty(n_steps)
    for i in range(n_steps):
        params = np.copy(best_params)
        params[modified_parameter] = initial_param_value + offset[i]
        beh.set_params(params)
        beh.reset()

        rewards = controller.episode_with(beh, mp_keys, mp_values)
        returns[i] = np.sum(rewards)

    plt.plot(offset, returns, label=labels[name], ls=linestyles[name],
             lw=2, c="k")

plt.xlabel("Weight offset")
plt.ylabel("Reward")
plt.xticks(np.linspace(-param_range / 2.0, param_range / 2.0, 5))
plt.yticks(np.linspace(-125, 0, 6))
plt.ylim((-125, 0))
plt.legend(loc="best")
plt.savefig("vary_weight_viapoint.pdf")
plt.show()
