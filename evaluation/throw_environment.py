import numpy as np
from bolero.wrapper import CppBLLoader
from bolero.environment import ContextualEnvironment
from bolero.utils import check_random_state
from time import sleep


class ThrowEnvironment(ContextualEnvironment):
    """Extract the relevant feedbacks from the SpaceBot environment."""
    def __init__(self, start=None, random_state=None, verbose=0):
        env_name = "spacebot_throw_environment"
        self.bll = CppBLLoader()
        self.bll.load_library(env_name)
        self.env = self.bll.acquire_contextual_environment(env_name)
        self.start = start
        self.random_state = check_random_state(random_state)
        self.verbose = verbose

    def init(self):
        self.env.init()

    def reset(self):
        self.env.reset()
        self.go_to_start()

    def go_to_start(self):
        if self.start is None:
            return

        n_joints = self.get_num_inputs()
        inputs = np.copy(self.start)
        outputs = np.empty(n_joints)
        n_steps = 1000
        for t in range(n_steps):
            self.env.get_outputs(outputs)
            if np.linalg.norm(self.start - outputs) < 0.01:
                break

            self.env.set_inputs(inputs)
            self.env.step_action()

        if self.verbose >= 1:
            print("[COMPI] Start position has been initialized.")

        sleep(0.1)

    def get_num_inputs(self):
        return self.env.get_num_inputs()

    def get_num_outputs(self):
        return self.env.get_num_outputs()

    def get_outputs(self, values):
        self.env.get_outputs(values)

    def set_inputs(self, values):
        self.env.set_inputs(values)

    def step_action(self):
        self.env.step_action()

    def is_evaluation_done(self):
        return self.env.is_evaluation_done()

    def is_behavior_learning_done(self):
        return self.env.is_behavior_learning_done()

    def request_context(self, context=None):
        if context is None:
            context = self.random_state.uniform([1.0, -1.0], [2.5, 1.0])
        return self.env.request_context(context)

    def get_num_context_dims(self):
        return self.env.get_num_context_dims()

    def get_maximum_feedback(self, context):
        return 0.0

    def get_feedback(self):
        feedbacks = self.env.get_feedback()
        print("Ball hits the ground at %s" % np.round(feedbacks[1:3], 2))
        return feedbacks[0]
