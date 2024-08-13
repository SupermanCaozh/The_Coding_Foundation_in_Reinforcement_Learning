# Created by Zehong Cao
# model is known a priori when you test the model-based algorithms, as this is a deterministic environment itself

import sys

sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np
from Algorithms.ValueIteration import value_iteration
from Algorithms.PolicyInteration import policy_iteration
from Algorithms.Monte_Carlo import Basic_MC, ExploringStarts_MC, e_greedy_MC
from Algorithms.Sarsa import sarsa, expected_sarsa, n_step_sarsa
from Algorithms.Q_Learning import *
from Algorithms.PolicyGradient import reinforce

if __name__ == "__main__":
    env = GridWorld()
    env.reset()
    # env.render()

    # all the algorithms mentioned in the book are implemented as follows

    # check the differences between value iteration and policy iteration
    optimal_value, _ = value_iteration(env)  # the ground truth state values
    # optimal_value, optimal_policy = policy_iteration(env, continuing=False)

    # check the differences among variants of basic Monte-Carlo Methods
    # optimal_value,optimal_policy = Basic_MC(env)
    # optimal_value, optimal_policy = ExploringStarts_MC(env)
    # optimal_value, optimal_policy = e_greedy_MC(env)

    # optimal_value, optimal_policy = sarsa(env, (0, 0))
    # optimal_value, optimal_policy = expected_sarsa(env, start_state=(0, 0))
    # optimal_value, optimal_policy = n_step_sarsa(env, start_state=(0, 0), steps=3)

    # optimal_value, optimal_policy = on_policy_ql(env, start_state=(0, 0))
    # optimal_value, optimal_policy = dqn(env, gt_state_values=optimal_value)
    optimal_value, optimal_policy = reinforce(env, optimal_value)

    env.render()
    env.add_policy(optimal_policy)
    env.add_state_values(optimal_value)

    env.render(animation_interval=5)
