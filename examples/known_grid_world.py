# Created by Zehong Cao
# model is known a priori when you test the model-based algorithms, as this is a deterministic environment itself

import sys

sys.path.append("..")
from src.grid_world import GridWorld
import gym
import random
import numpy as np
from Algorithms.ValueIteration import value_iteration
from Algorithms.PolicyInteration import policy_iteration
from Algorithms.Monte_Carlo import Basic_MC, ExploringStarts_MC, e_greedy_MC
from Algorithms.Sarsa import sarsa, expected_sarsa, n_step_sarsa
from Algorithms.Q_Learning import *
from Algorithms.PolicyGradient import *

if __name__ == "__main__":
    # TODO: Step 1, Choose your environment
    #  (the dpg() function is only compatible with the Pendulum environment since there is a continuous action space)
    #  (the a2c() function is only compatible with the CartPole environment since there is a state space with more observations)
    # 1.1. discrete action space
    env = GridWorld()
    # env = gym.make('CartPole-v1', render_mode='human')

    # 1.2. continuous action space
    # env = gym.make('Pendulum-v1', render_mode='human')

    # WARNING: Do not modify the following single line
    env.reset()

    # TODO: Step 2, Choose your algorithm to test all the algorithms mentioned in the book are implemented as follows

    # check the differences between value iteration and policy iteration
    optimal_value, optimal_policy = value_iteration(env)  # the ground truth state values
    # optimal_value, optimal_policy = policy_iteration(env)

    # check the differences among variants of basic Monte-Carlo Methods
    # optimal_value,optimal_policy = Basic_MC(env)
    # optimal_value, optimal_policy = ExploringStarts_MC(env)
    # optimal_value, optimal_policy = e_greedy_MC(env)

    # optimal_value, optimal_policy = sarsa(env, (0, 0))
    # optimal_value, optimal_policy = expected_sarsa(env, start_state=(0, 0))
    # optimal_value, optimal_policy = n_step_sarsa(env, start_state=(0, 0), steps=3)

    # optimal_value, optimal_policy = on_policy_ql(env, start_state=(0, 0))
    # optimal_value, optimal_policy = dqn(env, gt_state_values=optimal_value) # you could obtain the ground_truth state values from value_iteration() above
    # optimal_value, optimal_policy = reinforce(env)
    # optimal_value, optimal_policy = qac(env)
    # optimal_value, optimal_policy = a2c(env)
    # optimal_value, optimal_policy = off_policy_ac(env)
    # optimal_value, optimal_policy = dpg(env)

    # TODO: Step 3, Visualize the eventual policy by rendering the environment
    if isinstance(env, GridWorld):
        env.render()
        env.add_policy(optimal_policy)
        env.add_state_values(optimal_value)

        env.render(animation_interval=5)
