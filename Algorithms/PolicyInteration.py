# Created by Zehong Cao
# truncated policy iteration

import numpy as np
from examples.arguments import args


def policy_iteration(env, gamma=0.9, theta=1e-10, epochs=10, continuing=False):
    '''
    policy iteration for solving the Bellman Optimality Equation(BOE)
    :param env: instance of the environment
    :param gamma: discounted factor
    :param theta: threshold level for convergence
    :param epochs: number of iterations for policy evaluation
    :return:
    '''
    # initialize the state values
    V = np.random.uniform(args.reward_forbidden, args.reward_target,
                          env.num_states)  # initial guess of the state values
    policy = np.random.uniform(0, 1, (env.num_states, len(env.action_space)))
    # normalize the probability to choose any action for each state in rows
    policy /= policy.sum(axis=1)[:, np.newaxis]  # what if the initial guess of the optimal policy is stochastic?
    iter_count = 0
    while True:
        delta = 0
        iter_count += 1
        temp_v = V.copy()
        # policy evaluation
        for i in range(epochs):
            # iteratively compute the state values
            for s in range(env.num_states):
                state = (s % env.env_size[0], s // env.env_size[0])
                q_values = []
                for a, action in enumerate(env.action_space):
                    next_state, reward = env.get_next_state_reward(state, action)
                    if continuing is False and state == env.target_state:  # absorbing state
                        v_next_state = 0
                    else:
                        v_next_state = V[next_state]
                    q_values.append(reward + gamma * v_next_state)
                # TODO: compute the state values under the current policy
                # Note: the initial policy is stochastic, which is different to that in value iteration
                V[s] = np.dot(policy[s], np.array(q_values))
        # policy improvement
        for s in range(env.num_states):
            state = (s % env.env_size[0], s // env.env_size[0])
            q_values = []
            for a, action in enumerate(env.action_space):
                next_state, reward = env.get_next_state_reward(state, action)
                # TODO: compute the action values
                if continuing is False and state == env.target_state:  # absorbing state
                    v_next_state = 0
                else:
                    v_next_state = V[next_state]
                q_values.append(reward + gamma * v_next_state)
            # TODO: finish the policy improvement step
            max_idx = np.argmax(q_values)
            policy[s, max_idx] = 1
            policy[s, np.arange(len(env.action_space)) != max_idx] = 0
            # special case: absorbing state
            # TODO: complement the policy for absorbing state described as the first way for episodic tasks in the book
            if state == env.target_state:
                policy[s, -1] = 1
                policy[s, :-1] = 0
        # check the terminal condition
        delta = max(np.abs(temp_v - V))
        print(f"Iteration {iter_count}, delta: {delta}")
        if delta < theta:
            break

    return V, policy
