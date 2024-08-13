# Created by Zehong Cao

import numpy as np
from examples.arguments import args
import matplotlib.pyplot as plt


def sarsa(env, start_state, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
    '''
    Sarsa algorithm for solving the Bellman Optimality Equation(BOE)
    :param env: instance of the environment
    :param start_state: the start state of the agent
    :param gamma: discounted factor
    :param alpha: learning rate
    :param epsilon: exploration rate
    :param episodes: length of episodes
    :param iterations: number of iterations
    :return:
    '''
    # initialize the state values
    Q = np.zeros((env.num_states, len(env.action_space)))
    V = np.zeros(env.num_states)
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]

    lengths = []
    total_rewards = []
    for k in range(iterations):
        # TODO: what if the start state is not unique?
        state = start_state  # s
        s = state[1] * env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]  # a

        length = 0
        total_reward = 0
        while state != (4, 4):  # TODO: what if we don't specify the target state?
            # policy evaluation: action value update
            next_state, reward = env.get_next_state_reward(state, action)  # r,s
            next_action = np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])  # a
            # action value update
            Q[s, a] = Q[s, a] - alpha * (Q[s, a] - (reward + gamma * Q[next_state, next_action]))
            # policy improvement
            idx = np.argmax(Q[s])
            # e-greedy exploration
            policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
            policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
            # state value update
            V[s] = np.sum(policy[s] * Q[s])
            # update the state and action
            s = next_state
            state = (next_state % env.env_size[0], next_state // env.env_size[0])
            action = env.action_space[next_action]
            a = next_action
            length += 1
            total_reward += reward
        lengths.append(length)
        total_rewards.append(total_reward)

    # TODO: plot the graph of the convergence of the length of episodes and that of the total rewards of episodes
    fig = plt.subplots(2, 1)
    plt.subplot(2, 1, 1)
    plt.plot(lengths)
    plt.xlabel('Iterations')
    plt.ylabel('Length of episodes')
    plt.subplot(2, 1, 2)
    plt.plot(total_rewards)
    plt.xlabel('Iterations')
    plt.ylabel('Total rewards of episodes')
    plt.show()

    return V, policy


def expected_sarsa(env, start_state, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
    Q = np.zeros((env.num_states, len(env.action_space)))
    V = np.zeros(env.num_states)
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]
    for k in range(iterations):
        # TODO: what if the start state is not unique?
        state = start_state  # s
        s = state[1] * env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]  # a
        while state != (4, 4):  # TODO: what if we don't specify the target state?
            # policy evaluation: action value update
            next_state, reward = env.get_next_state_reward(state, action)  # r,s
            # action value update
            Q[s, a] = Q[s, a] - alpha * (Q[s, a] - (reward + gamma * V[next_state]))
            # policy improvement
            idx = np.argmax(Q[s])
            # e-greedy exploration
            policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
            policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
            # state value update
            V[s] = np.sum(policy[s] * Q[s])
            # update the state and action
            s = next_state
            state = (next_state % env.env_size[0], next_state // env.env_size[0])
            a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
            action = env.action_space[a]
    return V, policy


def n_step_sarsa(env, start_state, steps, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000, iterations=100):
    Q = np.zeros((env.num_states, len(env.action_space)))
    V = np.zeros(env.num_states)
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1)[:, np.newaxis]
    for k in range(iterations):
        state = start_state  # s
        s = state[1] * env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]  # a
        while state != (4, 4):
            rewards = 0
            next_state, reward = env.get_next_state_reward(state, action)
            rewards = rewards * gamma + reward
            next_action = env.action_space[np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])]
            for t in range(steps - 2):
                next_state, reward = env.get_next_state_reward(
                    (next_state % env.env_size[0], next_state // env.env_size[0]), next_action)
                rewards = rewards * gamma + reward
                next_action = env.action_space[np.random.choice(np.arange(len(env.action_space)), p=policy[next_state])]
            next_s = next_state
            next_a = np.random.choice(np.arange(len(env.action_space)), p=policy[next_s])
            Q[s, a] = Q[s, a] - alpha * (Q[s, a] - (rewards + (gamma ** steps) * Q[next_s, next_a]))
            idx = np.argmax(Q[s])
            policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
            policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
            V[s] = np.sum(policy[s] * Q[s])
            s = next_s
            state = (next_s % env.env_size[0], next_s // env.env_size[0])
            a = next_a
            action = env.action_space[a]
    return V, policy
