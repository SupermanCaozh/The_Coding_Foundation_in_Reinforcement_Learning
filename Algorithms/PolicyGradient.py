# Created by Zehong Cao

import numpy as np
import torch
import matplotlib.pyplot as plt


def e_greedy_policy(epsilon, action_space, p):
    if np.random.rand() < epsilon:
        return np.random.choice(len(action_space))
    else:
        return np.random.choice(len(action_space), p=p)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.001)


class Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


def reinforce(env, gt_state_values, start_state=(0, 0), end_state=(4, 4), alpha=0.01, gamma=0.9, iterations=1000):
    # initialize the policy net
    policy_net = Net(2, len(env.action_space))
    # policy_net.apply(init_weights)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=alpha)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))

    losses = []
    state_values = np.zeros(env.num_states)
    action_values = np.zeros((env.num_states, len(env.action_space)))
    deltas = []
    episode_lengths = []
    for k in range(iterations):
        # predict the current policy
        for s in range(env.num_states):
            state = torch.tensor(np.array((s % env.env_size[0], s // env.env_size[0])) / env.env_size[0],
                                 dtype=torch.float32).view(-1, 2)
            policy = policy_net(state).squeeze()
            if torch.isnan(policy).any():
                print(policy)
                raise ValueError("Nan detected.")
            policy_matrix[s] = policy.detach().numpy()
        # TODO: generate an sufficiently long episode and update the policy net
        # Question: what is the distribution of the states sampled in the episode?
        states = []
        actions = []
        rewards = []
        s = start_state[1] * env.env_size[0] + start_state[0]
        # Monte-Carlo
        while s != end_state[1] * env.env_size[0] + end_state[0]:
            # for i in range(episode_length):
            state = (s % env.env_size[0], s // env.env_size[0])
            # a = e_greedy_policy(0.1, env.action_space, policy_matrix[s] / np.sum(policy_matrix[s]))
            a = np.random.choice(len(env.action_space), p=policy_matrix[s] / np.sum(policy_matrix[s]))
            action = env.action_space[a]
            next_state, reward = env.get_next_state_reward(state, action)

            states.append(state)
            actions.append(a)  # actions index
            rewards.append(reward)

            s = next_state
        episode_lengths.append(len(states))
        # train the policy net by stochastic gradient ascend
        g = 0
        mean_criteria = 0
        optimizer.zero_grad()
        for t in range(len(states) - 1, -1, -1):
            g = gamma * g + rewards[t]  # q value estimation
            state = torch.tensor(np.array(states[t]) / env.env_size[0], dtype=torch.float32).view(-1, 2)
            action = actions[t]
            action_values[states[t][1] * env.env_size[0] + states[t][0], action] = g
            policy = policy_net(state).squeeze()
            # TODO: calculate the 'loss function'
            loss = -torch.log(policy.gather(0, torch.tensor(action, dtype=torch.long))) * g
            mean_criteria += loss.item()
            loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(mean_criteria / len(states))
        print(f"Interation {k + 1} has finished. Episode length is {len(states)}")
        delta = 0
        for s in range(env.num_states):
            state_values[s] = np.sum(policy_matrix[s] * action_values[s])  # calculate the state value estimation
            delta = max(delta, np.abs(state_values[s] - gt_state_values[s]))
        deltas.append(delta)
    # generate the optimal policy
    for s in range(env.num_states):
        state = torch.tensor(np.array((s % env.env_size[0], s // env.env_size[0])) / env.env_size[0],
                             dtype=torch.float32).view(-1, 2)
        policy = policy_net(state).squeeze().detach().numpy()

        policy_matrix[s, np.argmax(policy)] = 1
        policy_matrix[s, np.arange(len(env.action_space)) != np.argmax(policy)] = 0

    plt.subplots(2, 1, figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title(
        "Criteria")  # Question: what will this metric finally be? what does it mean? (connect it with the meaning of on-policy)
    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths)
    plt.title("Lengths of Sampling Episodes")
    plt.show()

    return state_values, policy_matrix
