# Created by Zehong Cao
# two q-learning algorithms are introduced here:
# 1. tabular q-learning
# 2. deep q-learning

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def on_policy_ql(env, start_state, alpha=0.1, gamma=0.9, epsilon=0.1, iterations=100):
    q = np.zeros((env.num_states, len(env.action_space)))
    v = np.zeros(env.num_states)
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy = policy / policy.sum(axis=1)[:, np.newaxis]
    for k in range(iterations):
        state = start_state
        s = state[1] * env.env_size[0] + state[0]
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]
        while state != (4, 4):
            next_state, reward = env.get_next_state_reward(state, action)
            q[s, a] = q[s, a] - alpha * (
                    q[s, a] - (reward + gamma * np.max(q[next_state])))  # solving the Bellman optimality equation

            # update the policy
            idx = np.argmax(q[s])
            policy[s, idx] = 1 - epsilon * (len(env.action_space) - 1) / len(env.action_space)
            policy[s, np.arange(len(env.action_space)) != idx] = epsilon / len(env.action_space)
            v[s] = np.sum(policy[s] * q[s])

            s = next_state
            state = (s % env.env_size[0], s // env.env_size[0])
            a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
            action = env.action_space[a]

    # consistent policy
    # TODO: why update the policy here?
    # for s in range(env.num_states):
    #     idx = np.argmax(q[s])
    #     policy[s, idx] = 1
    #     policy[s, np.arange(len(env.action_space)) != idx] = 0
    return v, policy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc3 = nn.Linear(200, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def dqn(env, gt_state_values, c_steps=5, num_experience=1000, alpha=0.01, gamma=0.9, batch_size=100, epochs=1000):
    # generate experience by the behavior policy and store them in the reply buffer
    policy = np.ones((env.num_states, len(env.action_space)))
    policy = policy / len(env.action_space)
    reply_buffer = []
    s = np.random.choice(env.num_states)
    visit_history = []
    losses = []
    deltas = []
    for e in range(num_experience):
        a = np.random.choice(np.arange(len(env.action_space)), p=policy[s])
        action = env.action_space[a]
        next_state, reward = env.get_next_state_reward((s % env.env_size[0], s // env.env_size[0]), action)
        reply_buffer.append(((s % env.env_size[0], s // env.env_size[0]), a, reward,
                             (next_state % env.env_size[0], next_state // env.env_size[0])))
        visit_history.append((s, a))
        s = next_state
    # instantiate the main network and target network
    main_net, target_net = Net(), Net()
    target_net.load_state_dict(main_net.state_dict())  # synchronize the parameters of two networks
    optimizer = optim.Adam(main_net.parameters(), lr=alpha)
    criterion = nn.MSELoss()
    for k in range(epochs):
        # take a mini batch sample from the reply buffer
        batch = np.random.choice(len(reply_buffer), batch_size)  # uniform distribution
        states = torch.tensor([reply_buffer[i][0] for i in batch], dtype=torch.float).view(-1, 2)
        actions = torch.tensor([reply_buffer[i][1] for i in batch], dtype=torch.long).view(-1, 1)
        rewards = torch.tensor([reply_buffer[i][2] for i in batch], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor([reply_buffer[i][3] for i in batch], dtype=torch.float).view(-1, 2)
        # update the main network
        # TODO: finish the calculation of predicted action values
        main_net_input = states / env.env_size[0]  # (s[0],s[1],a)
        q_values = main_net(main_net_input).gather(1, actions.long())
        # TODO: finish the calculation of target action values
        target_net_input = next_states / env.env_size[0]
        with torch.no_grad():
            next_q_values = target_net(target_net_input).max(dim=1)[0].view(-1, 1)
        target = rewards + gamma * next_q_values
        # TODO: finish the update of one of the two deep neural networks, think which one to be updated
        loss = criterion(q_values, target)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # TODO: synchronize the parameters of two networks
        if (k + 1) % c_steps == 0:
            target_net.load_state_dict(main_net.state_dict())
        # calculate the value function
        delta = 0
        for s in range(env.num_states):
            with torch.no_grad():
                q_values = main_net(
                    torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                                 dtype=torch.float).view(-1, 2))
            delta = max(delta, torch.abs(gt_state_values[s] - torch.max(q_values)).item())
        deltas.append(delta)
    # generate the target policy and target state values
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    value_matrix = np.zeros(env.num_states)
    for s in range(env.num_states):
        with torch.no_grad():
            q_values = main_net(torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                                             dtype=torch.float).view(-1, 2))
        value_matrix[s] = torch.max(q_values).item()
        idx = torch.argmax(q_values).item()
        policy_matrix[s, idx] = 1

    plt.subplots(3, 1, figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.hist([i[0] for i in visit_history])
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.subplot(3, 1, 3)
    plt.plot(deltas)
    plt.title('Value Error')
    plt.show()

    return value_matrix, policy_matrix
