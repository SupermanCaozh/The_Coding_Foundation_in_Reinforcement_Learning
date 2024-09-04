# Created by Zehong Cao

import numpy as np
import torch
import matplotlib.pyplot as plt
import random


class PolicyNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 100)
        self.fc2 = torch.nn.Linear(100, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # TODO：why softmax here?
        x = torch.nn.functional.softmax(x, dim=-1)
        # TODO：why tanh here?(Hint:connect it with the particular env tested here)
        # x = torch.tanh(x) * 2.0
        return x


class ValueNet(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 100)
        self.fc2 = torch.nn.Linear(100, 100)
        self.fc3 = torch.nn.Linear(100, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def reinforce(env, start_state=(0, 0), end_state=(4, 4), alpha=0.001, gamma=0.8, iterations=1000):
    # initialize the policy net
    policy_net = PolicyNet(2, len(env.action_space))
    # policy_net.apply(init_weights)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=alpha)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5)
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))

    losses = []
    state_values = np.zeros(env.num_states)
    action_values = np.zeros((env.num_states, len(env.action_space)))
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
            else:
                policy_matrix[s] = policy.detach().numpy()
        # TODO: generate a sufficiently long episode and update the policy net
        # Question: what is the distribution of the states sampled in the episode?
        states = []
        actions = []
        rewards = []
        s = start_state[1] * env.env_size[0] + start_state[0]
        # Monte-Carlo
        while s != end_state[1] * env.env_size[0] + end_state[0]:
            # for i in range(episode_length):
            state = (s % env.env_size[0], s // env.env_size[0])
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
        criterion = 0
        optimizer.zero_grad()
        for t in range(len(states) - 1, -1, -1):
            g = gamma * g + rewards[t]  # q value estimation

            state = torch.tensor(np.array(states[t]) / env.env_size[0], dtype=torch.float32).view(-1, 2)
            action = actions[t]
            action_values[states[t][1] * env.env_size[0] + states[t][0], action] = g
            policy = policy_net(state).squeeze()
            # TODO: calculate the 'loss function'
            loss = -torch.log(policy.gather(0, torch.tensor(action, dtype=torch.long))) * g
            criterion += loss.item()

            loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(criterion / len(states))
        print(f"Interation {k + 1} has finished. Episode length is {len(states)}")
        for s in range(env.num_states):
            state_values[s] = np.sum(policy_matrix[s] * action_values[s])  # calculate the state value estimation
    # generate the optimal deterministic policy
    for s in range(env.num_states):
        target_index = np.argmax(policy_matrix[s])
        policy_matrix[s, target_index] = 1
        policy_matrix[s, np.arange(len(env.action_space)) != target_index] = 0

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


def qac(env, start_state=(0, 0), end_state=(4, 4), alpha=0.01, gamma=0.9, iterations=50):
    # instantiate the policy network and the q network
    policy_net = PolicyNet(2, len(env.action_space))  # (s0,s1)|--> (a0,a1,a2,a3,a4)
    q_net = ValueNet(3, 1)  # take the (s0,s1,a_idx)|--> q(s,a)
    optimizer1 = torch.optim.Adam(policy_net.parameters(), lr=alpha * 0.1)  # for policy net
    optimizer2 = torch.optim.Adam(q_net.parameters(), lr=alpha)  # for q_value net

    criteria = torch.nn.MSELoss()

    q_losses = []
    policy_losses = []
    episode_lengths = []
    for k in range(iterations):
        # generate an episode
        states = []
        q_loss = 0
        p_loss = 0

        s = start_state[1] * env.env_size[0] + start_state[0]

        while s != end_state[1] * env.env_size[0] + end_state[0]:
            policy = policy_net(
                torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                             dtype=torch.float).view(-1, 2)).squeeze()
            a = np.random.choice(len(env.action_space), p=policy.detach().numpy() / np.sum(policy.detach().numpy()))
            action = env.action_space[a]
            q = q_net(torch.concatenate((torch.tensor(np.array(start_state) / env.env_size[0], dtype=torch.float).view(
                -1, 2), torch.tensor(a, dtype=torch.float).view(-1, 1)), dim=1))

            next_state, reward = env.get_next_state_reward((s % env.env_size[0], s // env.env_size[0]), action)
            policy_next = policy_net(
                torch.tensor(np.array([next_state % env.env_size[0], next_state // env.env_size[0]]) / env.env_size[0],
                             dtype=torch.float).view(-1, 2)).squeeze().detach().numpy()
            next_a = np.random.choice(len(env.action_space),
                                      p=policy_next / np.sum(policy_next))
            with torch.no_grad():
                q_next = q_net(torch.concatenate((
                    torch.tensor(
                        np.array([next_state % env.env_size[0], next_state // env.env_size[0]]) / env.env_size[0],
                        dtype=torch.float).view(
                        -1, 2), torch.tensor(next_a, dtype=torch.float).view(-1, 1)), dim=1))

            loss_p = -torch.log(policy.gather(0, torch.tensor(a, dtype=torch.long))) * (q.detach())
            optimizer1.zero_grad()
            loss_p.backward()
            optimizer1.step()

            target = reward + gamma * q_next
            loss_q = criteria(q, target)
            optimizer2.zero_grad()
            loss_q.backward()
            optimizer2.step()

            q_loss += loss_q.item()
            p_loss += loss_p.item()

            states.append(s)

            s = next_state

        episode_lengths.append(len(states))

        # optimizer1.step()
        policy_losses.append(p_loss / len(states))
        q_losses.append(q_loss / len(states))

        print(f"Interation {k + 1} has finished. Episode length is {len(states)}")
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))

    state_values = np.zeros(env.num_states)
    for s in range(env.num_states):
        policy_matrix[s] = policy_net(
            torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                         dtype=torch.float).view(-1, 2)).squeeze().detach().numpy()
        action_values = np.zeros(len(env.action_space))
        for a in range(len(env.action_space)):
            action_values[a] = q_net(torch.concatenate((
                torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
                             dtype=torch.float).view(-1, 2),
                torch.tensor(a, dtype=torch.float).view(-1, 1)), dim=1)).item()
        state_values[s] = np.sum(policy_matrix[s] * action_values)
    # calculate the optimal deterministic policy
    pp = np.zeros((env.num_states, len(env.action_space)))
    for s in range(env.num_states):
        pp[s, np.argmax(policy_matrix[s])] = 1

    plt.subplots(3, 1, figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.plot(q_losses)
    plt.title('Loss of Q network')
    plt.subplot(3, 1, 2)
    plt.plot(policy_losses)
    plt.title('Loss of Policy network')
    plt.subplot(3, 1, 3)
    plt.plot(episode_lengths)
    plt.title('Length of Sampling Episodes')
    plt.show()

    return state_values, pp


def a2c(env, alpha=0.01, gamma=0.9, iterations=500):
    env.action_space = [0, 1]

    policy_net = PolicyNet(4, len(env.action_space))  # (s0,s1,s2,s3)|--> (a0,a1,a2,a3,a4)
    v_net = ValueNet(4, 1)  # take the (s0,s1,s2,s3)|--> v(s)
    optimizer_p = torch.optim.Adam(policy_net.parameters(), lr=alpha * 0.1)  # for policy net
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=alpha)  # for q_value net
    v_losses = []
    p_losses = []
    episode_lengths = []

    return_list = []
    for k in range(iterations):
        # generate an episode
        states = []
        v_loss = torch.tensor(0, dtype=torch.float)
        p_loss = torch.tensor(0, dtype=torch.float)

        # s = start_state[1] * env.env_size[0] + start_state[0] # gw env
        s = env.reset()[0]  # cp env, return the observation(state)
        done = False
        episode_return = 0

        # optimizer_v.zero_grad()
        # optimizer_p.zero_grad()

        # while s != end_state[1] * env.env_size[0] + end_state[0]:
        while not done:  # till the episode ends
            # policy = policy_net(torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
            # dtype=torch.float).view(-1, 2)).squeeze()
            # a = np.random.choice(len(env.action_space), p=policy.detach().numpy() / np.sum(policy.detach().numpy()))

            policy = policy_net(torch.tensor(s, dtype=torch.float).view(-1, 4)).squeeze()  # actor
            action = torch.distributions.Categorical(policy).sample().item()

            v = v_net(torch.tensor(s, dtype=torch.float).view(-1, 4)).squeeze()  # critic

            # action = env.action_space[a]
            # next_state, reward = env.get_next_state_reward((s % env.env_size[0], s // env.env_size[0]), action)

            next_state, reward, done, _, _ = env.step(action)
            episode_return += reward

            # v_next = v_net(torch.tensor(
            #     np.array([next_state % env.env_size[0], next_state // env.env_size[0]]) / env.env_size[0],
            #     dtype=torch.float).view(-1, 2)).squeeze()
            v_next = v_net(torch.tensor(next_state, dtype=torch.float).view(-1, 4)).squeeze()

            target = reward + gamma * v_next * (1 - done)
            loss = torch.square(target.detach() - v)

            optimizer_v.zero_grad()
            loss.backward()
            optimizer_v.step()

            v_loss += loss

            loss = -torch.log(policy.gather(0, torch.tensor(action, dtype=torch.long))) * (target - v).detach()

            optimizer_p.zero_grad()
            loss.backward()
            optimizer_p.step()

            p_loss += loss

            states.append(s)
            s = next_state

        v_loss /= torch.tensor(len(states), dtype=torch.float)
        p_loss /= torch.tensor(len(states), dtype=torch.float)

        # v_loss.backward()
        # p_loss.backward()
        #
        # optimizer_v.step()
        # optimizer_p.step()

        episode_lengths.append(len(states))
        v_losses.append(v_loss.item())
        p_losses.append(p_loss.item())

        return_list.append(episode_return)
        print(
            f"Interation {k + 1} has finished. Episode length is {len(states)}. Critic Loss is {v_losses[-1]}. Actor Loss is {p_losses[-1]}")

    # policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    # v_matrix = np.zeros(env.num_states)
    # for s in range(env.num_states):
    # policy_matrix[s] = policy_net(
    #     torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
    #                  dtype=torch.float).view(-1, 2)).squeeze().detach().numpy()
    # v_matrix[s] = v_net(torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0],
    #                                  dtype=torch.float).view(-1, 2)).item()

    plt.subplots(2, 2, figsize=(40, 40))
    plt.subplot(2, 1, 1)
    plt.plot(return_list)
    plt.title('Total rewards in a single episode')
    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths)
    plt.title('Lengths of a single episode')
    plt.subplot(2, 2, 3)
    plt.plot(v_losses)
    plt.title('Training Loss of critic')
    plt.subplot(2, 2, 4)
    plt.plot(p_losses)
    plt.title('Training Loss of actor')
    plt.tight_layout()
    plt.show()

    return v_net, policy_net


def off_policy_ac(env, alpha=0.01, gamma=0.9, iterations=50):
    policy_net = PolicyNet(2, len(env.action_space))  # (s0,s1)|--> (a0,a1,a2,a3,a4)
    v_net = ValueNet(2, 1)  # take the (s0,s1)|--> v(s)
    optimizer_p = torch.optim.Adam(policy_net.parameters(), lr=alpha * 0.1)  # for policy net
    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=alpha)  # for q_value net

    behavior_policy = np.ones((env.num_states, len(env.action_space))) / len(env.action_space)

    criteria = torch.nn.MSELoss()

    for k in range(iterations):
        # generate an episode
        states = []
        v_loss = torch.tensor(0, dtype=torch.float)
        p_loss = torch.tensor(0, dtype=torch.float)

        s = env.reset()[0]
        done = False
        while not done:  # till the episode ends
            state_idx = s[1] * env.env_size[0] + s[0]
            states.append(state_idx)

            action_idx = torch.distributions.Categorical(torch.tensor(behavior_policy[state_idx])).sample().item()
            action = env.action_space[action_idx]

            # current state-related items
            v = v_net(torch.tensor(np.array(s) / env.env_size[0], dtype=torch.float).view(-1, 2)).squeeze()  # critic
            policy = policy_net(torch.tensor(np.array(s) / env.env_size[0], dtype=torch.float).view(-1, 2)).squeeze()

            next_state, reward, done, _ = env.step(action)
            s = next_state

            with torch.no_grad():
                v_next = v_net(
                    torch.tensor(np.array(next_state) / env.env_size[0], dtype=torch.float).view(-1, 2)).squeeze()

            target = reward + gamma * v_next * (1 - done)

            loss = criteria(v, target.detach()) * (
                    policy.gather(0, torch.tensor(action_idx, dtype=torch.long)) / behavior_policy[
                state_idx, action_idx]).detach()

            v_loss += loss.item()

            optimizer_v.zero_grad()
            loss.backward()
            optimizer_v.step()

            loss = -torch.log(policy.gather(0, torch.tensor(action_idx, dtype=torch.long))) * (target - v).detach() * (
                    policy.gather(0, torch.tensor(action_idx, dtype=torch.long)).detach() / behavior_policy[
                state_idx, action_idx])

            optimizer_p.zero_grad()
            loss.backward()
            optimizer_p.step()

            p_loss += loss
        print(
            f"Interation {k + 1} has finished. Advantaged quantity is {v_loss.item() / len(states)}. Actor Loss is {p_loss.item() / len(states)}")

    V = np.zeros(env.num_states)
    P = np.zeros((env.num_states, len(env.action_space)))  # the consistent deterministic optimal policy
    for s in range(env.num_states):
        state = torch.tensor(np.array([s % env.env_size[0], s // env.env_size[0]]) / env.env_size[0])
        with torch.no_grad():
            V[s] = v_net(torch.tensor(state, dtype=torch.float).view(-1, 2)).squeeze()
            p = policy_net(torch.tensor(state, dtype=torch.float).view(-1, 2)).squeeze()
            P[s, torch.argmax(p)] = 1

    return V, P


def dpg(env, alpha=0.001, gamma=0.99, iterations=1000, batch_size=128):
    actor = PolicyNet(3, 1)  # (s0,s1,s2)|-->action index
    critic = ValueNet(4, 1)  # (s0,s1,s2,a)|-->action value
    optimizer_a = torch.optim.Adam(actor.parameters(), lr=alpha)
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=alpha * 10)

    td_errors = np.empty(iterations)  # since the dpg solves the BOE, the td errors should converge to zero
    td_errors.fill(100)

    sample_threshold = 1000

    replay_buffer = []

    return_list = []

    for k in range(iterations):
        s = env.reset()[0]
        done = False

        episode_reward = 0
        episode_len = 0
        while not done:
            with (torch.no_grad()):
                action = actor(
                    torch.tensor(np.array(s), dtype=torch.float).view(-1, 3)).squeeze().detach().item()
                action += np.random.normal(0, 0.1)  # noise added in the behavior policy

            next_state, reward, done, _, _ = env.step([action])

            # supplement the buffer
            experience = (s, action, reward, next_state, done)
            replay_buffer.append(experience)

            s = next_state
            episode_len += 1
            episode_reward = episode_reward * gamma + reward

            if not done and episode_len >= 200:
                done = True

        return_list.append(episode_reward)

        # replay buffer trick to stabilize the training
        if len(replay_buffer) >= sample_threshold:
            samples = np.random.choice(len(replay_buffer), batch_size, replace=True)
            states, actions, rewards, next_states, dones = zip(*[replay_buffer[i] for i in samples])

            # q(st,mu(st))
            actions_pred = actor(torch.tensor(states, dtype=torch.float).view(-1, 3)).squeeze()
            critic_input = torch.concatenate((
                torch.tensor(states, dtype=torch.float).view(-1, 3),
                actions_pred.view(-1, 1)), dim=1)
            q = critic(critic_input).squeeze()

            # actor update
            loss = torch.mean(-q)
            optimizer_a.zero_grad()
            loss.backward()
            optimizer_a.step()

            # q(st+1,at+1)
            with torch.no_grad():
                next_action_pred = actor(torch.tensor(next_states, dtype=torch.float).view(-1, 3)).squeeze()
                critic_next_input = torch.concatenate((torch.tensor(next_states, dtype=torch.float).view(-1, 3),
                                                       next_action_pred.view(-1, 1)), dim=1)
                q_next = critic(critic_next_input).squeeze()

            target = torch.tensor(rewards, dtype=torch.float).unsqueeze(-1) + gamma * q_next.unsqueeze(-1).detach() * (
                    1 - torch.tensor(dones, dtype=torch.float).unsqueeze(-1))

            # critic update
            q_without_action_pred = critic(torch.concatenate((torch.tensor(states, dtype=torch.float).view(-1, 3),
                                                              torch.tensor(actions, dtype=torch.float).view(-1, 1)),
                                                             dim=1))
            loss = torch.mean(torch.square(target - q_without_action_pred))

            optimizer_c.zero_grad()
            loss.backward()
            optimizer_c.step()

            td_errors[k] = loss.item()
        print(f"Iteration {k + 1} with {episode_len} steps. TD Errors {td_errors[k]}.")

    plt.subplots(2, 1, figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(td_errors)
    plt.title('The accumulated TD Errors in each iteration')
    plt.subplot(2, 1, 2)
    plt.plot(return_list)
    plt.title('The return gained in a single episode')
    plt.show()

    return critic, actor
