import sys

sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()
    for t in range(50):
        env.render()
        action = random.choice(
            env.action_space)  # equivalent to uniformly generate a sequence of random actions for each step
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state + (np.array([1, 1]))}, Reward: {reward}, Done: {done}")
        # change below to make the process continuing task
        # if done:
        #     break

    # Add policy
    policy_matrix = np.random.rand(env.num_states,
                                   len(env.action_space))  # normal distribution for the probability to choose any action
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    env.add_policy(policy_matrix)

    # Add state values
    values = np.random.uniform(0, 10, (env.num_states,))  # for example, which doesn't mean anything in practice
    env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=5)
