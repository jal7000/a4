# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import torch
import numpy as np
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from padm_env import create_env
from constants import BLACKHOLE_POINTS, HELL_COORDINATE_POINTS
import os

from datetime import datetime

from datetime import datetime

now = datetime.now()


# User definitions:
# -----------------
train_dqn = True
test_dqn = False
render = False

#! Define env attributes (environment specific)
no_actions = 4
no_states = 2

# Hyperparameters:
# ----------------
learning_rate = 0.006
gamma = 0.99
buffer_limit = 80_0
batch_size = no_actions*16
num_episodes = 100_00
max_steps = 10_0

goal_coordinates = (5, 5)

hell_state_coordinates = HELL_COORDINATE_POINTS
blackhole_coordinates = BLACKHOLE_POINTS



# Main:
# -----
if train_dqn:
    env = create_env(goal_coordinates=goal_coordinates,
               hell_state_coordinates=hell_state_coordinates,
               blackhole_coordinates=blackhole_coordinates)

    #! Initialize the Q Net and the Q Target Net
    q_net = Qnet(no_actions=no_actions, no_states=no_states)
    q_target = Qnet(no_actions=no_actions, no_states=no_states)
    q_target.load_state_dict(q_net.state_dict())

    #! Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 20
    episode_reward = 0.0
    optimizer = optim.Adam(q_net.parameters(),
                           lr=learning_rate)

    rewards = []

    for n_epi in range(num_episodes):
        #! Epsilon decay (Please come up with your own logic)
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/1482))  # ! Linear annealing from 8% to 1%

        s, _ = env.reset()
        done = False
        #! Define maximum steps per episode, here 1,000
        for _ in range(max_steps):
            #! Choose an action (Exploration vs. Exploitation)
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _ = env.step(a)

            done_mask = 0.0 if done else 1.0

            #! Save the trajectories
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            episode_reward += r
            if(render):
                env.render()

            if done:
                break

        # if memory.size() > 8000:
        #     train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0

        #! Define a stopping condition for the game:
        if rewards[-10:] == [max_steps]*10:
            break
        
    env.close()

    #! Save the trained Q-net
    torch.save(q_net.state_dict(), "dqn.pth")

    #! Plot the training curve
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    env = create_env(goal_coordinates=goal_coordinates,
               hell_state_coordinates=hell_state_coordinates)

    dqn = Qnet(no_actions=no_actions, no_states=no_states)
    dqn.load_state_dict(torch.load("dqn.pth"))

    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            # Encourage taking suboptimal actions to increase path length
            action_values = dqn(torch.from_numpy(s).float()).detach().numpy()
            # From the pth file Sorted the actions
            sorted_actions = action_values.argsort()[::-1]
            #From the Actions sorted based on values taking any random
            suboptimal_action_index = np.random.choice(range(1, len(sorted_actions)))
            # Generate the Sorted Actions based on values
            action = sorted_actions[suboptimal_action_index]

            s_prime, reward, done, _ = env.step(action)
            s = s_prime

            episode_reward += reward
            env.render()

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
