# This is an implementation of the NN aimed at estimation of the Q-Values for each state-action pair in a given
# environment.
#
# States -> NN -> Q-values
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from buffer import ReplayBuffer
from environment import Environment
from params import args
from utilities import plot

torch.manual_seed(123)
device = torch.device(args.device)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        # DQN input vector size is 19 (around dimensions are 3x3x2=18, plus 1 state number = 19)
        state_vector_size = 19
        # Passing 19+19=38 (past and current states)
        # state_vector_size = 38
        self.fc1 = nn.Linear(in_features=state_vector_size, out_features=128)  # 24, or 128
        self.fc2 = nn.Linear(in_features=128, out_features=256)  # 24 and 32, or 128 and 256
        self.out = nn.Linear(in_features=256, out_features=args.actions_available)  # 32, or 256

    # forward() is executed when policy_net(t) or target_net(t) is called
    def forward(self, t):
        # print("t=", type(t), t.shape, t)
        t = t.flatten(start_dim=1)  # TODO: <- do not need this, as input is already flat vector
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


def get_current(policy_net, states, actions):
    return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))


def get_next(target_net, next_states):
    final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
    non_final_state_locations = (final_state_locations == False)
    non_final_states = next_states[non_final_state_locations]
    batch_size = next_states.shape[0]
    values = torch.zeros(batch_size).to(device)
    values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
    return values


# Loss - comparison of the outputted Q-values and the target Q-values (from the right side of Bellman equation).

# Use a stack of a few (four) consecutive frames in an order of which they appeared, to represent a single input.

# Use some convolutional layers, followed by non-linear activation function., and then the couple of fully connected
# layers. There will be no activation function, as we want raw Q-values of the network.

# Output - would consist of five nodes ("left", "up", "right", "down", "stay")

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'new_state', 'reward')
)


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.new_state)

    return (t1, t2, t3, t4)


def train():
    env = Environment()

    # 1. Initialize replay buffer
    buffer = ReplayBuffer(args.memory_size)

    # 2. Initialize network with random weights
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=args.dqn_lr)

    explor_rate = args.start_explor_rate
    rewards_all_episodes = []

    episode_steps = []

    loss = 0

    actions_history = np.zeros((5,), dtype=int)

    for episode in tqdm(range(args.episodes)):
        # 3. Initialize starting state for the episode
        # The first state is 000000, 000001 (19+19)
        state, t_state = env.reset()
        t_first_state = t_state.clone()  # to check if the action from 1st state is correct

        # Reset rewards
        rewards_curr_episode = 0.0
        actions_history[:] = 0

        for step in range(args.steps):

            # 4. Select action
            if random.uniform(0, 1) > explor_rate:
                with torch.no_grad():
                    t_action = policy_net(t_state).argmax(dim=1).to(device)  # exploit
            else:
                action = random.randrange(args.actions_available)
                t_action = torch.tensor([action]).to(device)  # explore

            actions_history[int(t_action[0])] += 1

            # 5. Execute action in the environment
            # 6. Observe reward and new (next) state
            new_state, reward, goal, t_reward, t_new_state = env.step(t_action)

            # 7. Store experience in replay buffer
            buffer.append(Experience(t_state, t_action, t_new_state, t_reward))

            # Set new state
            t_state = t_new_state

            # Add new reward
            rewards_curr_episode += reward

            if len(buffer) >= args.batch_size:
                experiences = buffer.sample(args.batch_size)

                t_states, t_actions, t_rewards, t_next_states = extract_tensors(experiences)

                current_q_values = get_current(policy_net, t_states, t_actions)
                next_q_values = get_next(target_net, t_next_states)
                target_q_values = (next_q_values * args.gamma) + t_rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step % 200 == 0:  # 10000
                # Check, what is the 1st recommended action
                with torch.no_grad():
                    t_action = policy_net(t_first_state).argmax(dim=1).to(device)

                print(f'episode={episode}, step={step}, loss={loss:.5f}, state={new_state}, '
                      f'reward={rewards_curr_episode:.3f}, rate={explor_rate:.3f}, actions={actions_history}, '
                      f'total_visited={env.get_total_visited()}, visited={env.get_visited()}, '
                      f'revisited={env.revisited}, capacity={len(buffer)}, 1st_action={t_action[0]}')
                # Let's see what we learned: what is the first step from y=1, x=1?

            # See if episode is finished
            if goal:
                episode_steps.append(step)
                plot(episode_steps, 100)
                print(f'##### GOAL REACHED ######, episode={episode}, step={step}, state={new_state}')
                break

        # Reduce the exploration rate
        explor_rate = args.end_explor_rate + (args.start_explor_rate - args.end_explor_rate) \
                      * np.exp(- args.ql_explor_decay_rate * episode)

        # Simplified (assuming min_explr_rate = 0 and max_explr_rate = 1:
        # explor_rate = np.exp(- args.explor_decay_rate * episode)

        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
