"""[Source] Chapter 6 M. Lapan, Deep Reinforcement Learning Hands-On: Apply modern RL methods to practical problems of chatbots,
robotics, discrete optimization, web automation, and more. Packt Publishing Ltd, 2020."""

import collections
import random
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import dqn_model
from environment_with_fire import EnvironmentWF
from params import args

torch.set_default_dtype(torch.float32)

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon, device):
        done_reward = None

        if np.random.random() < epsilon:
            action = random.randrange(5)
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()

        # buff_size = len(self.exp_buffer)
        # if buff_size % 1000 == 0 and buff_size < args.replay_start:
        #     print(f'Buffer length = {len(self.exp_buffer)}')

        return done_reward


def calc_loss(batch, net, tgt_net, device):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * args.gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def calc_loss_double_dqn(batch, net, tgt_net, gamma,
                         device="cpu", double=True):
    # states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        if double:
            next_state_acts = net(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_vals = tgt_net(next_states_v).gather(
                1, next_state_acts).squeeze(-1)
        else:
            next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, exp_sa_vals)


if __name__ == "__main__":

    device = torch.device(args.device)
    print('\nCurrent configuration:\n')
    # pprint(vars(args), sort_dicts=False)
    pprint(vars(args))

    # env = wrappers.make_env(args.env)
    env = EnvironmentWF()

    net = dqn_model.DQN(env.maze.shape, 5).to(device)
    tgt_net = dqn_model.DQN(env.maze.shape, 5).to(device)
    writer = SummaryWriter(comment="-" + 'Maze')
    print(net)

    buffer = ExperienceBuffer(args.memory_size)
    agent = Agent(env, buffer)
    epsilon = args.start_explor_rate

    optimizer = optim.Adam(net.parameters(), lr=args.dqn_lr)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = -99999999999.
    loss_t = None

    while True:
        frame_idx += 1
        epsilon = max(args.end_explor_rate, args.start_explor_rate - frame_idx / args.explor_decay_last_frame)

        reward = agent.play_step(net, epsilon, device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])

            print(f'\n==============================================================================================')
            print(f'Frames processed:{frame_idx:d}, done {len(total_rewards):d} games, mean_reward {m_reward:.3f}, '
                  f'best_reward {best_m_reward:.3f}, eps {epsilon:.2f}, speed {speed:.2f} f/s')
            print(f'==============================================================================================\n')

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            # if loss_t is not None:
            #     writer.add_scalar("loss", loss_t, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), "maze-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print(f'\n---------------------------------------------------------------------------------------')
                    print(f"Best mean reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                    print(f'---------------------------------------------------------------------------------------\n')
                best_m_reward = m_reward
            if m_reward > args.mean_reward_bound:
                print("Solved in %d frames!" % frame_idx)
                print('\nCurrent configuration:\n')
                pprint(vars(args), sort_dicts=False)
                break

        if len(buffer) < args.replay_start:
            continue

        if frame_idx % args.target_update == 0:
            print('Updated target NN')
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()

        if frame_idx % 1000 == 0:
            print(f'Frames processed:{frame_idx:d}, loss={loss_t:.5f}')

    # writer.close()
