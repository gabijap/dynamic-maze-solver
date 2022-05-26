"""
This is an experimental script for using the DQN with a linear neural network to solve the maze. It was experimented with
smaller mazes and then adapted to the original maze, however, this needs future work.

The code in this script is adapted from two sources:
[Source code] https://www.samyzaf.com/ML/rl/qmaze.html
[Source code] https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch_2020.py
"""

from __future__ import print_function

import argparse
import copy
import datetime
import random
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm

timestamp = time.strftime('%d%H%M')

parser = argparse.ArgumentParser(description='Maze solver with Linear DQN')
parser.add_argument('--comment', type=str, default=f'LDQN-({timestamp})')
parser.add_argument('--reward_scale', type=float, default=2.0)  # 2.0  # 10.0
parser.add_argument('--epsilon', type=float, default=1.0)
parser.add_argument('--eps_end', type=float, default=0.1)  # 0.01
parser.add_argument('--eps_dec', type=float, default=0.00005)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--episodes', type=int, default=2000)
parser.add_argument('--steps', type=int, default=2500)  # 20x20->300, 40x40->500, 80x80->2000
parser.add_argument('--batch_size', type=int, default=64)  # 64 # 128
parser.add_argument('--tau', type=int,
                    default=120)  # 120  # default=100, [40x40 tau=250 was bad, tau=70 was longer, 120 ended quicker, 140 was good]
# 20x20 tau=120 could not converge for a while.
parser.add_argument('--max_memory', type=int, default=204800)  # 160*160*8

args = parser.parse_args()

print('\nCurrent configuration:\n')
pprint(vars(args))

comment = args.comment
reward_scale = args.reward_scale
epsilon = args.epsilon
eps_end = args.eps_end
eps_dec = args.eps_dec
lr = args.lr
episodes = args.episodes
steps = args.steps
batch_size = args.batch_size
tau = args.tau

writer = SummaryWriter(comment=f'-{args.comment}')

# DEFINE MAZE

'''
maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
])
'''

maze10 = np.array([
    [1., 0., 1., 0., 1., 1., 1., 1., 0., 1.],
    [1., 1., 1., 1., 1., 0., 1., 1., 0., 1.],
    [1., 1., 1., 1., 1., 0., 1., 1., 1., 1.],
    [0., 0., 1., 0., 0., 1., 0., 1., 1., 1.],
    [1., 1., 1., 1., 0., 1., 0., 0., 0., 1.],
    [1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.]
])

maze11 = np.array([
    [1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
    [0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1.],
    [1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 1.],
    [1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1.],
    [1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.]
])

maze2 = np.concatenate((maze10, maze10), axis=0)
maze3 = np.concatenate((maze2, maze2), axis=1)

maze4 = np.concatenate((maze3, maze3), axis=0)
maze5 = np.concatenate((maze4, maze4), axis=1)

maze6 = np.concatenate((maze5, maze5), axis=0)
maze7 = np.concatenate((maze6, maze6), axis=1)  # 80x80

maze8 = np.concatenate((maze7, maze7), axis=0)
maze9 = np.concatenate((maze8, maze8), axis=1)  # 160x160

# EDIT HERE:
# maze = maze10
# maze = maze11
# maze = maze3  # 20x20 - 2 minutes
maze = maze5  # 40x40 - 10 minutes to solve, set steps=~500 per episode
# maze = maze7  # 80x80 -
# maze = maze9 # 160x160 - takes too long and does not solve

random.seed(2022)

flag_list = [0, 1, 2, 3, 5, 6, 7, 8]
#  [0, 1, 2
#   3,    5
#   6, 7, 8]

time_list = [0, 1, 2]


# load maze
def load_maze():
    global maze_cells

    '''
    maze = np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
    [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    [ 1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]
    ])'''

    # maze = np.load("COMP6247Maze20212022.npy", allow_pickle=False, fix_imports=True)
    # maze = np.ones((200, 200))

    maze_cells = np.zeros((maze.shape[0], maze.shape[1], 2), dtype=int)
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            maze_cells[i][j][0] = maze[i][j]
            # load the maze, with 1 denoting an empty location and 0 denoting a wall
            maze_cells[i][j][1] = 0
            # initialized to 0 denoting no fire


def get_local_maze_information(x, y):
    global maze_cells
    random_location = random.choice(flag_list)
    around = np.zeros((3, 3, 2), dtype=int)
    for i in range(maze_cells.shape[0]):
        for j in range(maze_cells.shape[1]):
            if maze_cells[i][j][1] == 0:
                pass
            else:
                maze_cells[i][j][1] = maze_cells[i][j][1] - 1  # decrement the fire time

    for i in range(3):
        for j in range(3):
            if x - 1 + i < 0 or x - 1 + i >= maze_cells.shape[0] or y - 1 + j < 0 or y - 1 + j >= maze_cells.shape[1]:
                around[i][j][0] = 0  # this cell is outside the maze, and we set it to a wall
                around[i][j][1] = 0
                continue
            around[i][j][0] = maze_cells[x - 1 + i][y - 1 + j][0]
            around[i][j][1] = maze_cells[x - 1 + i][y - 1 + j][1]
            if i == random_location // 3 and j == random_location % 3:
                if around[i][j][0] == 0:  # this cell is a wall
                    continue
                ran_time = random.choice(time_list)
                around[i][j][1] = ran_time + around[i][j][1]
                maze_cells[x - 1 + i][y - 1 + j][1] = around[i][j][1]
    return around

# Adapted from [Source code] https://www.samyzaf.com/ML/rl/qmaze.html
class Maze:
    def __init__(self, maze, rat=(0, 0)):

        load_maze()  # loads the maze environment

        self._maze = np.array(maze)  # REMOVE
        self.nrows = maze.shape[0]
        self.ncols = maze.shape[1]

        print(f'Maze shape: {maze.shape}')

        self.target = (self.nrows - 1, self.ncols - 1)  # 199, 199
        # self.target = (199, 199)
        self.free_cells = [(r, c) for r in range(self.nrows) for c in range(self.ncols) if
                           self._maze[r, c] == 1.0]  # REMOVE
        self.free_cells.remove(self.target)  # REMOVE
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)  # REMOVE
        row, col = rat
        self.maze[row, col] = 0.5  # REMOVE
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.nrows * self.ncols
        # self.min_reward = -0.5 * 199 * 199
        self.total_reward = 0
        self.visited = set()
        self.trace = [self.state]

    def update_state(self, action):
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:  # REMOVE
            self.visited.add((rat_row, rat_col))  # mark visited cell # REMOVE

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == 0:
                ncol -= 1
            elif action == 1:
                nrow -= 1
            if action == 2:
                ncol += 1
            elif action == 3:
                nrow += 1
        else:
            nmode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        if rat_row == self.nrows - 1 and rat_col == self.ncols - 1:
            # if rat_row == 199 and rat_col == 199:
            return +1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.25 / reward_scale  # -0.25
        if mode == 'invalid':
            return -0.75 / reward_scale  # -0.75
        if mode == 'valid':
            return -0.01 / reward_scale  # -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        self.trace.append(self.state)
        return envstate, reward, status

    def observe(self):
        row, col, valid = self.state
        around = get_local_maze_information(row, col)
        around[1][1] = 0.5
        envstate = around[:, :, 0:1].reshape(1, -1)
        return envstate

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        if rat_row == self.nrows - 1 and rat_col == self.ncols - 1:
            # if rat_row == 199 and rat_col == 199:
            return 'win'
        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        around = get_local_maze_information(row, col)
        if around[0][1][0] == 0:
            actions.remove(1)
        if around[2][1][0] == 0:
            actions.remove(3)
        if around[1][0][0] == 0:
            actions.remove(0)
        if around[1][2][0] == 0:
            actions.remove(2)

        return actions


def show(qmaze):
    # fig, axs = plt.subplots(1, 1, figsize=(36, 24), constrained_layout=True)
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 0.3  # rat cell
    canvas[nrows - 1, ncols - 1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        # state = T.tensor([prev_envstate], device=model.device, dtype=T.float32)
        state = T.tensor(prev_envstate, device=model.device, dtype=T.float32)
        actions = model.forward(state)
        action = T.argmax(actions).item()

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True


# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)


class DeepQNetwork_OLD(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.layer_1 = nn.Linear(in_features=9, out_features=128)  # TODO 256???
        self.layer_2 = nn.Linear(in_features=128, out_features=128)  # TODO 256???
        self.layer_3 = nn.Linear(in_features=128, out_features=64)
        self.output_layer = nn.Linear(in_features=64, out_features=n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, net_input):
        layer_1_output = nn.functional.relu(self.layer_1(net_input))
        layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = nn.functional.relu(self.layer_3(layer_2_output))
        actions = self.output_layer(layer_3_output)
        return actions

"""
Adapted from [Source code] 
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch_2020.py
"""
class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size, eps_end, eps_dec):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        print(f'Network configuration: {self.Q_eval}')

        self.Q_target = copy.deepcopy(self.Q_eval)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def update_target(self):
        state_dict = self.Q_eval.state_dict()
        self.Q_target.load_state_dict(state_dict)
        self.Q_target.to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
        self.Q_target.eval()

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation, valid_actions):
        if np.random.random() > self.epsilon:
            # state = T.tensor([observation], device=self.Q_eval.device, dtype=T.float32)

            state = T.tensor(observation, device=self.Q_eval.device, dtype=T.float32)

            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(valid_actions)
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch], device=self.Q_eval.device, dtype=T.float32)
        new_state_batch = T.tensor(
            self.new_state_memory[batch], device=self.Q_eval.device, dtype=T.float32)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
            self.reward_memory[batch], device=self.Q_eval.device)
        terminal_batch = T.tensor(
            self.terminal_memory[batch], device=self.Q_eval.device)

        ''' (Option 1)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)'''

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next_target = self.Q_target.forward(new_state_batch)
        q_target = reward_batch + self.gamma * T.max(q_next_target, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_end else self.eps_end

        # self.epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        return loss.detach().cpu()


# SHOW

qmaze = Maze(maze)
print(maze.shape)
print(maze.size)
show(qmaze)

# TRAIN

max_memory = 8 * maze.size
print(f'max_memory={max_memory}')

start_time = datetime.datetime.now()

qmaze = Maze(maze)

agent = Agent(gamma=0.99, epsilon=epsilon, batch_size=batch_size, n_actions=4, max_mem_size=max_memory, eps_end=eps_end,
              input_dims=[9], lr=lr, eps_dec=eps_dec)

# agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, max_mem_size=max_memory, eps_end=0.5, eps_dec=0.000071,
# input_dims=[9], lr=0.001)

win_history = []  # history of win/lose game
n_free_cells = len(qmaze.free_cells)
# hsize = qmaze.maze.size // 2  # history window size
hsize = qmaze.maze.shape[0] // 2
win_rate = 0.0
imctr = 1

for episode in tqdm(range(episodes)):
    episode_start_time = time.time()
    loss = 0.0
    rat_cell = random.choice(qmaze.free_cells)
    qmaze.reset(rat_cell)
    # qmaze.reset((1, 1))
    game_over = False

    # get initial envstate (1d flattened canvas)
    envstate = qmaze.observe()
    total_reward = 0.0

    step = 0
    while not game_over and step < steps:
        valid_actions = qmaze.valid_actions()
        if not valid_actions: break
        prev_envstate = envstate

        action = agent.choose_action(prev_envstate, valid_actions)

        # Apply action, get reward and new envstate
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            win_history.append(1)
            game_over = True
        elif game_status == 'lose':
            win_history.append(0)
            game_over = True
        else:
            # print('time_out')
            game_over = False

        total_reward += reward
        # Store episode (experience)
        agent.store_transition(prev_envstate, action, reward, envstate, game_status)

        # Train neural network model
        loss = agent.learn()

        if step % tau == 0:
            # print('updating target model')
            agent.update_target()

        step += 1

    if len(win_history) > hsize:
        win_rate = sum(win_history[-hsize:]) / hsize
        tqdm.write(
            f'win_hist_len={len(win_history)}, win_hist_sum={sum(win_history[-hsize:])}, hsize={hsize}, win_rate={win_rate}')

    # dt = datetime.datetime.now() - start_time
    # t = format_time(dt.total_seconds())
    speed = time.time() - episode_start_time
    tqdm.write(
        f"Epoch:{episode:03d}/{episodes - 1:d},eps:{agent.epsilon:.4f},loss:{loss:.7f},step:{step:d},total_reward:{total_reward:.3f},wins:{sum(win_history):d},win_rate:{win_rate:.3f},speed:{speed:.3f}s/episode")

    # plot_episode.append(episode)
    # plot_loss.append(loss)

    writer.add_scalar("epsilon", agent.epsilon, episode)
    writer.add_scalar("reward", total_reward, episode)
    writer.add_scalar("loss", loss, episode)
    writer.add_scalar("speed", time.time() - episode_start_time, episode)  # sec per iteration

    # print(epoch, n_epoch - 1, loss, step, sum(win_history), win_rate, t))
    # we simply check if training has exhausted all free cells and if in all
    # cases the agent won
    if win_rate > 0.9:
        epsilon = 0.05

    # if sum(win_history[-hsize:]) == hsize and completion_check(agent.Q_eval, qmaze):
    if win_rate > 0.98:
        tqdm.write("Reached 100%% win rate at epoch: %d" % (episode,))
        break
        # TODO: THIS IS HEAVY CHECK ENABLE LATER
        # if completion_check(agent.Q_eval, qmaze):
        #    tqdm.write("Reached 100%% win rate at epoch: %d" % (episode,))
        #    break

writer.close()
