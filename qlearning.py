# This implements:
#  1) train() - train the Q-table using Q-Learning algorithm
#  2) play() - play maze game using Q-table

import json
import random

import numpy as np
#from tensorboardX import SummaryWriter
from tqdm import tqdm

from environment import Environment
from params import args

#writer = SummaryWriter(comment=f'-{args.description}')

# Create separate random number generator
r = random.Random(1)


class Qlearning:
    def __init__(self):
        self.explor_rate = 0
        self.neglect_fire = 0
        # Initialize Q-table. The state size is as below:
        # As x=[0..200] y=[0..200], states = [0..200*201+200] = [0..40400]  => 40400+1
        # action space size: 5 ("left", "up", "right", "down", "stay")
        self.q_table = np.zeros(((40400 + 1) * 16, 5), dtype=np.float32)
        self.env = Environment()
        self.phase = 0
        self.win_history = []
        self.win_rate = 0.0
        self.hist_len = 0
        self.actions_history = np.zeros((5,), dtype=int)

    # Train weights in Q-table. This implements the Q-learning algorithm
    def train(self, neglect_fire, episodes, steps, start_explor_rate):
        self.neglect_fire = neglect_fire
        self.explor_rate = start_explor_rate
        self.win_history = []
        self.win_rate = 0.0
        self.hist_len = 300

        for episode in tqdm(range(episodes)):
            state = self.env.reset(self.neglect_fire)
            rewards_curr_episode = 0.0
            self.actions_history[:] = 0

            for step in range(steps):
                # Decide to explor or exploit
                action = np.argmax(self.q_table[state, :]) if r.uniform(0, 1) > self.explor_rate else r.randrange(0, 5)

                # Take a step
                new_state, reward, goal = self.env.step(action, self.neglect_fire)

                # Q-table update for the Q(s,a).
                # Bellman update - averages between old and new values of Q, rather than just overriding with the
                # new value. It is an approximation of the current state and action by summing the immediate
                # reward with the discounted value of the next state.
                #
                # States > 40400, [0..40400]
                # or x=[0..200] y=[0..200], states = [0..200*201+200] = [0..40400] - are fire states

                # Do not comment out this or this will slow down learning with fires significantly.
                if self.neglect_fire or state > 40400:
                    self.q_table[state, action] = self.q_table[state, action] * (1.0 - args.ql_lr) + \
                                                  args.ql_lr * (reward + args.disc_rate * np.max(
                        self.q_table[new_state, :]))

                # Set new state
                state = new_state

                # Add new reward
                rewards_curr_episode += reward
                self.actions_history[action] += 1

                # See if episode is finished
                if goal:
                    self.print_status(goal, episode, step, state, rewards_curr_episode)
                    break

            if episode % 50 == 0:
                tqdm.write(f'\nQ-table:\n{self.q_table[202:207]}')
                self.print_status(goal, episode, step, state, rewards_curr_episode)
                #self.plot_status(episode, step, rewards_curr_episode)
                self.play()

            # Reduce the exploration rate
            self.explor_rate = args.end_explor_rate + (start_explor_rate - args.end_explor_rate) \
                               * np.exp(- args.ql_explor_decay_rate * episode)

            # Calculate win rate
            self.update_win_rate(goal)

        #writer.flush()
        self.phase += 1

    def print_status(self, goal, episode, step, state, rewards_curr_episode):
        # Print the current status of the game
        tqdm.write(
            f'Goal={goal}|phase={self.phase}|neglect_fire={self.neglect_fire}|episode={episode}|steps={step}|'
            f'state={state}|rewards={rewards_curr_episode:.2f}|explr_rate={self.explor_rate:.3f}|'
            f'non_zero={np.count_nonzero(self.q_table)}|revisited={self.env.revisited}|fires={self.env.fires}|'
            f'walls={self.env.walls}|win_rate={self.win_rate:.2f}|actions={self.actions_history}')

    '''def plot_status(self, episode, steps, rewards_curr_episode):
        # Plot results on the Tensorboard (comment out as this is useful when training)
        writer.add_scalar("steps/episodes", steps, episode)
        writer.add_scalar("rewards/episodes", rewards_curr_episode, episode)
        writer.add_scalar("epsilon/episodes", self.explor_rate, episode)
        writer.add_scalar("non_zero/episodes", np.count_nonzero(self.q_table), episode)
        writer.add_scalar("revisited/episodes", self.env.revisited, episode)
        writer.add_scalar("fires/episodes", self.env.fires, episode)
        writer.add_scalar("walls/episdoes", self.env.walls, episode)
        writer.add_scalar("wins/episodes", self.win_rate, episode)'''

    def save_model(self, ql_model_file):
        # Save Q-table training parameters
        with open(args.params_file, 'w') as f:
            json.dump(vars(args), f, indent=2)

        # Save Q-table for further use
        with open(ql_model_file, 'wb') as f:
            np.save(f, self.q_table)
            print(f'\nModel saved to: {ql_model_file}')

        # Save Q-table for visual inspection
        np.savetxt(args.ql_model_csv_file, self.q_table, delimiter=",")

    def load_model(self, ql_model_file):
        with open(ql_model_file, 'rb') as f:
            self.q_table = np.load(f)
            print(f'\nModel loaded from: {ql_model_file}')

    # Solve maze with pretrained Q-table
    # This requires Q-table to be already in memory
    def play(self):
        # Reset the state of the environment back to the starting state
        state = self.env.reset(self.neglect_fire)

        # Reset rewards
        rewards_curr_episode = 0.0
        self.actions_history[:] = 0

        for step in tqdm(range(args.steps)):

            # No explore, just exploit
            action = np.argmax(self.q_table[state, :])

            # Step in the environment
            new_state, reward, goal = self.env.step(action, self.neglect_fire)

            # Set new state
            state = new_state

            # Add new reward
            rewards_curr_episode += reward
            self.actions_history[action] += 1

            # See if episode is finished
            if goal:
                print(f'Play attempt: WIN. Goal is reached in {step} steps')
                self.print_status(goal, self.neglect_fire, step, state, rewards_curr_episode)
                break

        if not goal:
            print(f'Play attempt: LOST. Goal is NOT reached in {step} steps')

    def update_win_rate(self, goal):
        self.win_history.append(goal)
        if len(self.win_history) > self.hist_len:
            self.win_rate = sum(self.win_history[-self.hist_len:]) / self.hist_len

    def copy_walls(self):
        block_len = 40401  # 201 * 201
        for block_num in range(1, 16):
            self.q_table[block_num * block_len: (block_num + 1) * block_len, :] = self.q_table[0: block_len, :]
