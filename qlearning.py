# This implements:
#  1) train() - train the Q-table using Q-Learning algorithm
#  2) play() - play maze game using Q-table

import json
import random

import numpy as np
from tqdm import tqdm

from environment import Environment
from params import args


class Qlearning:
    def __init__(self):

        # Initialize Q-table (state space size = 201*201, action space size: 5 ("left", "up", "right", "down", "stay")
        self.q_table = np.zeros((40401, 5), dtype=np.single)  # float32
        # self.print_q_table()

        # Initialize environment
        self.env = Environment()

    # Train weights in Q-table. This is the implementation of the Q-learning algorithm
    def train(self):

        print('\nStarted training')
        explor_rate = args.start_explor_rate
        rewards_all_episodes = []

        for episode in tqdm(range(args.episodes)):

            # Reset the state of the environment back to the starting state
            self.env.reset()
            state = 0

            # Reset rewards
            rewards_curr_episode = 0.0

            # for step in tqdm(range(args.steps)):
            for step in range(args.steps):

                # Decide to explor or exploit
                if random.uniform(0, 1) > explor_rate:
                    # Exploit - take action with highest Q-value
                    action = np.argmax(self.q_table[state, :])
                else:
                    # Explore - pick random action
                    action = random.randrange(0, 5)  # [0..4]

                new_state, reward, goal, _, _ = self.env.step(action)

                # Q-table update for the Q(s,a)
                self.q_table[state, action] = self.q_table[state, action] * (1.0 - args.ql_lr) + \
                                              args.ql_lr * (
                                                      reward + args.disc_rate * np.max(self.q_table[new_state, :]))

                # Set new state
                state = new_state

                # Add new reward
                rewards_curr_episode += reward

                # See if episode is finished
                if goal:
                    self.print_status(goal, episode, step, state, rewards_curr_episode, explor_rate)
                    break

            if episode % 10 == 0:
                self.print_q_table()
                self.print_status(goal, episode, step, state, rewards_curr_episode, explor_rate)

            # Reduce the exploration rate
            explor_rate = args.end_explor_rate + (args.start_explor_rate - args.end_explor_rate) \
                          * np.exp(- args.explor_decay_rate * episode)
            # Simplified (assuming min_explr_rate = 0 and max_explr_rate = 1:
            # explor_rate = np.exp(- args.explor_decay_rate * episode)

            # Add current episode to total rewards list
            rewards_all_episodes.append(rewards_curr_episode)

        self.print_stats(args.episodes, rewards_all_episodes)

        # Save Q-table for further use

        self.save_model()

        print('Completed training')

    def print_status(self, goal, episode, step, state, rewards_curr_episode, explor_rate=0.0):
        tqdm.write(f'Goal={goal}, episode={episode}, step={step}, state={state}, rewards={rewards_curr_episode:.2f}, '
                   f'explr_rate={explor_rate:.3f}, non_zero={np.count_nonzero(self.q_table)}, '
                   f'revisited={self.env.revisited}, fires={self.env.fires}')

    def print_q_table(self):
        tqdm.write(f'\nQ-table:\n{self.q_table[0:5]}')

    # Print average reward per hundred episodes
    def print_stats(self, episodes, rewards_all_episodes):
        period = 100
        rewards_per_period = np.split(np.array(rewards_all_episodes), episodes / period)
        count = period
        print(f'\nAverage reward per {period} episodes')
        for reward in rewards_per_period:
            avg_reward = sum(reward / period)
            print(f'{count}: {avg_reward:.2f}')
            count += period

    def save_model(self):
        # Save Q-table training parameters
        with open(args.params_file, 'w') as f:
            json.dump(vars(args), f, indent=2)

        # Save Q-table for further use
        with open(args.ql_model_file, 'wb') as f:
            np.save(f, self.q_table)
            print(f'Model saved to: {args.ql_model_file}')

        # Save Q-table for visual inspection
        np.savetxt(args.ql_model_csv_file, self.q_table, delimiter=",")

    def load_model(self):
        with open(args.ql_best_model_file, 'rb') as f:
            self.q_table = np.load(f)
            print(f'Model loaded from: {args.ql_best_model_file}')

    # Solve maze with pretrained Q-table
    def play(self):

        # Load pretrained Q-table
        self.load_model()

        print('\nStarted solving maze')

        # Reset the state of the environment back to the starting state
        self.env.reset()
        state = 0

        # Reset rewards
        rewards_curr_episode = 0.0

        for step in range(args.steps):

            # No explore, just exploit
            action = np.argmax(self.q_table[state, :])

            # Step in the environment
            new_state, reward, goal, _, _ = self.env.step(action)

            # Set new state
            state = new_state

            # Add new reward
            rewards_curr_episode += reward

            # See if episode is finished
            if goal:
                self.print_status(goal, 0, step, state, rewards_curr_episode)
                break

        print("\nCompleted solving maze")
