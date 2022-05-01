# This implements:
#  1) train() - train the Q-table using Q-Learning algorithm
#  2) play() - play maze game using Q-table

import random

import numpy as np
from tqdm import tqdm

from environment import Environment
# Including this reads command line parameters into "args"
from params import args


class Qlearning:
    def __init__(self):

        # Initialize Q-table (state space size = 201*201, action space size: 5 ("left", "up", "right", "down", "stay")
        self.q_table = np.zeros((40401, 5), dtype=np.single)  # float32
        self.print_q_table()

        # Initialize environment
        self.env = Environment()

    # Print Q-Table
    def print_q_table(self):
        print(f'\nQ-table:\n{self.q_table[0:10]}')

    # Train weights in Q-table. This is the implementation of the Q-learning algorithm
    def train(self):

        print("Started training")
        explor_rate = args.explor_rate
        rewards_all_episodes = []

        for episode in tqdm(range(args.episodes)):

            # Reset the state of the environment back to the starting state
            self.env.reset()
            state = 0

            # Reset rewards
            rewards_curr_episode = 0.0

            for step in range(args.steps):

                # Decide to explor or exploit
                if random.uniform(0, 1) > explor_rate:
                    # Exploit - take action with highest Q-value
                    action = np.argmax(self.q_table[state, :])
                else:
                    # Explore - pick random action
                    action = random.randrange(0, 5)  # [0..4]

                new_state, reward, goal = self.env.step(action)

                # Q-table update for the Q(s,a)
                self.q_table[state, action] = self.q_table[state, action] * (1 - args.lr) + \
                                              args.lr * (reward + args.disc_rate * np.max(self.q_table[new_state, :]))

                # Set new state
                state = new_state

                # Add new reward
                rewards_curr_episode += reward

                # See if episode is finished
                if goal:
                    print(f"\nGoal reached: episode={episode}, y=199, x=199, state={state}, steps={step}, "
                          f"rewards={rewards_curr_episode:.5f}")
                    break

            if episode % 10 == 0:
                self.print_q_table()
                print(f"Episode={episode}, step={step}, state={state}, rewards={rewards_curr_episode:.5f}, "
                      f"explr_rate={explor_rate:.5f}, non_zero={np.count_nonzero(self.q_table)}")

            # Reduce the exploration rate
            # explor_rate = args.end_explor_rate + (args.start_explor_rate - args.end_explor_rate) \
            #                                    * np.exp(- args.explor_decay_rate * episode)
            # Simplified (assuming min_explr_rate = 0 and max_explr_rate = 1:
            explor_rate = np.exp(- args.explor_decay_rate * episode)

            # Add current episode to total rewards list
            rewards_all_episodes.append(rewards_curr_episode)

        self.print_stats(args.episodes, rewards_all_episodes)

        # Save updated Q-table
        np.savetxt('qtable.csv', self.q_table, delimiter=",")

        print("Completed training")

    # Print average reward per hundred episodes
    def print_stats(self, episodes, rewards_all_episodes):
        period = 100
        rewards_per_period = np.split(np.array(rewards_all_episodes), episodes / period)
        count = period
        print(f'\nAverage reward per{period} episodes')
        for reward in rewards_per_period:
            avg_reward = sum(reward / period)
            print(count, ": ", str(avg_reward))
            count += period

    # Play using weights in Q-table
    def play(self):
        print("TODO")
