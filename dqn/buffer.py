"""[Source] Chapter 6 M. Lapan, Deep Reinforcement Learning Hands-On: Apply modern RL methods to practical problems of chatbots,
robotics, discrete optimization, web automation, and more. Packt Publishing Ltd, 2020."""

import random


class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.push_count = 0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # TODO: below line is slow
            self.buffer[self.push_count % self.capacity] = experience
            # TODO: below line is fast, but agent does not go far
            # self.memory = []
            # self.memory.append(experience)
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
