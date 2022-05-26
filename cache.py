# Preload all maze to memory for faster training

import numpy as np
from tqdm import tqdm

from params import args
from read_maze import load_maze, get_local_maze_information


class Cache:
    def __init__(self):
        # This should be loaded only once
        load_maze()

        self.maze = np.zeros((args.maze_size + 2, args.maze_size + 2, 2), dtype=int)  # or dtype=np.float32

        self.fill_cache(2)
        print("Cache initialization completed")

    # This loads all maze into memory for faster access. It takes about 2-3 minutes to read all maze
    def fill_cache(self, pattern=1):

        print("Reading maze into cache")

        if pattern == 1:
            # This is simple maze for testing.
            self.maze[1:args.maze_size + 1, 1:args.maze_size + 1, 0] = 1.0
            # add few bars:
            self.maze[8, :-2, 0] = 0.0
            self.maze[10, 2:, 0] = 0.0
        else:
            for y in tqdm(range(1, 200, 3)):
                for x in range(1, 200, 3):
                    around = get_local_maze_information(y, x)
                    self.maze[y - 1, x - 1] = around[0, 0]
                    self.maze[y - 1, x - 0] = around[0, 1]
                    self.maze[y - 1, x + 1] = around[0, 2]
                    self.maze[y - 0, x - 1] = around[1, 0]
                    self.maze[y - 0, x - 0] = around[1, 1]
                    self.maze[y - 0, x + 1] = around[1, 2]
                    self.maze[y + 1, x - 1] = around[2, 0]
                    self.maze[y + 1, x + 0] = around[2, 1]
                    self.maze[y + 1, x + 1] = around[2, 2]

        print("Completed reading maze into cache")

    # This is fast get maze information version, that does not support fire
    def get_cached_local_maze_information_no_fire_support(self, y, x):
        return self.maze[y - 1:y + 2, x - 1:x + 2]

    # It takes about 2-3 minutes to read all maze. Thus save it to file, so that do not need to wait evey run
    def save_cache(self):
        with open('data/maze_cache_v2.npy', 'wb') as f:
            np.save(f, self.maze)

    # It takes about 2-3 minutes to read all maze. Thus it is faster to read it from file saved last time
    def load_cache(self):
        with open('data/maze_cache_v2.npy', 'rb') as f:
            self.maze = np.load(f)

    # Save maze with fires into CSV for visual inspection
    def save_csv(self):
        with open('data/maze_with_fires.csv', 'w') as f:
            for y in range(0, 201):  # y = [0..200]
                # Generate one line
                line = ""
                for x in range(0, 201):  # x = [0..200]
                    # around = get_local_maze_information(y, x)
                    if self.maze[y, x, 0] == 0:
                        # Print wall cell
                        line = line + f'-1,'
                    else:
                        # Print empty or fire cell
                        line = line + f'{self.maze[y, x, 1]:1d},'
                f.write(f'{line}\n')

    # Print maze to screen
    def render(self):
        for y in range(0, 201):  # y = [0..200]
            for x in range(0, 201):  # x = [0..200]
                # around = get_local_maze_information(y, x)

                if self.maze[y, x, 0] == 0:
                    # Print wall cell
                    print(f'{self.maze[y, x, 0]:1d}', end="")
                else:
                    # Print empty or fire cell
                    print(f'{self.maze[y, x, 1]:1d}', end="")
            print('\n')
