# State      - Description                                   Reward
# S(1,1)     - Agent's starting point                         0
#            - Empty, subject to fire                        -0.04
#            - Empty, if fire for around[i][j][1] time units
#              can not visit                                 -0.04
#            - Wall - can not visit                          -1.0
#            - Repeated cell visit                           -0.25
# G(199,199) - Goal - game over                              +1.0

import numpy as np
import torch

# from cache import Cache
from params import args
from read_maze import load_maze, get_local_maze_information


class Environment:
    def __init__(self):
        # Create the environment
        self.device = torch.device(args.device)
        self.x = 1
        self.y = 1
        self.around = np.zeros((3, 3, 2), dtype=int)

        self.total_visited_cells = set()  # Larger is better during training
        self.visited_cells = set()
        self.revisited = 0  # Less is better after training
        self.no_revisits = False  # Checks model is ready
        self.reward = 0
        self.path = []
        self.trace = []
        self.fires = 0

        # As x != 0 and y != 0 (always), will use x = 0, and y = 0 as special initial state.
        self.t_last_state_vector = torch.zeros((1, 19), dtype=torch.float32, device=self.device)

        # self.maze_cache = Cache()

        self.reset()

    def reset(self):
        # reset to a start position
        self.x = 1
        self.y = 1

        self.total_visited_cells.update(self.visited_cells)
        self.visited_cells = set()
        self.revisited = 0
        self.no_revisits = False
        self.reward = 0
        self.path = []
        self.trace = []
        self.fires = 0

        # Load maze from configuration file
        load_maze()

        self.around = get_local_maze_information(1, 1)
        # self.around = self.maze_cache.get_cached_local_maze_information_no_fire_support(1, 1)

        return 0, self.t_get_state_vector()  # The first state is 000000, 000001 (19+19)

    def check_if_visited(self):
        # Check if cell is already visited before:
        if (self.y, self.x) in self.visited_cells:
            self.reward = args.visited_cell
            self.revisited += 1
        else:
            self.visited_cells.add((self.y, self.x))
            self.reward = args.adjacent_cell

    # Takes next step in the environment. Returns state corresponding to the Q-table row
    def step(self, action):

        # record path (including trace)
        if args.play:
            self.path.append((self.y, self.x, self.around, action))

        # We can take one of 5 actions ("0 - left", "1 - up", "2 - right", "3 - down", "4 - stay")
        # move right (or else do nothing if at the right edge or target location is not empty)
        if action == 2 and self.x < 199 and self.around[1][2][0] == 1:
            if self.around[1][2][1] == 0 or args.neglect_fire:  # There is no fire
                self.x += 1
                if self.x == 199 and self.y == 199:  # goal?
                    self.reward = args.goal_cell
                    return self.results(True)
                else:
                    self.check_if_visited()
            else:  # There is fire (around[1][2][1] > 0). You can not visit this location
                # Coordinates do not change
                self.fires += 1
                self.reward = args.fire_cell

        # move left (or else do nothing if at the left edge or target location is not empty)
        elif action == 0 and self.x > 1 and self.around[1][0][0] == 1:
            if self.around[1][0][1] == 0 or args.neglect_fire:  # There is no fire
                self.x -= 1
                self.check_if_visited()
            else:  # There is fire (around[1][0][1] > 0). You can not visit this location
                # Coordinates do not change
                self.fires += 1
                self.reward = args.fire_cell

        # move up (or else do nothing if at the top edge or target location is not empty)
        elif action == 1 and self.y > 1 and self.around[0][1][0] == 1:
            if self.around[0][1][1] == 0 or args.neglect_fire:  # There is no fire
                self.y -= 1
                self.check_if_visited()
            else:  # There is fire (around[0][1][1] > 0). You can not visit this location
                # Coordinates do not change
                self.fires += 1
                self.reward = args.fire_cell

        # move down (or else do nothing if at the bottom edge or target location is not empty)
        elif action == 3 and self.y < 199 and self.around[2][1][0] == 1:
            if self.around[2][1][1] == 0 or args.neglect_fire:  # There is no fire
                self.y += 1
                if self.x == 199 and self.y == 199:  # goal?
                    self.reward = args.goal_cell
                    return self.results(True)
                else:
                    self.check_if_visited()
            else:  # There is fire (around[2][1][1] > 0). You can not visit this location
                # Coordinates do not change
                self.fires += 1
                self.reward = args.fire_cell

        # stay (do nothing)
        elif action == 4:
            # Should get penalty as staying affects minimum traversal time.
            self.reward = args.stayed_cell
            pass

        # hit the wall
        else:
            self.check_if_visited()
            self.reward = args.wall_cell

        return self.results(False)

    def results(self, goal):
        # If goal reached, try to print the shortest path
        if goal:
            if args.play and not args.train:
                self.save_path_and_trace()

        self.around = get_local_maze_information(self.y, self.x)
        # self.around = self.maze_cache.get_cached_local_maze_information_no_fire_support(self.y, self.x)

        # Return state, reward, goal. State - is a row number in a Q-Table. state = (self.y - 1) * 199 + self.x
        return (self.y - 1) * 199 + self.x, self.reward, goal, \
               torch.tensor([self.reward]).to(self.device), self.t_get_state_vector()

    def get_visited(self):
        return len(self.visited_cells)

    def get_total_visited(self):
        return len(self.total_visited_cells)

    # 1. path with the minimum traversal time from the top left corner (1, 1), to the bottom right corner (199, 199).
    # 2. trace in the environment, at each time unit, what was observed in surroundings, and what action was taken.
    def save_path_and_trace(self):
        with open(args.ql_path_file, 'w') as fp, open(args.ql_trace_file, 'w') as ft:
            print(f'\nShortest path length: {len(self.path)}, revisited: {self.revisited}, fires: {self.fires}\n'
                  f'step, y, x, action (0:left, 1:up, 2:right, 3:down, 4:stay)', file=fp)
            for i in range(len(self.path)):
                print(f'{i + 1}, {self.path[i][0]}, {self.path[i][1]}, {self.path[i][3]}', file=fp)
                print(f'\nStep {i}. Situation around y={self.path[i][0]}, x={self.path[i][1]}:\n'
                      f'{self.path[i][2][:, :, 0]}\nFires status:\n{self.path[i][2][:, :, 1]}\n'
                      f'Action selected: {self.path[i][3]}\n', file=ft)
            print(f'\nShortest path saved to: {args.ql_path_file}\n'
                  f'Trace saved to: {args.ql_trace_file}\n')

    # Coordinates y and x never get equal to 0:
    #    +------> x
    #    | 1    199
    #    |
    #  y V 199
    #

    # Convert x and y into a row number (state) in a Q-Table
    def state(self):
        # Length of row is 199, index starts at 1
        # length_of_row = 199
        return (self.y - 1) * 199 + self.x

    def t_get_state_vector(self):
        # Return state vector = state number and around content
        # state = [(self.y - 1.0) * 199.0 + self.x] / 100000.0  # this is x, y converted to state number
        state = ((self.y - 1.0) * 199.0 + self.x) / 100000.0  # this is x, y converted to state number
        t_state_vector = torch.from_numpy(np.append(state, self.around.reshape(18)).reshape(1, 19)).float().to(
            self.device)
        t_combined = torch.cat((self.t_last_state_vector, t_state_vector), 1)
        self.t_last_state_vector = t_state_vector.clone()
        # return t_state_vector  # tensor vector of 19 elements

        ###### print(f'\n### State vector t_combined: {t_combined[0,0]:.5f}..,{t_combined[0,19]:.5f}..')

        return t_combined  # tensor vector of 19 + 19 elements of the past and current state.
