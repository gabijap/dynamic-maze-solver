import numpy as np

from cache import Cache
from params import args
from read_maze import load_maze, get_local_maze_information


class Environment:
    def __init__(self):
        self.x = 1
        self.y = 1
        self.around = np.zeros((3, 3, 2), dtype=int)
        self.reward = 0
        self.fire_code = 0  # decimal number from 0..15, encoding 1 of 4 fire locations (left, up, right, down): '1111'
        self.visited_valid_cells = set()

        # This should be loaded only once
        load_maze()

        if args.train:
            self.maze_cache = Cache()

        # Interesting stats
        self.total_visited_cells = set()  # Larger is better during training
        self.revisited = 0  # Less is better after training
        self.no_revisits = False  # Checks if model is ready
        self.steps = 0
        self.fires = 0
        self.walls = 0
        self.path = []
        self.trace = []
        self.shortest = np.full(2, 999999999)

        self.reset(0)

    def reset(self, neglect_fire):
        self.x = 1
        self.y = 1
        self.reward = 0
        self.fire_code = 0
        self.visited_valid_cells = set()

        # Interesting stats
        self.total_visited_cells.update(self.visited_valid_cells)
        self.revisited = 0
        self.no_revisits = False
        self.steps = 0
        self.fires = 0
        self.walls = 0
        self.path = []
        self.trace = []

        if neglect_fire:
            # Use cached maze information without fires
            self.around = self.maze_cache.get_cached_local_maze_information_no_fire_support(1, 1)
        else:
            # Use maze information with fires
            # load_maze()
            self.around = get_local_maze_information(1, 1)

        return self.state(neglect_fire)

    # Check if cell is already visited before
    def check_if_visited(self):
        if (self.y, self.x) in self.visited_valid_cells:
            self.reward = args.visited_cell
            self.revisited += 1
        else:
            self.visited_valid_cells.add((self.y, self.x))
            self.reward = args.adjacent_cell

    # Takes next step in the environment. Returns state corresponding to the Q-table row
    def step(self, action, neglect_fire):

        # record path (including trace)
        if args.play:
            self.path.append((self.y, self.x, self.around, action))
        self.steps += 1

        # We can take one of 5 actions ("0 - left", "1 - up", "2 - right", "3 - down", "4 - stay")
        # move right (or else do nothing if at the right edge or target location is not empty)
        if action == 2:
            if self.around[1][2][0]:
                if self.around[1][2][1] == 0 or neglect_fire:  # There is no fire
                    self.x += 1
                    if self.x == 199 and self.y == 199:  # goal?
                        self.reward = args.goal_cell
                        return self.results(True, neglect_fire)
                    else:
                        self.check_if_visited()
                else:  # hit fire (around[1][2][1] > 0). You can not visit this location. Coordinates do not change
                    self.fires += 1
                    self.reward = args.fire_cell
            # hit the wall
            else:
                self.walls += 1
                self.reward = args.wall_cell

        # move left (or else do nothing if at the left edge or target location is not empty)
        elif action == 0:
            if self.around[1][0][0]:
                if self.around[1][0][1] == 0 or neglect_fire:  # There is no fire
                    self.x -= 1
                    self.check_if_visited()
                else:  # hit fire (around[1][0][1] > 0). You can not visit this location
                    # Coordinates do not change
                    self.fires += 1
                    self.reward = args.fire_cell
            # hit the wall
            else:
                self.walls += 1
                self.reward = args.wall_cell

        # move up (or else do nothing if at the top edge or target location is not empty)
        elif action == 1:
            if self.around[0][1][0]:
                if self.around[0][1][1] == 0 or neglect_fire:  # There is no fire
                    self.y -= 1
                    self.check_if_visited()
                else:  # hit fire (around[0][1][1] > 0). You can not visit this location. Coordinates do not change
                    self.fires += 1
                    self.reward = args.fire_cell
            # hit the wall
            else:
                self.walls += 1
                self.reward = args.wall_cell

        # move down (or else do nothing if at the bottom edge or target location is not empty)
        elif action == 3:
            if self.around[2][1][0]:
                if self.around[2][1][1] == 0 or neglect_fire:  # There is no fire
                    self.y += 1
                    if self.x == 199 and self.y == 199:  # goal?
                        self.reward = args.goal_cell
                        return self.results(True, neglect_fire)
                    else:
                        self.check_if_visited()
                else:  # hit fire (around[2][1][1] > 0). You can not visit this location. Coordinates do not change
                    self.fires += 1
                    self.reward = args.fire_cell
            # hit the wall
            else:
                self.walls += 1
                self.reward = args.wall_cell

        # stay (do nothing)
        elif action == 4:
            # Should get penalty as staying affects minimum traversal time
            self.reward = args.stayed_cell
            pass

        return self.results(False, neglect_fire)

    def results(self, goal, neglect_fire):
        # If goal reached, print the shortest path
        # if goal and not neglect_fire:  # TODO: PUT HIS BACK
        if goal:
            if self.shortest[neglect_fire] > self.steps:
                print(f'\nShortest path updated: {self.shortest[neglect_fire]}->{self.steps} steps\n')
                self.shortest[neglect_fire] = self.steps
                if args.play and not args.train:
                    self.save_path_and_trace()

        if neglect_fire:
            # Use cached maze information without fires
            self.around = self.maze_cache.get_cached_local_maze_information_no_fire_support(self.y, self.x)
        else:
            # Use maze information with fires
            self.around = get_local_maze_information(self.y, self.x)

        # Return state, reward, goal. State - is a row number in a Q-Table. state = (self.y - 1) * 201 + self.x
        return self.state(neglect_fire), self.reward, goal

    def get_visited(self):
        return len(self.visited_valid_cells)

    def get_total_visited(self):
        return len(self.total_visited_cells)

    def save_path_and_trace(self):
        with open(args.ql_output_file, 'w') as f:
            # Print the path with the minimum traversal time from the top left corner (1, 1), to the bottom right
            # corner (199, 199).
            print(f'\nShortest path length: {len(self.path)}, revisited: {self.revisited}, fires: {self.fires}\n'
                  f'step, y, x, action (0:left, 1:up, 2:right, 3:down, 4:stay)', file=f)

            # Print the trace in the environment, at each time unit, what was observed in surroundings, and what
            # action was taken
            for i in range(len(self.path)):
                print(f'{i + 1}, {self.path[i][0]}, {self.path[i][1]}, {self.path[i][3]}', file=f)

            for i in range(len(self.path)):
                print(f'\nStep {i}. Situation around y={self.path[i][0]}, x={self.path[i][1]}:\n'
                      f'{self.path[i][2][:, :, 0]}\nFires status:\n{self.path[i][2][:, :, 1]}\n'
                      f'Action selected: {self.path[i][3]}\n', file=f)

            print(f'\nShortest path and trace saved to: {args.ql_output_file}\n')

    # Coordinates y and x never get equal to 0:
    #    +------> x
    #    | 1    199
    #    |
    #  y V 199
    #

    # Convert x and y into a row number (state) in a Q-Table
    def state_simple(self):
        # length_of_row = 201, index starts at 0
        return self.y * 201 + self.x

    # Convert y, x, fire_code into a row number (state) in a Q-Table
    def state(self, neglect_fire):
        # length_of_row = 201, index starts at 0, there are [0..15] fire states ('1111'), including no fire '0000'
        if neglect_fire:
            # Fire will not be taken into account
            return self.y * 201 + self.x
        else:
            self.around[self.around > 1] = 1  # drop time steps to 1
            # Encodes fire location according to digital bits: binary 1111 (left, up, right, down),
            # to decimal numbers: 1..15
            self.fire_code = self.around[1][0][1] + self.around[0][1][1] * 2 + self.around[1][2][1] * 4 + \
                             self.around[2][1][1] * 8

            # state = self.y * 201 + self.x  # [0..40400 (201*201)]
            # offset = state * int(self.fire_code)
            # return offset + state
            # return state * int(self.fire_code) + state
            return (self.y * 201 + self.x) * (self.fire_code + 1)
