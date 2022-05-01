# Coordinates:
#    +------> x
#    | 1    199
#    |
#  y V 199
#

# State      - Description                                   Reward
# S=1 (1,1)  - Agent's starting point                         0
#            - Empty, subject to fire                        -0.04
#            - Empty, if fire for around[i][j][1] time units
#              can not visit                                 -0.04
#            - Wall - can not visit                          -100.0
#            - Repeated cell visit                           -0.25
# G(199,199) - Goal - game over                              +10000.0


# Including this reads command line parameters into "args"
from params import args
from read_maze import load_maze, get_local_maze_information


# from cache import Cache

class Environment:
    def __init__(self):
        # Create the environment
        self.x = 1
        self.y = 1
        self.visited_cells = set()
        self.reward = 0

        # self.maze_cache = Cache()

        self.reset()

    def reset(self):
        # reset to a start position
        self.x = 1
        self.y = 1
        self.visited_cells = set()
        self.reward = 0

        # Load maze from configuration file
        load_maze()

    def check_if_visited(self):
        # Check if we already visited this cell:
        if (self.y, self.x) in self.visited_cells:
            self.reward = args.visited_cell
        else:
            self.visited_cells.add((self.y, self.x))
            self.reward = args.adjacent_cell

    # Takes next step in the environment. Returns state corresponding to the Q-table row
    def step(self, action):
        around = get_local_maze_information(self.y, self.x)
        # around = self.maze_cache.get_cached_local_maze_information_no_fire_support(self.y, self.x)

        # We can take one of 5 actions ("0 - left", "1 - up", "2 - right", "3 - down", "4 - stay")
        # move right (or else do nothing if at the right edge or target location is not empty)
        if action == 2 and self.x < 199 and around[1][2][0] == 1:
            if around[1][2][1] == 0 or args.neglect_fire:  # There is no fire
                self.x += 1
                # Could be the goal?
                if self.x == 199 and self.y == 199:
                    self.reward = args.goal_cell
                    return (self.y - 1) * 199 + self.x, self.reward, True
                else:
                    self.check_if_visited()
            else:  # There is fire (around[1][2][1] > 0). You can not visit this location
                # Coordinates do not change
                self.reward = args.fire_cell

        # move left (or else do nothing if at the left edge or target location is not empty)
        elif action == 0 and self.x > 1 and around[1][0][0] == 1:
            if around[1][0][1] == 0 or args.neglect_fire:  # There is no fire
                self.x -= 1
                self.check_if_visited()
            else:  # There is fire (around[1][0][1] > 0). You can not visit this location
                # Coordinates do not change
                self.reward = args.fire_cell

        # move up (or else do nothing if at the top edge or target location is not empty)
        elif action == 1 and self.y > 1 and around[0][1][0] == 1:
            if around[0][1][1] == 0 or args.neglect_fire:  # There is no fire
                self.y -= 1
                self.check_if_visited()
            else:  # There is fire (around[0][1][1] > 0). You can not visit this location
                # Coordinates do not change
                self.reward = args.fire_cell

        # move down (or else do nothing if at the bottom edge or target location is not empty)
        elif action == 3 and self.y < 199 and around[2][1][0] == 1:
            if around[2][1][1] == 0 or args.neglect_fire:  # There is no fire
                self.y += 1
                # Could be the goal?
                if self.x == 199 and self.y == 199:
                    self.reward = args.goal_cell
                    return (self.y - 1) * 199 + self.x, self.reward, True
                else:
                    self.check_if_visited()
            else:  # There is fire (around[2][1][1] > 0). You can not visit this location
                # Coordinates do not change
                self.reward = args.fire_cell

        # stay (do nothing)
        elif action == 4:
            # Should get some penalty as staying affects minimum traversal time.
            self.reward = args.stayed_cell
            pass

        # hit the wall (or tried to get out of maze)
        else:
            self.check_if_visited()
            self.reward = args.wall_cell

        # Return state, reward, goal. State - is a row number in a Q-Table. state = (self.y - 1) * 199 + self.x
        return (self.y - 1) * 199 + self.x, self.reward, False

    def print_situations(self, around):
        print(f"Situation around: y={self.y}, x={self.x}:")
        print(f"{around[0][0]},{around[0][1]},{around[0][2]}")
        print(f"{around[1][0]},{around[1][1]},{around[1][2]}")
        print(f"{around[2][0]},{around[2][1]},{around[2][2]}")

    '''
    # Convert x and y into a row number (state) in a Q-Table
    def state(self):
        # Length of row is 199, index starts at 1
        # length_of_row = 199
        return (self.y - 1) * 199 + self.x
    '''
