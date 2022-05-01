# Read configuration from command line parameters

import argparse


def read_params():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Maze solver with Q-learning")

    # Configuration parameters for Q-learning algorithm
    parser.add_argument("--episodes", type=int, default=3000)  # 3000, 10000
    parser.add_argument("--steps", type=int, default=50000)  # 50000
    parser.add_argument("--lr", type=int, default=0.1)
    parser.add_argument("--disc_rate", type=float, default=0.99)
    parser.add_argument("--explor_rate", type=float, default=1.0)
    parser.add_argument("--start_explor_rate", type=float, default=1.0)
    parser.add_argument("--end_explor_rate", type=float, default=0.0)  # 0.001
    parser.add_argument("--explor_decay_rate", type=float, default=0.005)  # 0.005  # 0.001

    # Rewards configuration
    # Reward for moving to adjacent cell
    parser.add_argument("--adjacent_cell", type=float, default=-0.04)
    # Reward for moving to goal cell (maximum reward)
    parser.add_argument("--goal_cell", type=float, default=10000.0)
    # Reward for moving to a bloacked ("fire") cell
    # Attempt to move to a blocked cell is invalid and will not be executed, but penalty reward will apply
    parser.add_argument("--fire_cell", type=float, default=-0.04)  # -0.75
    # Reward for moving outside maze boundaries, or on walls
    parser.add_argument("--wall_cell", type=float, default=-100.0)  # -0.8
    # Reward for moving to a cell which was already visited (discourage wondering)
    parser.add_argument("--visited_cell", type=float, default=-0.25)
    # Reward for staying in a cell (encourage moving)
    # Set to -100.0 when fire disabled, -0.1 otherwise
    parser.add_argument("--stayed_cell", type=float, default=-1.0)

    # This disables fire
    parser.add_argument("--neglect_fire", default=False)

    return parser.parse_args()


# Define global variable to access command line arguments
args = read_params()
