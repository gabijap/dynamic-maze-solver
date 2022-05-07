# Read configuration from command line parameters

import argparse
import time


def read_params():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Maze solver with Q-learning')
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    best_checkpoint = '2022_05_06_15_45_09'

    # Common for QL and DQN
    parser.add_argument('--description', type=str, default=f'Maze solver test ({timestamp})')
    parser.add_argument('--method', type=str, choices=['ql', 'dqn'], default='ql')  # 'dqn', 'ql'
    parser.add_argument('--neglect_fire', type=int, choices=[0, 1], default=0)  # 0 or 1
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')  # 'cpu' or 'cuda'
    parser.add_argument('--train', type=int, choices=[0, 1], default=0)  # 0 or 1
    parser.add_argument('--play', type=int, choices=[0, 1], default=1)  # 0 or 1
    parser.add_argument('--checkpoint', type=str, default=timestamp)
    parser.add_argument('--best-checkpoint', type=str, default=best_checkpoint)

    parser.add_argument('--episodes', type=int, default=2000)  # 3000, 10000
    parser.add_argument('--steps', type=int, default=50000)  # 50000
    parser.add_argument('--disc_rate', type=float, default=0.99)
    parser.add_argument('--start_explor_rate', type=float, default=1.0)  # 0.9
    parser.add_argument('--end_explor_rate', type=float, default=0.01)  # 0.1 0.001
    parser.add_argument('--explor_decay_rate', type=float, default=0.005)  # 0.005  # 0.001
    # 0.0001 - ql took 1200 episodes x 50,000 steps to find goal - too slow => could be increased

    # Rewards configuration (common)
    # Reward for moving to adjacent cell
    parser.add_argument('--adjacent_cell', type=float, default=-0.04)  # -0.04
    # Reward for moving to goal cell (maximum reward)
    parser.add_argument('--goal_cell', type=float, default=+1.0)  # +1.0
    # Reward for moving to a blocked ("fire") cell
    # Attempt to move to a blocked cell is invalid and will not be executed, but penalty reward will apply
    parser.add_argument('--fire_cell', type=float, default=-0.04)  # -0.04
    # Reward for moving outside maze boundaries, or on walls
    parser.add_argument('--wall_cell', type=float, default=-1.0)  # -1.0
    # Reward for moving to a cell which was already visited (discourage wondering)
    parser.add_argument('--visited_cell', type=float, default=-0.25)  # -0.25
    # Reward for staying on a cell (encourage moving)
    # Set to -100.0 when fire disabled, -0.1 otherwise
    parser.add_argument('--stayed_cell', type=float, default=-1.0)  # -1.0

    # QL configuration
    parser.add_argument('--ql_lr', type=float, default=0.1)

    # DQN configuration
    parser.add_argument('--batch_size', type=int, default=256)  # 256
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--target_update', type=int, default=5)  # TODO: Original = 10
    parser.add_argument('--memory_size', type=int, default=50000)  # 256  # 50000 1000000
    parser.add_argument('--dqn_lr', type=float, default=0.001)  # 0.001

    # Other (less important)
    parser.add_argument('--actions_available', type=int, default=5)
    parser.add_argument('--params_file', type=str, default=f'./checkpoint/{timestamp}_params.json')
    parser.add_argument('--ql_best_model_file', type=str, default=f'./checkpoint/{best_checkpoint}_ql_qtable.npy')
    parser.add_argument('--ql_model_file', type=str, default=f'./checkpoint/{timestamp}_ql_qtable.npy')
    parser.add_argument('--ql_model_csv_file', type=str, default=f'./checkpoint/{timestamp}_ql_qtable.csv')
    parser.add_argument('--ql_path_file', type=str, default=f'./checkpoint/{best_checkpoint}_ql_path.csv')
    parser.add_argument('--ql_trace_file', type=str, default=f'./checkpoint/{best_checkpoint}_ql_trace.txt')

    return parser.parse_args()


# Global variable to access command line arguments
args = read_params()
