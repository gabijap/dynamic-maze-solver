# Read configuration from command line

import argparse
import time


def read_params():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Maze solver with Q-learning')
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    best_checkpoint = '2022_05_23_22_27_18'

    # Common for QL
    parser.add_argument('--maze_size', type=int, default=199)

    parser.add_argument('--description', type=str, default=f'maze_solver_({timestamp})')
    parser.add_argument('--ql', action='store_true')  # store_true or store_false
    parser.add_argument('--dqn_linear', action='store_true')  # store_true or store_false
    parser.add_argument('--dqn_cnn', action='store_true')  # store_true or store_false
    parser.add_argument('--neglect_fire', type=int, choices=[0, 1], default=0)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--train', action='store_true')  # store_true or store_false
    parser.add_argument('--play', action='store_true')  # store_true or store_false
    parser.add_argument('--checkpoint', type=str, default=timestamp)
    parser.add_argument('--best-checkpoint', type=str, default=best_checkpoint)

    parser.add_argument('--episodes', type=int, default=3500)  # 200 2000 3000 10000
    parser.add_argument('--steps', type=int, default=50000)  # 1000 50000
    parser.add_argument('--disc_rate', type=float, default=0.99)
    parser.add_argument('--start_explor_rate', type=float, default=1.0)  # 1.0 or 0.9
    parser.add_argument('--start_explor_rate_fire', type=float, default=0.88)  # 0.42 0.3 0.43
    parser.add_argument('--end_explor_rate', type=float, default=0.01)  # 0.01 0.1 0.001
    parser.add_argument('--ql_explor_decay_rate', type=float, default=0.0004)  # 0.001 <- use this # 0.005
    parser.add_argument('--ql_lr', type=float, default=0.08)  # 0.1 <- put this back to 0.1

    # Rewards configuration (common)
    parser.add_argument('--fire_cell', type=float, default=-0.9)  # -0.25 -0.5 -0.04
    parser.add_argument('--wall_cell', type=float, default=-0.09)  # -0.375 -0.75 -1.0
    parser.add_argument('--stayed_cell', type=float, default=-0.0002)  # -0.2 -0.04 -0.65, -1.0
    parser.add_argument('--visited_cell', type=float, default=-0.0002)  # -0.125 -0.25 -0.25
    parser.add_argument('--adjacent_cell', type=float, default=-0.0001)  # -0.02 -0.04 -0.01 -0.04
    parser.add_argument('--towards_cell', type=float, default=0.0000)  # 0.001 0.0 0.05
    parser.add_argument('--goal_cell', type=float, default=1.0)  # +1.0

    # DQN configuration
    parser.add_argument('--explor_decay_last_frame', type=float, default=150000)  # Best decay rate without fire: 150000
    parser.add_argument('--frames', type=int, default=4)  # 2 or 4
    parser.add_argument('--batch_size', type=int, default=32)  # 32 256
    parser.add_argument('--gamma', type=float, default=0.999)  # 0.99
    parser.add_argument('--target_update', type=int, default=500)  # 1000 is better than 3000.
    parser.add_argument('--memory_size', type=int, default=20000)  # # 1000 50000 10000 5000 256 50000 1000000
    parser.add_argument('--replay_start', type=float, default=1000)  # 1000
    parser.add_argument('--dqn_lr', type=float, default=0.0001)  # 0.001

    # Training finish conditions
    # Condition to stop training if this mean reward is reached
    parser.add_argument('--mean_reward_bound', type=float, default=-22.0)  # -22.0
    # max steps before agent is considered lost
    parser.add_argument('--max_steps', type=int, default=250000)

    # Other (less important)
    parser.add_argument('--actions_available', type=int, default=5)
    parser.add_argument('--params_file', type=str, default=f'./checkpoint/{timestamp}_params.json')
    parser.add_argument('--ql_best_walls_model_file', type=str,
                        default='./checkpoint/2022_05_23_22_03_21_ql_qtable_walls.npy')
    parser.add_argument('--ql_best_fires_model_file', type=str,
                        default='./checkpoint/2022_05_23_22_27_18_ql_qtable_fires.npy')
    parser.add_argument('--ql_walls_model_file', type=str, default=f'./checkpoint/{timestamp}_ql_qtable_walls.npy')
    parser.add_argument('--ql_fires_model_file', type=str, default=f'./checkpoint/{timestamp}_ql_qtable_fires.npy')

    parser.add_argument('--ql_model_csv_file', type=str, default=f'./checkpoint/{timestamp}_ql_qtable.csv')
    parser.add_argument('--ql_output_file', type=str, default=f'./checkpoint/{best_checkpoint}_ql_output.txt')

    return parser.parse_args()


# Global variable to access command line arguments
args = read_params()
