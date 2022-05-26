# Q-Learning

Implementation of the basic Q-Learning algorithm on dynamic maze. Exact sample maze (with fires) is provided below:

![image](https://user-images.githubusercontent.com/25717176/165905402-b84a6632-d155-46c4-9651-6e55137fc592.png)


## How to run

To train the Q-table run below. This might take time. Observe the training values through the values printed on the
screen.

    make train

To solve the maze using pre-trained Q-table run below, and check for the *path*, *trace* and other output files in
the `checkpoint` directory.

    make play

Print the output path and trace result file.

    make output

## Results

Results could be found in the `checkpoint` directory.

### Path

`*_ql_path.csv` - contains *path* with the minimum traversal time from the top left corner (1, 1), to the bottom right
corner (199, 199). The printout shows the step number, y and x coordinates, and the action taken (`0` - left, `1` - up, `2` - right,
`3` - down, `4` - stay).

Sample:

    Shortest path length: 4175, revisited: 0, fires: 591
    step, y, x, action (0:left, 1:up, 2:right, 3:down, 4:stay)
    1, 1, 1, 3
    2, 2, 1, 3
    3, 3, 1, 3
    ...

### Trace

`*_trace.txt` - contains *trace* in the environment, at each time unit, what was observed in surroundings, and what
action was taken. The printout shows the step number, y and x coordinates, "around" array of free cells, also
"around" array of fires, also the action taken.

Sample:

    Step 0. Situation around y=1, x=1:
    [[0 0 0]
     [0 1 1]
     [0 1 0]]
    Fires status:
    [[0 0 0]
     [0 0 2]
     [0 0 0]]
    Action selected: 3


    Step 1. Situation around y=2, x=1:
    [[0 1 1]
     [0 1 0]
     [0 1 0]]
    Fires status:
    [[0 0 1]
     [0 0 0]
     [0 0 0]]
    Action selected: 3

    ...

### Training

There is number of values tracked during the training to monitor the training process:

- `goal` - whether the goal is reached
- `episode` - episode number
- `step` - step number within the episode. Once the goal is found, this shows the number of steps taken to the goal (
  path length)
- `state` - state in the Q-table, which is states vs actions. State is defined by y and x
  coordinates: `state = (y - 1) * 199 + x`
- `rewards` - cumulative rewards for the episode or till the goal is reached. It is easy to observe, that the rewards
  are increasing, as the model gets better.
- `explr_rate` - exploratory rate (between 1 and 0). This helps to observe, that the training (exploration) is taking
  sufficiently long, so that Q-table is updates, but no longer than needed.
- `non_zero` - number of non zero (updated) elements in Q-table. This shows, how the training is progressing, as more
  and more cells get visited and the states get updated with the Q-values
- `revisited` - number of revisited cells. This shows the quality of the model, as there is no need to revisit the same
  cells. With the training, this value gets smaller, till it reaches zero, which is also corresponding to the shortest
  path to the goal.
- `fires` - number of encountered fires (time steps, that agent could not move in the desired direction, because of the
  fire). As this value fluctuates randomly, thus the length of the path also changes. Switching off fires could be
  controlled though the `--neglect_fire=1` parameter, in which case the `steps` would show the exact number of time
  steps to the goal.

## Files tree

Below is the brief summary of the directory tree.

    ./
    ├── README.md
    ├── buffer.py                                       -> Buffer (reply memory) implementation for DQN training
    ├── cache.py                                        -> cache buffer for faster access to maze (without fires) 
    ├── checkpoint                                      -> Various result files
    │   ├── 2022_05_06_15_45_09_params.json             -> Result parameters file
    │   ├── 2022_05_06_15_45_09_ql_path.csv             -> Result path file
    │   ├── 2022_05_06_15_45_09_ql_qtable.csv           -> Result Q-Table file in CSV format
    │   ├── 2022_05_06_15_45_09_ql_qtable.npy           -> Result Q-Table file in binary format
    │   └── 2022_05_06_15_45_09_ql_trace.txt            -> Result trace file
    ├── data                                            -> Miscellaneous illustration files
    │   ├── maze.csv                                    -> Maze in CSV format
    │   ├── qtable.csv                                  -> Q-Table in CSV format
    │   └── qtable.xlsx                                 -> Q-Table in Excel format
    ├── dqn.py                                          -> DQN implementation
    ├── environment.py                                  -> Maze environment implementation
    ├── main.py                                         -> Main entry point
    ├── COMP6247Maze20212022.npy                        -> Original maze data file
    ├── params.py                                       -> Parameters file
    ├── qlearning.py                                    -> Q-Learning implementation
    ├── read_maze.py                                    -> Original maze interface
    └── read_maze_fast.py                               -> Maze interface optimized for speed


