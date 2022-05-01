# Q-Learning

Implementation of the basic Q-Learning algorighm on dynamic maze. Exact sample maze (with fires) is provided below:

![image](https://user-images.githubusercontent.com/25717176/165905402-b84a6632-d155-46c4-9651-6e55137fc592.png)


## To Do

- [x] improve performance through loading all maze in advance
- [x] tune basic parameters (found least steps required at: max_steps_per_episode = 50,000, exploration_decay_rate = 0.005)
- [ ] support handling fires
- [ ] rewrite Q-table to NN

## Questions

- [ ] is calling get_local_maze_information(y, x) before each step is required to find any new fires, or calling just once an caching is fine?

## How to run

```
python main.py
```

## Major files

- `main.py` - main
- `environment.py` - provides basic environment functions such as reset(), step(), reward() and is_done()
- `qlearning.py` - provides implementation of the basic q-learning algorithm and learn() and play() functions
- `maze20212022.npy` - definition of the maze
- `read_maze.py` - functions to read maze (could not be edited)

