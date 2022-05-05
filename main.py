# Including this reads command line parameters into "args"
from pprint import pprint

from dqn import train
from params import args
from qlearning import Qlearning

if __name__ == "__main__":

    print("Starting")
    pprint(vars(args))

    if args.method == "dqn":
        # Solve by DQN method
        train()
    else:
        # Solve by Q-learning method
        ql = Qlearning()
        ql.train()
        # ql.play()

    print("Completed")
    pprint(vars(args))
