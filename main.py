# Including this reads command line parameters into "args"
from pprint import pprint

from dqn import train
from params import args
from qlearning import Qlearning

if __name__ == "__main__":

    print('\nCurrent configuration:\n')
    pprint(vars(args), sort_dicts=False)

    if args.method == "ql":
        # Solve by Q-learning method
        ql = Qlearning()

        if args.train:
            ql.train()
        if args.play:
            ql.play()

    else:
        # Solve by DQN method
        if args.train:
            train()
        if args.play:
            print('TBD')