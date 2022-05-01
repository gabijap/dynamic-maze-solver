# Including this reads command line parameters into "args"
from params import args

from qlearning import Qlearning

if __name__ == "__main__":

    print("Starting")
    print(args)

    ql = Qlearning()
    ql.train()
    # ql.play()

    print("Completed")
    print(args)
