# Including this reads command line parameters into "args"
from pprint import pprint

# from dqn import train
from params import args
from qlearning import Qlearning

if __name__ == "__main__":

    print('\nCurrent configuration:\n')
    pprint(vars(args), sort_dicts=False)

    if args.ql:

        # Solve by Q-learning method
        ql = Qlearning()

        if args.train:
            # Train once without fire to memorize walls (first-phase of training)
            ql.train(1, 5500, args.steps, args.start_explor_rate)

            # Save Q-table for further use
            # ql.save_model(args.ql_walls_model_file)

            # Load model trained without fires
            # ql.load_model(args.ql_best_walls_model_file)

            # Copy walls, so that we do not hit at least walls during fires 
            # ql.copy_walls()

            # Train with fires (second-phase of training)
            ql.train(0, 9500, args.steps, args.start_explor_rate_fire)  # THIS WAS OK: 5500

            ql.save_model(args.ql_fires_model_file)

            ql.play()

        if args.play:
            ql.load_model(args.ql_best_fires_model_file)
            ql.play()
