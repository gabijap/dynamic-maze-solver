# Makefile for managing the project
# To see available commands, run "make" or "make help"

help:
	# This shows available parameters
	python main.py --help

train:
	# This runs the training of maze walls without considering fires
	python main.py --ql --train

play:
	# This runs the play with the trained model, also outputs the output file with path and trace
	python main.py --ql --play --ql_best_fires_model_file=./checkpoint/2022_05_23_22_27_18_ql_qtable_fires.npy

output:
	# This prints the path with the minimum traversal time from the top left corner (1, 1), to the bottom
	# right corner (199, 199).
	more ./checkpoint/2022_05_23_22_27_18_ql_output.txt

restore:
	# This restores the pre-trained models from the backup
	cp ./final/2022_05_23_22_03_21_ql_qtable_walls.npy ./checkpoint/
	cp ./final/2022_05_23_22_27_18_ql_qtable_fires.npy ./checkpoint/
	cp ./final/2022_05_23_22_27_18_ql_output.txt ./checkpoint/
