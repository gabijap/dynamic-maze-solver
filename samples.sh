#!/bin/bash


case "$1" in

    1)
        echo "Sample without fires"
        # Sample without fires and with cache - very fast starts finding goal in 5 mins on M1 Mac
        python main.py --neglect_fire=1 \
           --stayed_cell=-1.0 --fire_cell=-0.5 --visited_cell=-0.25 --wall_cell=-0.75 --towards_cell=0.0 \
           --adjacent_cell=-0.04 \
           --ql_explor_decay_rate=0.005 --steps=50000
        ;;

    2)
        echo  "Sample without fires"
        # Sample without fires and with cache - very fast starts finding goal in 5 mins on M1 Mac
        # (smaller normalized stayed cell reward -0.4 )
        python main.py --neglect_fire=1 \
           --stayed_cell=-0.4 --fire_cell=-0.5 --visited_cell=-0.25 --wall_cell=-0.75 --towards_cell=0.0 \
           --adjacent_cell=-0.04 \
           --ql_explor_decay_rate=0.005 --steps=50000
        ;;

    3)
        echo  "Sample without fires"
        # Sample without fires and with cache - very fast starts finding goal in 5 mins on M1 Mac
        # (smaller normalized rewards [-1..+1])
        python main.py --neglect_fire=1 \
           --stayed_cell=-0.2 --fire_cell=-0.25 --visited_cell=-0.125 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.005 --steps=50000
        ;;



    4)
        echo "Sample with fires"
        # Sample with fires and with fast maze implementation
        python main.py --neglect_fire=1 \
           --stayed_cell=-0.2 --fire_cell=-0.25 --visited_cell=-0.125 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.005 --steps=50000
        ;;

    5)
        echo "Sample with fires"
        # Sample with fires and with fast maze implementation
        # Slower decay rate, as more time is required to learn larger space
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.2 --fire_cell=-0.25 --visited_cell=-0.125 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.001 --steps=50000 --episodes=2500
        ;;

    6)
        echo "Sample with fires"
        # Sample with fires and with fast maze implementation
        # Even slower decay rate, as more time is required to learn larger space
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.2 --fire_cell=-0.25 --visited_cell=-0.125 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.0005 --steps=50000 --episodes=2500
        ;;

    7)
        echo "Sample with fires - 7"
        # Sample with fires and with fast maze implementation
        # Larger negative rewards for entering fire cells, walls and stayed cells
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.25 --fire_cell=-0.65 --visited_cell=-0.125 --wall_cell=-0.75 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.001 --steps=50000 --episodes=2500
        ;;

    8)
        echo "Sample with fires - 8"
        # Sample with fires and with fast maze implementation
        # Larger negative rewards for entering fire cells, walls and stayed cells
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.25 --fire_cell=-0.75 --visited_cell=-0.125 --wall_cell=-0.75 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.001 --steps=50000 --episodes=2500
        ;;

    10)
        echo "Sample with fires - 10 - perfect convergence"  # REPEAT THIS - SHORTEST CONVERGENCE: TODO PUT BACK EPISODES TO: 3500
        # Sample with fires and with fast maze implementation
        # Slower decay rate, as more time is required to learn larger space
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.2 --fire_cell=-0.25 --visited_cell=-0.125 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.001 \
           --steps=50000 --episodes=6500 \
           --debug=1
        # python main.py --play=1 --train=0
        ;;

    11)
        echo "Sample with fires - 11 - perfect convergence"
        # Sample with fires and with fast maze implementation
        # Slower decay rate, as more time is required to learn larger space
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.1 --fire_cell=-0.2 --visited_cell=-0.05 --wall_cell=-0.3 --towards_cell=0.000 \
           --adjacent_cell=-0.01 \
           --ql_explor_decay_rate=0.001 --steps=50000 --episodes=3500
        ;;

    12)
        echo "Sample with fires - 12 - THIS DID NOT CONVERGE, REWARDS ARE TOO SMALL"
        # Sample with fires and with fast maze implementation
        # Slower decay rate, as more time is required to learn larger space
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.01 --fire_cell=-0.02 --visited_cell=-0.005 --wall_cell=-0.03 --towards_cell=0.000 \
           --adjacent_cell=-0.001 \
           --ql_explor_decay_rate=0.001 --steps=50000 --episodes=3500
        ;;

    13)
        echo "Sample with fires - 13"
        # Sample with fires and with fast maze implementation
        # Less steps
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.1 --fire_cell=-0.2 --visited_cell=-0.05 --wall_cell=-0.3 --towards_cell=0.000 \
           --adjacent_cell=-0.01 \
           --ql_explor_decay_rate=0.001 --steps=45000 --episodes=4000
        ;;

    14)
        echo "Sample with fires - 14 - perfect convergence"
        # Sample with fires and with fast maze implementation
        # Less steps
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.1 --fire_cell=-0.2 --visited_cell=-0.05 --wall_cell=-0.3 --towards_cell=0.000 \
           --adjacent_cell=-0.01 \
           --ql_explor_decay_rate=0.001 --steps=35000 --episodes=4500
        ;;

    15)
        echo "Sample with fires - 15"
        # Sample with fires and with fast maze implementation
        # larger decay rate
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.1 --fire_cell=-0.2 --visited_cell=-0.05 --wall_cell=-0.3 --towards_cell=0.000 \
           --adjacent_cell=-0.01 \
           --ql_explor_decay_rate=0.005 --steps=48000 --episodes=3300
           --description=exp015_larger_dec_rate_0.005
        ;;

    16)
        echo "Sample with fires - 16"
        # Sample with fires and with fast maze implementation
        # larger decay rate
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.1 --fire_cell=-0.2 --visited_cell=-0.05 --wall_cell=-0.3 --towards_cell=0.000 \
           --adjacent_cell=-0.01 \
           --ql_explor_decay_rate=0.009 --steps=48000 --episodes=3300
           --description=exp016_larger_dec_rate_0.009
        ;;


    17)
        echo "Sample with fires - 17 - perfect convergence"
        # Sample with fires and with fast maze implementation
        # smoother decay rate
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.1 --fire_cell=-0.2 --visited_cell=-0.05 --wall_cell=-0.3 --towards_cell=0.000 \
           --adjacent_cell=-0.01 --end_explor_rate=0.0 \
           --ql_explor_decay_rate=0.002 --steps=48000 --episodes=3300
           --description=exp017_smoother_dec_rate_0.002
        ;;

    18)
        echo "Sample with fires - 18 - perfect convergence"
        # Sample with fires and with fast maze implementation
        # smoother decay rate
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.1 --fire_cell=-0.2 --visited_cell=-0.05 --wall_cell=-0.3 --towards_cell=0.000 \
           --adjacent_cell=-0.01 --end_explor_rate=0.0 \
           --ql_explor_decay_rate=0.0015 --steps=48000 --episodes=3300 \
           --description=exp017_smoother_dec_rate_0.0015
        ;;

    19)
        echo "Sample with fires - 19 - perfect convergence"  # REPEAT THIS - SHORTEST CONVERGENCE
        # Sample with fires and with fast maze implementation
        # Larger rewards, also positive towards reward
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.19 --fire_cell=-0.23 --visited_cell=-0.123 --wall_cell=-0.373 --towards_cell=0.003 \
           --adjacent_cell=-0.02 --end_explor_rate=0.0 \
           --ql_explor_decay_rate=0.0011 --steps=48000 --episodes=3300 \
           --description=exp019_towards_cell_0_003
        # python main.py --play=1 --train=0
        ;;

    20)
        echo "Sample with fires - 20 - perfect convergence"  # REPEAT THIS - SHORTEST CONVERGENCE
        # Sample with fires and with fast maze implementation
        # Larger towards reward
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.19 --fire_cell=-0.23 --visited_cell=-0.123 --wall_cell=-0.373 --towards_cell=0.007 \
           --adjacent_cell=-0.02 --end_explor_rate=0.0 \
           --ql_explor_decay_rate=0.0011 --steps=48000 --episodes=3300 \
           --description=exp020_towards_cell_0_007
        # python main.py --play=1 --train=0
        ;;

    23)
        echo "Sample with fires - 23 - perfect convergence"  # REPEAT THIS - SHORTEST CONVERGENCE
        # Sample with fires and with fast maze implementation
        # Even slower decay rate
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.2 --fire_cell=-0.25 --visited_cell=-0.125 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 --end_explor_rate=0.0 \
           --ql_explor_decay_rate=0.0008 --steps=48000 --episodes=3300 \
           --description=exp023_slower_dec_0.0008
        # python main.py --play=1 --train=0
        ;;

    24)
        echo "Sample with fires - 24 - perfect convergence"  # REPEAT THIS - SHORTEST CONVERGENCE
        # Sample with fires and with fast maze implementation
        # Even slower decay rate
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.2 --fire_cell=-0.25 --visited_cell=-0.125 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 --end_explor_rate=0.0 \
           --ql_explor_decay_rate=0.0006 --steps=48000 --episodes=3300 \
           --description=exp024_slower_dec_0.0006
        # python main.py --play=1 --train=0
        ;;

    25)
        echo "Sample with fires - 25 - perfect convergence"  # REPEAT THIS - SHORTEST CONVERGENCE
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.2 --fire_cell=-0.25 --visited_cell=-0.125 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.001 \
           --steps=50000 --episodes=2000 \
           --description=exp025_episodes_2000_eof \
           --debug=1
        ;;


    26)
        echo "Sample with fires - 26 - perfect convergence"  # REPEAT THIS - SHORTEST CONVERGENCE
        python main.py --neglect_fire=0 \
           --stayed_cell=-0.04 --fire_cell=-0.25 --visited_cell=-0.1 --wall_cell=-0.375 --towards_cell=0.001 \
           --adjacent_cell=-0.02 \
           --ql_explor_decay_rate=0.001 \
           --steps=50000 --episodes=2000 \
           --description=exp026_stayed_0.04_eof \
           --debug=1
        ;;

# -------- START OVER -----

    30)
        echo "Sample without fires - 30 - perfect convergence on 1.5 minutes"
        python main.py --neglect_fire=1 \
           --fire_cell=-0.75 --wall_cell=-0.75 \
           --stayed_cell=-0.25  --visited_cell=-0.25 \
           --adjacent_cell=-0.04 --towards_cell=0.001 \
           --ql_explor_decay_rate=0.001 \
           --steps=50000 --episodes=3000 \
           --description=Exp-030 \
           --debug=1
        ;;

    31)
        echo "Sample without fires - 31 - perfect convergence on 1.5 minutes"  # REPEAT THIS
        python main.py --neglect_fire=1 \
           --fire_cell=-0.3 --wall_cell=-0.3 \
           --stayed_cell=-0.1  --visited_cell=-0.1 \
           --adjacent_cell=-0.01 --towards_cell=0.001 \
           --ql_explor_decay_rate=0.001 \
           --steps=50000 --episodes=1500 \
           --description=Exp-031 \
           --debug=1
        ;;

    32)
        echo "Sample without fires - 32 - perfect convergence on 1.5 minutes"  # REPEAT THIS
        python main.py --neglect_fire=1 \
           --fire_cell=-0.7 --wall_cell=-0.7 \
           --stayed_cell=-0.1  --visited_cell=-0.1 \
           --adjacent_cell=-0.01 --towards_cell=0.001 \
           --ql_explor_decay_rate=0.001 \
           --steps=50000 --episodes=1500 \
           --description=Exp-032 \
           --debug=1
        ;;

    33)
        echo "Sample without fires - 33 - perfect convergence on 1.5 minutes"  # REPEAT THIS
        python main.py --neglect_fire=1 \
           --fire_cell=-0.3 --wall_cell=-0.3 \
           --stayed_cell=-0.1  --visited_cell=-0.1 \
           --adjacent_cell=-0.01 --towards_cell=0.001 \
           --ql_explor_decay_rate=0.001 \
           --steps=50000 --episodes=1500 \
           --description=final-without-fires-033 \
           --checkpoint=final-without-fires
        ;;

    34)
        echo "Sample without fires - 34 - perfect convergence on 1.5 minutes"  # REPEAT THIS
        python main.py --neglect_fire=1 \
           --fire_cell=-0.3 --wall_cell=-0.3 \
           --stayed_cell=-0.1  --visited_cell=-0.1 \
           --adjacent_cell=-0.01 --towards_cell=0.001 \
           --ql_explor_decay_rate=0.001 \
           --steps=50000 --episodes=1500 \
           --description=final-with-fires-034 \
           --best-checkpoint=final-without-fires \
           --checkpoint=final-with-fires
        ;;


    99)
        echo "This is test "
        ;;

    *)
        echo "Please specify valid experiment number"
        ;;

esac

