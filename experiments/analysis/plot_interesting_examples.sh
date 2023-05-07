#!/bin/sh

##################################
#           CONFIG START         #
##################################

# Select which plots should be created
CREATE_FORCED_MOVES="True"
CREATE_BEST_MOVES_MIDDLEGAMES="True"
CREATE_BEST_MOVES_ENDGAMES="True"
CREATE_MIRROR_BOARD="True"

# Some general options
SAVE_PLOT="True"
SHOW_PLOT="False"
LC0_CONFIG="remote_400_nodes.ini"
NETWORK_NAME="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207"

# Define the names of the result files
FILE_FORCED_MOVES="experiments/results/final_data/parent_child_results_ENGINE_local_400_nodes_DATA_final_forced_moves_master_fen_differences_sorted_double.csv"
FILE_BEST_MOVES_MIDDLEGAMES="experiments/results/final_data/parent_child_results_ENGINE_local_400_nodes_DATA_final_middlegame_master_dense_fen_differences_sorted_double.csv"
FILE_BEST_MOVES_ENDGAMES="experiments/results/final_data/parent_child_results_ENGINE_local_400_nodes_DATA_final_endgame_master_dense_fen_differences_sorted_double.csv"
FILE_MIRROR_BOARD="experiments/results/final_data/mirror_results_ENGINE_local_400_nodes_DATA_final_middlegame_master_dense_fen_differences_sorted_double.csv"

# Define the base names of the resulting images
BASE_NAME_FORCED_MOVES="experiments/results/final_data/plots/forced_moves"
BASE_NAME_BEST_MOVES_MIDDLEGAMES="experiments/results/final_data/plots/best_moves_middlegame"
BASE_NAME_BEST_MOVES_ENDGAMES="experiments/results/final_data/plots/best_moves_endgame"
BASE_NAME_MIRROR_BOARD="experiments/results/final_data/plots/mirror"

##  Define the groups of experiments which use the same arguments
# Arguments for all experiments
COMMON_ARGS="--num_examples 20 --engine_config_name $LC0_CONFIG --network_name $NETWORK_NAME"

# Arguments for parent-child-type experiments
PARENT_CHILD_EXPERIMENT_ARGS="--score_type1 best_move --score_type2 best_move --fen_column1 parent_fen --fen_column2 child_fen --flip_second_score --show_best_move_first"

# Arguments for mirror-type experiments
MIRROR_EXPERIMENT_ARGS="--score_type1 best_move --score_type2 best_move --fen_column1 fen --fen_column2 create_mirror"

##################################
#           CONFIG END           #
##################################

if [ $SAVE_PLOT == "True" ]; then
    SAVE_PLOT_STRING="--save_plot"
else
    SAVE_PLOT_STRING=""
fi

if [ $SHOW_PLOT == "True" ]; then
    SHOW_PLOT_STRING="--show_plot"
else
    SHOW_PLOT_STRING=""
fi

COMMON_ARGS="$COMMON_ARGS $SAVE_PLOT_STRING $SHOW_PLOT_STRING"

## Plot the interesting examples
if [ $CREATE_FORCED_MOVES == "True" ]; then
    echo "============"
    echo "Forced moves"
    echo "============"
    python3 experiments/analysis/plot_interesting_examples.py --result_path $FILE_FORCED_MOVES $COMMON_ARGS $PARENT_CHILD_EXPERIMENT_ARGS --save_path_base $BASE_NAME_FORCED_MOVES
fi

if [ $CREATE_BEST_MOVES_MIDDLEGAMES == "True" ]; then
    echo "======================="
    echo "Best moves (middlegame)"
    echo "======================="
    python3 experiments/analysis/plot_interesting_examples.py --result_path $FILE_BEST_MOVES_MIDDLEGAMES $COMMON_ARGS $PARENT_CHILD_EXPERIMENT_ARGS --save_path_base $BASE_NAME_BEST_MOVES_MIDDLEGAMES
fi

if [ $CREATE_BEST_MOVES_ENDGAMES == "True" ]; then
    echo "====================="
    echo "Best moves (endgames)"
    echo "====================="
    python3 experiments/analysis/plot_interesting_examples.py --result_path $FILE_BEST_MOVES_ENDGAMES $COMMON_ARGS $PARENT_CHILD_EXPERIMENT_ARGS --save_path_base $BASE_NAME_BEST_MOVES_ENDGAMES
fi

if [ $CREATE_MIRROR_BOARD == "True" ]; then
    echo "============"
    echo "Mirror board"
    echo "============"
    python3 experiments/analysis/plot_interesting_examples.py --result_path $FILE_MIRROR_BOARD $COMMON_ARGS $MIRROR_EXPERIMENT_ARGS --save_path_base $BASE_NAME_MIRROR_BOARD
fi
