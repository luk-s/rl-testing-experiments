#!/bin/sh

##################################
#           CONFIG START         #
##################################

# Select which plots should be created
CREATE_FORCED_MOVES="False"
CREATE_BEST_MOVES_MIDDLEGAMES="False"
CREATE_MIRROR_BOARD="True"
CREATE_TRANSFORM_BOARD="True"

# Some general options
SAVE_PLOT="True"
SHOW_PLOT="False"
LARGE_FONT="True"
LC0_CONFIG="leela_remote_400_nodes.ini"
NETWORK_NAME="T807785-b124efddc27559564d6464ba3d213a8279b7bd35b1cbfcf9c842ae8053721207"

# Define the names of the result files
FILE_FORCED_MOVES="experiments/results/final_data/leela_chess_zero_results2/forced_move_400_nodes_400k_forced_move_positions.csv"
FILE_BEST_MOVES_MIDDLEGAMES="experiments/results/final_data/leela_chess_zero_results2/recommended_move_400_nodes_400k_middlegame_positions.csv"
FILE_MIRROR_BOARD="experiments/results/final_data/leela_chess_zero_results2/mirror_position_400_nodes_400k_middlegame_positions.csv"
FILE_TRANSFORM_BOARD="experiments/results/final_data/leela_chess_zero_results2/board_transformation_400_nodes_200k_no_pawns_synthetic_positions.csv"

# Define the base names of the resulting images
BASE_NAME_FORCED_MOVES="experiments/results/final_data/leela_chess_zero_results2/plots/forced_moves"
BASE_NAME_BEST_MOVES_MIDDLEGAMES="experiments/results/final_data/leela_chess_zero_results2/plots/best_moves_middlegame"
BASE_NAME_MIRROR_BOARD="experiments/results/final_data/leela_chess_zero_results2/plots/mirror"
BASE_NAME_TRANSFORM_BOARD="experiments/results/final_data/leela_chess_zero_results2/plots/transform"

##  Define the groups of experiments which use the same arguments
# Arguments for all experiments
COMMON_ARGS="--num_examples 20  --plot_example --engine_config_name $LC0_CONFIG --network_name $NETWORK_NAME"

# Arguments for parent-child-type experiments
PARENT_CHILD_EXPERIMENT_ARGS=" --fen_column1 fen1 --fen_column2 fen2 --flip_second_score --show_best_move_first"

# Arguments for transform-type experiments
TRANSFORM_EXPERIMENT_ARGS="--build_fens_from_transformations --fen_column1 fen --score_column1 original"

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

if [ $LARGE_FONT == "True" ]; then
    LARGE_FONT_STRING="--large_font"
else
    LARGE_FONT_STRING=""
fi


COMMON_ARGS="$COMMON_ARGS $SAVE_PLOT_STRING $SHOW_PLOT_STRING $LARGE_FONT_STRING"

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

if [ $CREATE_MIRROR_BOARD == "True" ]; then
    echo "============"
    echo "Mirror board"
    echo "============"
    python3 experiments/analysis/plot_interesting_examples.py --result_path $FILE_MIRROR_BOARD $COMMON_ARGS $TRANSFORM_EXPERIMENT_ARGS --save_path_base $BASE_NAME_MIRROR_BOARD
fi

if [ $CREATE_TRANSFORM_BOARD == "True" ]; then
    echo "=================="
    echo "Transformed board"
    echo "=================="
    python3 experiments/analysis/plot_interesting_examples.py --result_path $FILE_TRANSFORM_BOARD $COMMON_ARGS $TRANSFORM_EXPERIMENT_ARGS --save_path_base $BASE_NAME_TRANSFORM_BOARD
fi