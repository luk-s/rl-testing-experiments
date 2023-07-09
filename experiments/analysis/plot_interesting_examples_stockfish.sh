#!/bin/sh

##################################
#           CONFIG START         #
##################################

# Select which plots should be created
CREATE_FORCED_MOVES="True"
CREATE_BEST_MOVES_MIDDLEGAMES="True"
CREATE_MIRROR_BOARD="True"
CREATE_TRANSFORM_BOARD="True"

# Some general options
STOCKFISH_VERSION="CLASSIC"
SAVE_PLOT="True"
SHOW_PLOT="False"
LARGE_FONT="True"

if [ $STOCKFISH_VERSION == "NNUE" ]; then
    PREFIX="nnue"
    NUM_NODES=81000
    NUM_BOARDS_MAIN="400k"
    STOCKFISH_CONFIG="stockfish_remote_"$NUM_NODES"_nodes.ini"
    RESULT_DIR="stockfish_nnue_results"
else
    PREFIX="classic"
    NUM_NODES=1400000
    NUM_BOARDS_MAIN="200k"
    STOCKFISH_CONFIG="stockfish_classic_remote_"$NUM_NODES"_nodes.ini"
    RESULT_DIR="stockfish_classic_results"
fi

echo "Using stockfish config: $STOCKFISH_CONFIG"

# Define the names of the result files
FILE_FORCED_MOVES="experiments/results/final_data/$RESULT_DIR/stockfish_"$PREFIX"_forced_move_"$NUM_NODES"_nodes_"$NUM_BOARDS_MAIN"_forced_move_positions_q_scores_differences_sorted.csv"
FILE_BEST_MOVES_MIDDLEGAMES="experiments/results/final_data/$RESULT_DIR/stockfish_"$PREFIX"_recommended_move_"$NUM_NODES"_nodes_"$NUM_BOARDS_MAIN"_middlegame_positions_q_scores_differences_sorted.csv"
FILE_MIRROR_BOARD="experiments/results/final_data/$RESULT_DIR/stockfish_"$PREFIX"_mirror_position_"$NUM_NODES"_nodes_"$NUM_BOARDS_MAIN"_middlegame_positions_q_scores_differences_sorted.csv"
FILE_TRANSFORM_BOARD="experiments/results/final_data/$RESULT_DIR/stockfish_"$PREFIX"_board_transformation_"$NUM_NODES"_nodes_200k_no_pawns_synthetic_positions_q_scores_differences_sorted.csv"

# Define the base names of the resulting images
BASE_NAME_FORCED_MOVES="experiments/results/final_data/"$RESULT_DIR"/plots/forced_moves"
BASE_NAME_BEST_MOVES_MIDDLEGAMES="experiments/results/final_data/"$RESULT_DIR"/plots/best_moves_middlegame"
BASE_NAME_MIRROR_BOARD="experiments/results/final_data/"$RESULT_DIR"/plots/mirror"
BASE_NAME_TRANSFORM_BOARD="experiments/results/final_data/"$RESULT_DIR"/plots/transform"

##  Define the groups of experiments which use the same arguments
# Arguments for all experiments
COMMON_ARGS="--num_examples 20  --engine_config_name $STOCKFISH_CONFIG"

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