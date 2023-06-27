#!/bin/sh

##################################
#           CONFIG START         #
##################################

# Select which plots should be created
CREATE_FORCED_MOVES="False" #"True"
CREATE_BEST_MOVES_MIDDLEGAMES="False" #"True"
CREATE_MIRROR_BOARD="False" #"True"
CREATE_TRANSFORMATION_BOARD="False" #"True"
CREATE_DIFFERENTIAL_TESTING="True"

# Some general options
SAVE_RESULT_CSV="True" # "False"
SAVE_PLOT="False" #"True"
SHOW_PLOT="True" #"False"

# Define the names of the result files
FILE_FORCED_MOVES="experiments/results/final_data/forced_move_400_nodes_400k_forced_move_positions.csv"
FILE_BEST_MOVES_MIDDLEGAMES="experiments/results/final_data/recommended_move_400_nodes_400k_middlegame_positions.csv"
FILE_MIRROR_BOARD="experiments/results/final_data/mirror_position_400_nodes_400k_middlegame_positions.csv"
FILE_TRANSFORMATION_BOARD="experiments/results/final_data/board_transformation_400_nodes_200k_no_pawns_synthetic_positions.csv"
FILE_DIFFERENTIAL_TESTING="experiments/results/final_data2/differential_testing_leela_400_nodes_T80_vs_leela_285_nodes_T78.txt"

# Define the titles of the plots
TITLE_FORCED_MOVES="Forced moves"
TITLE_BEST_MOVES_MIDDLEGAMES="Recommended moves"
TITLE_MIRROR_BOARD="Mirror board"
TITLE_TRANSFORMATION_BOARD="Transform board"
TITLE_DIFFERENTIAL_TESTING="Differential testing"

##  Define the groups of experiments which use the same arguments
# Arguments for all experiments
COMMON_ARGS="--x_limit_min 0 --x_limit_max 2"

# Arguments for parent-child-type experiments
# Optionally add this parameter --q_vals_to_flip child_score if you're using data directly coming from experiments
PARENT_CHILD_EXPERIMENT_ARGS="--column_name1 score1 --column_name2 score2"

# Arguments for mirror-type experiments
MIRROR_EXPERIMENT_ARGS="--column_name1 original --column_name2 mirror"

# Arguments for transformation-type experiments
TRANSFORMATION_EXPERIMENT_ARGS="--column_names original rot90 rot180 rot270 flip_diag flip_anti_diag flip_hor flip_vert"

# Arguments for differential testing
DIFFERENTIAL_TESTING_ARGS="--column_name1 score1 --column_name2 score2"

##################################
#           CONFIG END           #
##################################

if [ $SAVE_RESULT_CSV == "True" ]; then
    SAVE_RESULT_CSV_STRING="--save_csv"
else
    SAVE_RESULT_CSV_STRING=""
fi

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

COMMON_ARGS="$COMMON_ARGS $SAVE_RESULT_CSV_STRING $SAVE_PLOT_STRING $SHOW_PLOT_STRING"

## Plot the distributions
if [ $CREATE_FORCED_MOVES == "True" ]; then
    echo "============"
    echo "Forced moves"
    echo "============"
    python3 experiments/analysis/plot_distribution.py --result_path $FILE_FORCED_MOVES --title $TITLE_FORCED_MOVES $COMMON_ARGS $PARENT_CHILD_EXPERIMENT_ARGS
fi


if [ $CREATE_BEST_MOVES_MIDDLEGAMES == "True" ]; then
    echo "======================="
    echo "Best moves (middlegame)"
    echo "======================="
    python3 experiments/analysis/plot_distribution.py --result_path $FILE_BEST_MOVES_MIDDLEGAMES --title $TITLE_BEST_MOVES_MIDDLEGAMES $COMMON_ARGS $PARENT_CHILD_EXPERIMENT_ARGS
fi

if [ $CREATE_MIRROR_BOARD == "True" ]; then
    echo "============"
    echo "Mirror board"
    echo "============"
    python3 experiments/analysis/plot_distribution.py --result_path $FILE_MIRROR_BOARD --title $TITLE_MIRROR_BOARD $COMMON_ARGS $MIRROR_EXPERIMENT_ARGS
fi

if [ $CREATE_TRANSFORMATION_BOARD == "True" ]; then
    echo "====================="
    echo "Transformation board"
    echo "====================="
    python3 experiments/analysis/plot_distribution.py --result_path $FILE_TRANSFORMATION_BOARD --title $TITLE_TRANSFORMATION_BOARD $COMMON_ARGS $TRANSFORMATION_EXPERIMENT_ARGS
fi

if [ $CREATE_DIFFERENTIAL_TESTING == "True" ]; then
    echo "====================="
    echo "Differential testing"
    echo "====================="
    python3 experiments/analysis/plot_distribution.py --result_path $FILE_DIFFERENTIAL_TESTING --title $TITLE_DIFFERENTIAL_TESTING $COMMON_ARGS $DIFFERENTIAL_TESTING_ARGS
fi