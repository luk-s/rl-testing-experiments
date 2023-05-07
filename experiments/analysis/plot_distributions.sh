#!/bin/sh

##################################
#           CONFIG START         #
##################################

# Select which plots should be created
CREATE_FORCED_MOVES="False"
CREATE_BEST_MOVES_MIDDLEGAMES="True"
CREATE_BEST_MOVES_ENDGAMES="False"
CREATE_MIRROR_BOARD="False"

# Some general options
SAVE_RESULT_CSV="True"
SAVE_PLOT="True"
SHOW_PLOT="False"

# Define the names of the result files
FILE_FORCED_MOVES="experiments/results/final_data/parent_child_results_ENGINE_local_400_nodes_DATA_final_forced_moves_master_fen.txt"
# FILE_BEST_MOVES_MIDDLEGAMES="experiments/results/final_data/parent_child_results_ENGINE_local_400_nodes_DATA_final_middlegame_master_dense_fen.txt"
FILE_BEST_MOVES_MIDDLEGAMES="experiments/results/parent_child_testing/results_ENGINE_local_400_nodes_DATA_parent_child_experiment_middlegames_most_important_fens_2023_05_07_17:11:38.txt"
FILE_BEST_MOVES_ENDGAMES="experiments/results/final_data/parent_child_results_ENGINE_local_400_nodes_DATA_final_endgame_master_dense_fen.txt"
FILE_MIRROR_BOARD="experiments/results/final_data/mirror_results_ENGINE_local_400_nodes_DATA_final_middlegame_master_dense_fen.txt"

# Define the titles of the plots
TITLE_FORCED_MOVES="Forced moves"
TITLE_BEST_MOVES_MIDDLEGAMES="Best moves (middlegames)"
TITLE_BEST_MOVES_ENDGAMES="Best moves (endgames)"
TITLE_MIRROR_BOARD="Mirror board"

##  Define the groups of experiments which use the same arguments
# Arguments for all experiments
COMMON_ARGS="--x_limit_min 0 --x_limit_max 2"

# Arguments for parent-child-type experiments
PARENT_CHILD_EXPERIMENT_ARGS="--column_name1 parent_score --column_name2 child_score --q_vals_to_flip child_score"

# Arguments for mirror-type experiments
MIRROR_EXPERIMENT_ARGS="--column_name1 original --column_name2 mirror"

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

if [ $CREATE_BEST_MOVES_ENDGAMES == "True" ]; then
    echo "====================="
    echo "Best moves (endgames)"
    echo "====================="
    python3 experiments/analysis/plot_distribution.py --result_path $FILE_BEST_MOVES_ENDGAMES --title $TITLE_BEST_MOVES_ENDGAMES $COMMON_ARGS $PARENT_CHILD_EXPERIMENT_ARGS
fi

if [ $CREATE_MIRROR_BOARD == "True" ]; then
    echo "============"
    echo "Mirror board"
    echo "============"
    python3 experiments/analysis/plot_distribution.py --result_path $FILE_MIRROR_BOARD --title $TITLE_MIRROR_BOARD $COMMON_ARGS $MIRROR_EXPERIMENT_ARGS
fi
