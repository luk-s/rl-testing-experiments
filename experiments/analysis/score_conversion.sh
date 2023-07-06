# RESULT_DIR="/local/home/flurilu/Master-Thesis/rl-testing-experiments/experiments/results/final_data4"
RESULT_DIR="/home/lukas/Polybox_ETH/Master-Thesis/Code/rl-testing-experiments/experiments/results/final_data4"

CONVERT_RECOMMENDED_MOVES="False"
CONVERT_FORCED_MOVES="True"
CONVERT_MIRROR_BOARD="True"
CONVERT_TRANSFORMATION_BOARD="True"

# FILE_RECOMMENDED_MOVES="results_ENGINE_stockfish_local_81000_nodes_DATA_fens_recommended_move_2023_06_18_19:13:25.txt"
# FILE_FORCED_MOVES="results_ENGINE_stockfish_local_81000_nodes_DATA_fens_forced_move_2023_06_18_19:11:19.txt"
# FILE_MIRROR_BOARD="results_ENGINE_stockfish_local_81000_nodes_DATA_fens_board_mirroring_2023_06_18_19:17:47.txt"
# FILE_TRANSFORMATION_BOARD="results_ENGINE_stockfish_local_81000_nodes_DATA_fens_board_transformations_2023_06_18_19:21:45.txt"

FILE_RECOMMENDED_MOVES=""
FILE_FORCED_MOVES="results_ENGINE_stockfish_classic_local_1400000_nodes_DATA_fens_forced_move_2023_06_26_18:59:29.txt"
FILE_MIRROR_BOARD="results_ENGINE_stockfish_classic_local_1400000_nodes_DATA_fens_board_mirroring_2023_06_26_18:59:42.txt"
FILE_TRANSFORMATION_BOARD="results_ENGINE_stockfish_classic_local_1400000_nodes_DATA_fens_board_transformations_2023_06_26_18:59:52.txt"

OPTIONS_RECOMMENDED_MOVES="--score_columns parent_score child_score --main_fen_column parent_fen --increase_ply_for_columns child_score"
OPTIONS_FORCED_MOVES="--score_columns parent_score child_score --main_fen_column parent_fen --increase_ply_for_columns child_score"
OPTIONS_MIRROR_BOARD="--score_columns original mirror --main_fen_column fen"
OPTIONS_TRANSFORMATION_BOARD="--score_columns original rot90 rot180 rot270 flip_diag flip_anti_diag flip_hor flip_vert --main_fen_column fen"

if [ $CONVERT_RECOMMENDED_MOVES == "True" ]; then
    echo "Converting recommended move boards"
    python3 experiments/analysis/convert_stockfish_scores_to_leela.py --input_path $RESULT_DIR/$FILE_RECOMMENDED_MOVES $OPTIONS_RECOMMENDED_MOVES
fi

if [ $CONVERT_FORCED_MOVES == "True" ]; then
    echo "Converting forced move boards"
    python3 experiments/analysis/convert_stockfish_scores_to_leela.py --input_path $RESULT_DIR/$FILE_FORCED_MOVES $OPTIONS_FORCED_MOVES
fi

if [ $CONVERT_MIRROR_BOARD == "True" ]; then
    echo "Converting mirrored boards"
    python3 experiments/analysis/convert_stockfish_scores_to_leela.py --input_path $RESULT_DIR/$FILE_MIRROR_BOARD $OPTIONS_MIRROR_BOARD
fi

if [ $CONVERT_TRANSFORMATION_BOARD == "True" ]; then
    echo "Converting transformed boards"
    python3 experiments/analysis/convert_stockfish_scores_to_leela.py --input_path $RESULT_DIR/$FILE_TRANSFORMATION_BOARD $OPTIONS_TRANSFORMATION_BOARD
fi