# RESULT_DIR="/local/home/flurilu/Master-Thesis/rl-testing-experiments/experiments/results/final_data"
RESULT_DIR="/home/lukas/Polybox_ETH/Master-Thesis/Code/rl-testing-experiments/experiments/results/final_data"

CONVERT_RECOMMENDED_MOVES="True"
CONVERT_FORCED_MOVES="True"
CONVERT_MIRROR_BOARD="True"
CONVERT_TRANSFORMATION_BOARD="True"

# Stockfish NNUE files
# FILE_RECOMMENDED_MOVES="stockfish_nnue_results/stockfish_nnue_recommended_move_81000_nodes_400k_middlegame_positions.txt"
# FILE_FORCED_MOVES="stockfish_nnue_results/stockfish_nnue_forced_move_81000_nodes_400k_forced_move_positions.txt"
# FILE_MIRROR_BOARD="stockfish_nnue_results/stockfish_nnue_mirror_position_81000_nodes_400k_middlegame_positions.txt"
# FILE_TRANSFORMATION_BOARD="stockfish_nnue_results/stockfish_nnue_board_transformation_81000_nodes_200k_no_pawns_synthetic_positions.txt"

# Classic Stockfish files
# FILE_RECOMMENDED_MOVES="stockfish_classic_results/stockfish_classic_recommended_move_1400000_nodes_200k_middlegame_positions.txt"
# FILE_FORCED_MOVES="stockfish_classic_results/stockfish_classic_forced_move_1400000_nodes_200k_forced_move_positions.txt"
# FILE_MIRROR_BOARD="stockfish_classic_results/stockfish_classic_mirror_position_1400000_nodes_200k_middlegame_positions.txt"
# FILE_TRANSFORMATION_BOARD="stockfish_classic_results/stockfish_classic_board_transformation_1400000_nodes_200k_no_pawns_synthetic_positions.txt"

# Classic Stockfish fewer nodes files
FILE_RECOMMENDED_MOVES="stockfish_classic_results/stockfish_classic_recommended_move_81000_nodes_100k_middlegame_positions.txt"
FILE_FORCED_MOVES="stockfish_classic_results/stockfish_classic_forced_move_81000_nodes_100k_forced_move_positions.txt"
FILE_MIRROR_BOARD="stockfish_classic_results/stockfish_classic_mirror_position_81000_nodes_100k_middlegame_positions.txt"
FILE_TRANSFORMATION_BOARD="stockfish_classic_results/stockfish_classic_board_transformation_81000_nodes_50k_no_pawns_synthetic_positions.txt"


OPTIONS_RECOMMENDED_MOVES="--score_columns score1 score2 --main_fen_column fen1 --increase_ply_for_columns child_score"
OPTIONS_FORCED_MOVES="--score_columns score1 score2 --main_fen_column fen1 --increase_ply_for_columns child_score"
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