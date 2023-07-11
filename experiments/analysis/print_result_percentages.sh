RESULT_DIR="/home/lukas/Polybox_ETH/Master-Thesis/Code/rl-testing-experiments/experiments/results/final_data"
LEELA_DIR="leela_chess_zero_results"
EVOLUTIONARY_ALGORITHM_DIR="evolutionary_algorithm_results"
STOCKFISH_CLASSIC_DIR="stockfish_classic_results"
STOCKFISH_NNUE_DIR="stockfish_nnue_results"

PRINT_LEELA_RESULTS="True"
PRINT_SCALING_RESULTS="True"
PRINT_EVOLUTIONARY_ALGORITHM_RESULTS="True"
PRINT_STOCKFISH_CLASSIC_RESULTS="True"
PRINT_STOCKFISH_CLASSIC_RESULTS_FEWER_NODES="True"
PRINT_STOCKFISH_NNUE_RESULTS="True"

if [ $PRINT_LEELA_RESULTS == "True" ]; then
    echo "======================"
    echo "Printing Leela results"
    echo "======================"
    echo "Board transformations"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/board_transformation_400_nodes_200k_no_pawns_synthetic_positions.csv
    echo "Recommended moves"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/recommended_move_400_nodes_400k_middlegame_positions.csv
    echo "Differential testing"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/differential_testing_leela_400_nodes_T80_vs_leela_285_nodes_T78_400k_middlegame_positions.csv
    echo "Forced moves"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/forced_move_400_nodes_400k_forced_move_positions.csv
    echo "Position mirroring"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/mirror_position_400_nodes_400k_middlegame_positions.csv
fi

if [ $PRINT_SCALING_RESULTS == "True" ]; then
    echo "========================"
    echo "Printing scaling results"
    echo "========================"
    echo "Recommended moves, 1 node"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/recommended_move_1_node_100k_middlegame_positions.csv
    echo "Recommended moves, 100 nodes"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/recommended_move_100_nodes_100k_middlegame_positions.csv
    echo "Recommended moves, 200 nodes"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/recommended_move_200_nodes_100k_middlegame_positions.csv
    echo "Recommended moves, 400 nodes"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/recommended_move_400_nodes_100k_middlegame_positions.csv
    echo "Recommended moves, 800 nodes"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/recommended_move_800_nodes_100k_middlegame_positions.csv
    echo "Recommended moves, 1600 nodes"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/recommended_move_1600_nodes_100k_middlegame_positions.csv
fi

if [ $PRINT_EVOLUTIONARY_ALGORITHM_RESULTS == "True" ]; then
    echo "========================================"
    echo "Printing evolutionary algorithm results"
    echo "========================================"
    echo "Random boards"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$LEELA_DIR/board_transformation_1600_nodes_50k_no_pawns_synthetic_positions.csv
    echo "Adversarial boards, run 1"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$EVOLUTIONARY_ALGORITHM_DIR/board_transformation_1600_nodes_50k_no_pawns_synthetic_positions_evolutionary_algorithm1_filtered_sorted.csv
    echo "Adversarial boards, run 2"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$EVOLUTIONARY_ALGORITHM_DIR/board_transformation_1600_nodes_50k_no_pawns_synthetic_positions_evolutionary_algorithm2_filtered_sorted.csv
    echo "Adversarial boards, run 3"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$EVOLUTIONARY_ALGORITHM_DIR/board_transformation_1600_nodes_50k_no_pawns_synthetic_positions_evolutionary_algorithm3_filtered_sorted.csv
fi

if [ $PRINT_STOCKFISH_CLASSIC_RESULTS == "True" ]; then
    echo "==================================="
    echo "Printing Stockfish Classic results"
    echo "==================================="
    echo "Recommended moves"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_CLASSIC_DIR/stockfish_classic_recommended_move_1400000_nodes_200k_middlegame_positions_q_scores_differences_sorted.csv
    echo "Position mirroring"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_CLASSIC_DIR/stockfish_classic_mirror_position_1400000_nodes_200k_middlegame_positions_q_scores_differences_sorted.csv
    echo "Forced moves"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_CLASSIC_DIR/stockfish_classic_forced_move_1400000_nodes_200k_forced_move_positions_q_scores_differences_sorted.csv
    echo "Board transformations"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_CLASSIC_DIR/stockfish_classic_board_transformation_1400000_nodes_200k_no_pawns_synthetic_positions_q_scores_differences_sorted.csv
fi

if [ $PRINT_STOCKFISH_CLASSIC_RESULTS_FEWER_NODES == "True" ]; then
    echo "==========================================="
    echo "Printing Stockfish Classic fewer nodes results"
    echo "==========================================="
    echo "Recommended moves"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_CLASSIC_DIR/stockfish_classic_recommended_move_81000_nodes_100k_middlegame_positions_q_scores_differences_sorted.csv
    echo "Position mirroring"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_CLASSIC_DIR/stockfish_classic_mirror_position_81000_nodes_100k_middlegame_positions_q_scores_differences_sorted.csv
    echo "Forced moves"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_CLASSIC_DIR/stockfish_classic_forced_move_81000_nodes_100k_forced_move_positions_q_scores_differences_sorted.csv
    echo "Board transformations"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_CLASSIC_DIR/stockfish_classic_board_transformation_81000_nodes_50k_no_pawns_synthetic_positions_q_scores_differences_sorted.csv
fi

if [ $PRINT_STOCKFISH_NNUE_RESULTS == "True" ]; then
    echo "================================"
    echo "Printing Stockfish NNUE results"
    echo "================================"
    echo "Recommended moves"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_NNUE_DIR/stockfish_nnue_recommended_move_81000_nodes_400k_middlegame_positions_q_scores_differences_sorted.csv
    echo "Position mirroring"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_NNUE_DIR/stockfish_nnue_mirror_position_81000_nodes_400k_middlegame_positions_q_scores_differences_sorted.csv
    echo "Forced moves"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_NNUE_DIR/stockfish_nnue_forced_move_81000_nodes_400k_forced_move_positions_q_scores_differences_sorted.csv
    echo "Board transformations"
    python3 experiments/analysis/print_result_percentages.py --result_path $RESULT_DIR/$STOCKFISH_NNUE_DIR/stockfish_nnue_board_transformation_81000_nodes_200k_no_pawns_synthetic_positions_q_scores_differences_sorted.csv
fi
