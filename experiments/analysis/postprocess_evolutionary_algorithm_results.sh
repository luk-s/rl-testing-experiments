RESULT_DIR="/home/lukas/Polybox_ETH/Master-Thesis/Code/rl-testing-experiments/experiments/results/final_data/evolutionary_algorithm_results2"

MAX_NUM_ROWS=50000
SAVE_CSV="True"
PRINT_VIOLATION_STATISTICS="False"

GENERAL_ARGUMENTS="--max_num_rows $MAX_NUM_ROWS"

if [ $SAVE_CSV == "True" ]; then
    GENERAL_ARGUMENTS="$GENERAL_ARGUMENTS --save_csv"
fi
echo "============================================="
echo "Postprocessing evolutionary algorithm results"
echo "============================================="
echo "First results"
python3 experiments/analysis/postprocess_evolutionary_algorithm_results.py --file_path $RESULT_DIR/board_transformation_1600_nodes_50k_no_pawns_synthetic_positions_evolutionary_algorithm1.txt $GENERAL_ARGUMENTS
echo "Second results"
python3 experiments/analysis/postprocess_evolutionary_algorithm_results.py --file_path $RESULT_DIR/board_transformation_1600_nodes_50k_no_pawns_synthetic_positions_evolutionary_algorithm2.txt $GENERAL_ARGUMENTS
echo "Third results"
# python3 experiments/analysis/postprocess_evolutionary_algorithm_results.py --file_path $RESULT_DIR/board_transformation_1600_nodes_50k_no_pawns_synthetic_positions_evolutionary_algorithm3.txt $GENERAL_ARGUMENTS
