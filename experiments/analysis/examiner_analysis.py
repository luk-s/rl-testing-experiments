from pathlib import Path

import numpy as np
import pandas as pd
from load_results import load_data

if __name__ == "__main__":
    result_folder = Path(__file__).parent.parent / Path(
        "results/examiner_testing/main_experiment"
        # "results/forced_moves/main_experiment"
    )
    result_file = Path(
        "results_VICTIM_ENGINE_local_400_nodes_EXAMINER_ENGINE_local_25_depth_stockfish_DATA_database.txt"  # noqa E501
    )

    df, config = load_data(result_folder / result_file)

    # Compute the number of games contained in the dataset
    num_games = 1
    old_move_number = 0
    game_start_indices = [0]
    success_indices = []
    row_index = 0
    success = df["success"]
    for fen in df["board_original"]:
        move_number = int(fen.split(" ")[5])

        # Check if a new game starts
        if move_number < 5 and old_move_number > 10:
            num_games += 1
            game_start_indices.append(row_index)

        # Check if a successful row was found
        if success[row_index] == 1:
            success_indices.append(row_index)

        old_move_number = move_number
        row_index += 1

    print(f"Number of games: {num_games}")
    print(f"Game start indices: {game_start_indices}")
    print(f"Success indices: {success_indices}")
    print("Finished!")
