from pathlib import Path

import numpy as np
from chess import Board
from load_results import compute_differences, load_data

from rl_testing.util.util import fen_to_file_name, plot_board

if __name__ == "__main__":
    result_folder = Path(__file__).parent.parent / Path(
        "results/differential_testing/main_experiment"
        # "results/differential_testing/main_experiment/results_fixed_and_long"
        # "results/forced_moves/main_experiment"
    )
    # result_file = Path("results_ENGINE_local_5000_nodes_DATA_random_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_forced_moves_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_late_move_fen_database.txt")
    result_files = [
        Path("results_ENGINE_local_1000_nodes_DATA_late_move_fen_database.txt"),
        Path("results_ENGINE_local_2500_nodes_DATA_late_move_fen_database.txt"),
        Path("results_ENGINE_local_5000_nodes_DATA_late_move_fen_database.txt"),
        Path("results_ENGINE_local_10000_nodes_DATA_late_move_fen_database.txt"),
    ]
    boards_to_plot = []  # [4]
    save = True
    save_directory = Path(__file__).parent / "images"
    num_outliers_per_file = 100
    fen_key = "FEN"
    only_position = False

    fen_lists = []

    # For each result file, gather the 100 boards with the largest differences
    for result_file in result_files:
        print(f"Extracting outliers from file {result_file}")
        # Extract the boards with the largest difference
        dataframe, _ = load_data(result_folder / result_file)
        dataframe = compute_differences(dataframe=dataframe, column_name1="Q1", column_name2="Q2")
        fen_largest_differece = dataframe[:num_outliers_per_file][fen_key].values

        # If we're only interested in the board position, reduce the FEN string to the board position
        if only_position:
            extract_position = lambda string: string.split("_")[0]
            extract_position_np = np.vectorize(extract_position)
            fen_largest_differece = extract_position_np(fen_largest_differece)

        fen_largest_differece_list = list(fen_largest_differece)
        fen_largest_differece_list = [
            str(fen).replace("_", " ") for fen in fen_largest_differece_list
        ]
        fen_lists.append(fen_largest_differece_list)

    # Get a list of all unique board positions
    all_fens = []
    for fen_list in fen_lists:
        all_fens += fen_list
    all_fens = list(set(all_fens))

    # Build a statistics dict
    num_present_dict = {}

    # Initialize the dict
    for num_occurences in range(1, len(result_files) + 1):
        num_present_dict[num_occurences] = []

    # Fill the dict
    for fen in all_fens:
        num_present = 0
        for fen_list in fen_lists:
            if fen in fen_list:
                num_present += 1

        num_present_dict[num_present].append(fen)

    # Print the result
    statistics_dict = {}
    for num_occurences in range(1, len(result_files) + 1):
        title = f"\nBoards which are present in {num_occurences} result file" + (
            "s:" if num_occurences > 1 else ":"
        )
        print(title)
        print("=" * len(title))
        for fen in num_present_dict[num_occurences]:
            print(fen)

        statistics_dict[num_occurences] = len(num_present_dict[num_occurences])

    print(f"\nStatistics: {statistics_dict}")

    # Save plots of interesting boards
    for num_occurences in sorted(boards_to_plot):
        for fen in num_present_dict[num_occurences]:
            board = Board(fen=fen)
            plot_board(
                board,
                fen=fen,
                save=save,
                show=not save,
                fontsize=14,
                save_path=save_directory / fen_to_file_name(fen, ".png"),
            )
