from pathlib import Path

import pandas as pd
from load_results import compute_differences, load_data

RESULT_DIRECTORY = Path(__file__).parent.parent / Path("results")
DATA_DIRECTORY = Path(__file__).parent.parent.parent / Path("data")

if __name__ == "__main__":
    ################
    # CONFIG START #
    ################
    result_folder = RESULT_DIRECTORY / Path("differential_testing/main_results")

    result_file_names = [
        Path("results_ENGINE_local_1_node_DATA_database.txt"),
        Path("results_ENGINE_local_100_nodes_DATA_database.txt"),
        Path("results_ENGINE_local_200_nodes_DATA_database.txt"),
        Path("results_ENGINE_local_400_nodes_DATA_database.txt"),
        Path("results_ENGINE_local_1000_nodes_DATA_database.txt"),
        Path("results_ENGINE_local_2500_nodes_DATA_database.txt"),
        # Path("results_ENGINE_local_5000_nodes_DATA_database.txt"),
        # Path("results_ENGINE_local_10000_nodes_DATA_database.txt"),
    ]

    score_difference_threshold = 0.5
    column_name1, column_name2 = "score1", "score2"
    ################
    #  CONFIG END  #
    ################

    dataframe_list = []

    # Iterate over all result files
    for result_file_name in result_file_names:
        # Load the data
        result_file_path = result_folder / result_file_name
        dataframe, _ = load_data(result_path=result_file_path)

        # Compute the differences
        dataframe = compute_differences(dataframe, column_name1, column_name2)

        # Only keep rows where "difference" is above the threshold
        dataframe = dataframe[dataframe["difference"] > score_difference_threshold]

        # Add the dataframe to the list
        dataframe_list.append(dataframe)

    # Concatenate all dataframes
    dataframe = pd.concat(dataframe_list, ignore_index=True, axis=0)

    # Drop all columns except "fen"
    dataframe = dataframe[["fen"]]

    # Drop all rows with duplicate fens
    dataframe = dataframe.drop_duplicates(subset="fen")

    # Store the dataframe
    dataframe.to_csv(DATA_DIRECTORY / Path("interesting_fens.csv"), index=False)
