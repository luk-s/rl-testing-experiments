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

    result_file_patterns = [
        "results_ENGINE_local_minibatch_1_*",
    ]

    score_difference_threshold = 0.5
    column_name1, column_name2 = "score1", "score2"
    ################
    #  CONFIG END  #
    ################

    dataframe_list = []

    # Get all result files
    result_file_names = []
    for result_file_pattern in result_file_patterns:
        result_file_names += list(result_folder.glob(result_file_pattern))

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

    # Store the dataframe as txt file
    dataframe.to_csv(DATA_DIRECTORY / Path("interesting_fens.txt"), index=False, header=False)
