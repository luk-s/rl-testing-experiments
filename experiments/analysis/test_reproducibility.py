from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from load_results import load_data

RESULT_DIRECTORY = Path(__file__).parent.parent / Path("results")


if __name__ == "__main__":
    ################
    # CONFIG START #
    ################
    result_folder = RESULT_DIRECTORY / Path("differential_testing/main_results")

    result_file_name1 = "results_ENGINE_local_100_nodes_DATA_database.txt"

    result_file_name2 = "results_ENGINE_local_100_nodes_DATA_database_FIRST_RUN.txt"
    ################
    #  CONFIG END  #
    ################

    # Load the dataframes from both result files
    dataframe1, _ = load_data(result_path=result_folder / Path(result_file_name1))
    dataframe2, _ = load_data(result_path=result_folder / Path(result_file_name2))

    # Add the columns of dataframe2 as additional columns to dataframe1.
    dataframe1 = dataframe1.join(dataframe2, lsuffix="_1", rsuffix="_2")

    # Compute the absolute difference between the scores of the two dataframes
    dataframe1["score_difference1"] = dataframe1["score1_1"] - dataframe1["score1_2"]
    dataframe1["score_difference2"] = dataframe1["score2_1"] - dataframe1["score2_2"]
    dataframe1["score_difference1"] = dataframe1["score_difference1"].abs()
    dataframe1["score_difference2"] = dataframe1["score_difference2"].abs()

    # Plot histograms of the score differences
    plt.rc("font", size=24)

    ax1 = plt.subplot(1, 2, 1)
    ax1.hist(dataframe1["score_difference1"], bins=np.linspace(0, 2, 200))
    ax1.set_yscale("log")
    ax1.set_xlabel("Absolute score difference")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Network T807301")

    ax2 = plt.subplot(1, 2, 2)
    ax2.hist(dataframe1["score_difference2"], bins=np.linspace(0, 2, 200))
    ax2.set_yscale("log")
    ax2.set_xlabel("Absolute score difference")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Network T785469")

    plt.suptitle("Score differences between two runs of the same experiment")

    plt.show()
