from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from load_results import load_data

from rl_testing.util.util import cp2q

RESULT_DIRECTORY = Path(__file__).parent.parent / Path("results")
IMAGE_DIRECTORY = Path(__file__).parent / Path("images")


class AnalysisResult:
    __slots__ = ["fen", "max_diff", "max_std", "scores"]

    def __init__(self, fen: str, max_diff: float, max_std: float, scores: np.ndarray) -> None:
        self.fen = fen
        self.max_diff = max_diff
        self.max_std = max_std
        self.scores = scores


def plot_results(
    results: Tuple[Dict[str, AnalysisResult], Dict[str, AnalysisResult]],
    log_num_largest_std: Union[int, str] = 0,
    num_bins: int = 200,
    plot=True,
    save=False,
    save_path: Union[str, Path] = "",
) -> None:
    combination_dict = {}

    assert (
        isinstance(results, dict) or len(results) == 2
    ), "Results must be either a dictionary or a tuple of two dictionaries"

    if isinstance(results, dict):
        results = [results]

    # Iterate over all dictionaries
    for result_dict in results:
        # Iterate over all fens
        for fen, result in result_dict.items():
            # Add the result to the combination dictionary
            combination_dict[fen] = combination_dict.get(fen, []) + [result]

    # For each fen, compute the combined maximum difference and standard deviation
    result_list = []
    for fen, analysis_result_list in combination_dict.items():
        # Get the scores of all the results
        scores = np.concatenate([result.scores for result in analysis_result_list], axis=0)

        # Compute the maximum difference and standard deviation
        max_diff = np.max(scores) - np.min(scores)
        std = np.std(scores)

        # Add the result to the list
        result_list.append((fen, max_diff, std, analysis_result_list))

    # Sort the list by the maximum difference and standard deviation descending
    results_max_diff = sorted(result_list, key=lambda x: x[1], reverse=True)
    results_max_std = sorted(result_list, key=lambda x: x[2], reverse=True)

    # Compute the histogram bins
    bins = np.linspace(-1, 1, num_bins)

    # If 'log_num_largest_std' is 'all', plot all results
    if log_num_largest_std == "all":
        log_num_largest_std = len(results_max_diff)

    # If images shall be saved, create the directory if it does not exist
    if save and log_num_largest_std > 0:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    # Plot the histogram of the 'plot_num_largest_std' results with the largest standard deviation
    for row_data, data_name in zip(
        [results_max_diff, results_max_std], ["Maximum difference", "Maximum std"]
    ):
        for i in range(log_num_largest_std):
            fen, max_diff, std, analysis_result_list = row_data[i]

            title_string = (
                f"[{data_name} {i+1}/{log_num_largest_std}] {max_diff = }, {std = }\n{fen = }"
            )
            print(title_string)  # For convenience to easily copy the fen

            if plot or save:
                # Create the (potentially stacked) histogram
                bottom = None
                for analysis_result in analysis_result_list:
                    values, _ = np.histogram(analysis_result.scores, bins=bins)
                    plt.hist(analysis_result.scores, bins=bins, rwidth=0.9, bottom=bottom)
                    bottom = values

                # Add a title and labels
                plt.title(title_string, fontsize=20)
                plt.xlabel("Score", fontsize=18)
                plt.ylabel("Frequency", fontsize=18)
                plt.yscale("log")
                plt.ylim(0.1, 10**4)

                # Show the plot
                if plot:
                    plt.show()

                if save:
                    plt.savefig(
                        Path(save_path) / Path(f"{data_name.replace(' ','_')}_{i+1}.png"),
                        bbox_inches="tight",
                    )

                plt.close()


def analyze_result_files(dataframe_list: list) -> Dict[str, AnalysisResult]:
    # Get number of columns
    number_of_columns = len(dataframe_list[0].columns)
    assert number_of_columns % 2 == 1, "Number of columns must be odd"
    assert all(
        len(dataframe_list[i].columns) == number_of_columns for i in range(len(dataframe_list))
    ), "All dataframes must have the same number of columns"  # noqa: E501
    number_of_scores = (number_of_columns - 1) // 2

    fen_results: Dict[str, np.ndarray] = {}

    # Iterate over all provided dataframes
    for dataframe in dataframe_list:
        # Iterate over all rows and extract the scores
        for index, row in dataframe.iterrows():
            # Extract the fen and the scores
            fen = row["fen"]
            scores = row[[f"score{i}" for i in range(number_of_scores)]].values

            # Convert the scores from centipawns to q-values
            scores = list(map(lambda score: cp2q(score), scores))

            # Add the scores to the dictionary
            if fen in fen_results:
                fen_results[fen] = np.concatenate((fen_results[fen], scores))
            else:
                fen_results[fen] = scores

    result_dict = {}

    # Iterate over all fens and compute the standard deviation and the maximum difference between
    # any two elements
    for fen, scores in fen_results.items():
        assert fen not in result_dict, "Fen must not be in the result dictionary"
        std = np.std(scores)
        max_diff = np.max(scores) - np.min(scores)

        # Add the results to the list
        result_dict[fen] = AnalysisResult(fen, max_diff, std, scores)

    return result_dict


if __name__ == "__main__":
    ################
    # CONFIG START #
    ################
    result_folder = RESULT_DIRECTORY / Path("score_positions")

    result_file_names_group1 = [
        "results_ENGINE_local_400_nodes_DATA_interesting_fen_database_NETWORK_T785469*",  # noqa: E501
        # "results_ENGINE_local_100_nodes_DATA_interesting_fen_database_NETWORK_T785469*",  # noqa: E501
        # "results_ENGINE_local_1_node_DATA_interesting_fen_database_NETWORK_T785469*",  # noqa: E501
    ]
    result_file_names_group2 = [
        "results_ENGINE_local_400_nodes_DATA_interesting_fen_database_NETWORK_T807301*",  # noqa: E501
        # "results_ENGINE_local_100_nodes_DATA_interesting_fen_database_NETWORK_T807301*",  # noqa: E501
        # "results_ENGINE_local_1_node_DATA_interesting_fen_database_NETWORK_T807301*",  # noqa: E501
    ]
    image_subdirectory = "400_nodes"
    ################
    #  CONFIG END  #
    ################

    dataframe_list1 = []
    dataframe_list2 = []
    result_dict_list = []

    for result_file_patterns, dataframe_list in zip(
        [result_file_names_group1, result_file_names_group2], [dataframe_list1, dataframe_list2]
    ):
        result_file_paths = []
        # If the result file pattern ends with a '*', all files in the result folder that match the
        # pattern are loaded. Otherwise, only the file with the exact name is loaded.
        for result_file_pattern in result_file_patterns:
            if result_file_pattern.endswith("*"):
                result_file_paths += list(result_folder.glob(result_file_pattern))
            else:
                result_file_paths.append(result_folder / Path(result_file_pattern))

        # Load all dataframes from the result files
        for result_file_path in result_file_paths:
            # Load the data
            dataframe, _ = load_data(result_path=result_file_path)

            dataframe_list.append(dataframe)

    # Analyze the dataframe lists
    result_dict1 = analyze_result_files(dataframe_list1)
    result_dict2 = analyze_result_files(dataframe_list2)

    # Plot the results
    plot_results(
        [result_dict1, result_dict2],
        log_num_largest_std="all",
        plot=False,
        num_bins=200,
        save=True,
        save_path=IMAGE_DIRECTORY / Path(image_subdirectory),
    )
