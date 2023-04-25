from pathlib import Path
from typing import Optional, Tuple

import chess
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from load_results import (
    compare_columns_and_filter,
    compute_differences,
    flip_q_values,
    load_data,
)

from rl_testing.util.chess import plot_board

RESULT_DIRECTORY = Path(__file__).parent.parent / Path("results")


def differences_density_plot(
    dataframe: pd.DataFrame,
    column_name1: Optional[str] = None,
    column_name2: Optional[str] = None,
    x_limits: Tuple[float, float] = (0, 2),
    y_limits: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> pd.DataFrame:
    if column_name1 is not None and column_name2 is not None:
        dataframe = compute_differences(dataframe, column_name1, column_name2)
    else:
        dataframe = dataframe.sort_values(by="difference", ascending=False)

    print(dataframe[:100].to_string())
    plt.hist(dataframe["difference"], bins=1000, range=x_limits)

    if y_limits is not None:
        plt.ylim(bottom=y_limits[0], top=y_limits[1])

    max_difference = dataframe["difference"].max()
    font = {"size": 16}
    matplotlib.rc("font", **font)
    plt.xlabel("Value difference", fontdict=font)
    plt.ylabel("Amount", fontdict=font)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale("log")
    print(max_difference)
    _, y_max = plt.gca().get_ylim()
    plt.vlines([max_difference], ymin=[0], ymax=y_max, colors=["red"])
    if title is not None:
        plt.title(title)
    plt.show()

    return dataframe


if __name__ == "__main__":
    # result_folder = RESULT_DIRECTORY / Path("differential_testing/main_results")
    result_folder = RESULT_DIRECTORY / Path("final_temp_data")
    # result_folder = RESULT_DIRECTORY / Path("evolutionary_algorithm/max_oracle_queries")
    # result_folder = RESULT_DIRECTORY / Path("parent_child_testing")
    # result_folder = RESULT_DIRECTORY / Path("forced_moves/main_experiment/")
    # result_file = Path("results_ENGINE_local_5000_nodes_DATA_random_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_forced_moves_fen_database.txt")
    # result_file = Path("results_ENGINE_local_dag_1_node_DATA_database.txt")
    # result_file = Path("results_ENGINE_local_minibatch_1_1_node_DATA_database.txt")
    # result_file = Path("results_ENGINE_local_minibatch_1_400_nodes_DATA_database.txt")
    # result_file = Path("results_ENGINE_local_2500_nodes_DATA_database.txt")
    # result_file = Path("oracle_queries_2023-02-24_18:55:04.txt")
    # result_file = Path("oracle_queries_2023-02-24_18:55:44.txt")
    # result_file = Path("oracle_queries_2023-02-25_08:01:01.txt")
    # result_file = Path("oracle_queries_2023-02-25_08:01:24.txt")
    # result_file = Path("oracle_queries_2023-02-25_11:49:11.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_middlegame_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_endgame_fen_database.txt")

    # This one contains the example we're using in the paper
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_combined_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_endgame_fen_database.txt")
    # result_file = Path("results_ENGINE_local_400_nodes_DATA_middlegame_fen_database.txt")
    result_file = Path(
        "mirror_results_ENGINE_local_400_nodes_DATA_final_middlegame_master_dense_fen.txt"  # Not good! Current outliers are just instabilities # noqa
        # "parent_child_results_ENGINE_local_400_nodes_DATA_final_forced_moves_master_fen.txt"  # Good! # noqa
        # "parent_child_results_ENGINE_local_400_nodes_DATA_final_middlegame_master_dense_fen.txt"  # Almost good # noqa
        # "parent_child_results_ENGINE_local_400_nodes_DATA_final_endgame_master_dense_fen.txt"  # Good! # noqa
    )

    title = "Recommended move testing: Endgame positions from Master games"
    column_name1, column_name2 = "original", "mirror"  # "parent_score", "child_score"  #
    column_difference_name = None  # "fitness"
    result_path = result_folder / result_file
    x_limits = (0, 2)
    y_limits = None  # (0, 10)
    q_vals_to_flip = []  # ["child_score"]  # ["Q2"]
    columns_to_compare = []  # ["Move1", "Move2"]
    compare_string = "!="

    dataframe, _ = load_data(result_path=result_path)

    if column_name1 is None or column_name2 is None:
        assert (
            column_difference_name is not None
        ), "If 'column_name1' or 'column_name2' is None, 'column_difference_name' must be set."

        dataframe["difference"] = dataframe[column_difference_name]

    for column_name in q_vals_to_flip:
        dataframe = flip_q_values(dataframe, column_name=column_name)

    if columns_to_compare:
        dataframe = compare_columns_and_filter(
            dataframe, *columns_to_compare, compare_string=compare_string
        )

    dataframe = differences_density_plot(
        dataframe=dataframe,
        column_name1=column_name1,
        column_name2=column_name2,
        x_limits=x_limits,
        y_limits=y_limits,
        title=title,
    )

    # Store the dataframe
    store_path = str(result_path).split(".")[0] + "_differences_sorted.csv"
    dataframe.to_csv(store_path)
