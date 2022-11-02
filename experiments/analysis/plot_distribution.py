from pathlib import Path
from typing import List, Optional, Tuple
from unittest import result

import matplotlib.pyplot as plt
import pandas as pd

from load_results import compute_differences, flip_q_values, load_data

RESULT_DIRECTORY = Path(__file__).parent.parent / Path("results")


def differences_density_plot(
    dataframe: pd.DataFrame,
    column_name1: str,
    column_name2: str,
    x_limits: Tuple[float, float] = (0, 2),
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    dataframe["difference"] = dataframe[column_name1] - dataframe[column_name2]
    dataframe["difference"] = dataframe["difference"].abs()
    plt.hist(dataframe["difference"], bins=1000, range=x_limits)

    if y_limits is not None:
        plt.ylim(bottom=y_limits[0], top=y_limits[1])

    max_difference = dataframe["difference"].max()
    print(max_difference)
    plt.vlines([max_difference], ymin=[0], ymax=8)
    plt.show()


if __name__ == "__main__":
    # result_folder = RESULT_DIRECTORY / Path("differential_testing/main_experiment/")
    result_folder = RESULT_DIRECTORY / Path("forced_moves/main_experiment/")
    result_file = Path("results_ENGINE_local_1_node_DATA_forced_moves_fen_database.txt")
    column_name1, column_name2 = "Q1", "Q2"
    result_path = result_folder / result_file
    x_limits = (0, 2)
    y_limits = (0, 10)
    q_vals_to_flip = ["Q2"]

    dataframe, _ = load_data(result_path=result_path)
    for column_name in q_vals_to_flip:
        dataframe = flip_q_values(dataframe, column_name=column_name)

    differences_density_plot(
        dataframe=dataframe,
        column_name1=column_name1,
        column_name2=column_name2,
        x_limits=x_limits,
        y_limits=y_limits,
    )
