from pathlib import Path

import numpy as np
import pandas as pd
from load_results import load_data
from plot_distribution import differences_density_plot

if __name__ == "__main__":
    result_folder = Path(__file__).parent.parent / Path(
        "results/parent_child_testing/"
        # "results/forced_moves/main_experiment"
    )
    result_file = Path(
        # "results_ENGINE_local_1_node_DATA_database.txt",
        # "results_ENGINE_local_100_nodes_DATA_database.txt",
        "results_ENGINE_local_200_nodes_DATA_database.txt",
        # "results_ENGINE_local_400_nodes_DATA_database.txt",  # noqa E501
    )

    df, config = load_data(result_folder / result_file)

    # Drop invalid rows
    df = df[~df.isin(["invalid"]).any(axis=1)]

    result_list = list(df.values)
    move_list = list(range(11))
    smallest_differences = []
    for row in result_list:
        row = list(row)
        original_index = -1
        child_indices = move_list.copy()
        empty_index = 10
        for index, value in enumerate(row):
            if not isinstance(row[index], str) and np.isnan(row[index]):
                if original_index == -1:
                    original_index = index
                    del child_indices[index]
                else:
                    empty_index = index
                    row = row[:index]
                    break

        original_value = float(row[original_index + 1])
        child_values = []
        for i in range(2, len(row), 2):
            if i != original_index + 1:
                child_values.append(float(row[i]))

        original_value = -original_value
        child_values = np.array(child_values)
        differences = np.abs((child_values - original_value))
        smallest_differences.append(differences.min())

    df["smallest_difference"] = smallest_differences
    df["zero"] = 0

    differences_density_plot(dataframe=df, column_name1="smallest_difference", column_name2="zero")
