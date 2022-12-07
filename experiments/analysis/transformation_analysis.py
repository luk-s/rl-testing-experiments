from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_results import load_data
from plot_distribution import differences_density_plot

if __name__ == "__main__":
    result_folder = Path(__file__).parent.parent / Path(
        "results/transformation_testing"
        # "results/forced_moves/main_experiment"
    )
    result_file = Path("results_ENGINE_local_400_nodes_DATA_database.txt")
    data_columns = [
        "original",
        "rot90",
        "rot180",
        "rot270",
        "flip_diag",
        "flip_anti_diag",
        "flip_hor",
        "flip_vert",
    ]

    df, config = load_data(result_folder / result_file)

    # Compute minimum and maximum value of each data column
    df["min_value"] = df[data_columns].min(axis=1)
    df["max_value"] = df[data_columns].max(axis=1)

    # Plot the differences between the minimum and maximum value
    differences_density_plot(
        dataframe=df,
        column_name1="max_value",
        column_name2="min_value",
    )

    id_min = df[data_columns].idxmin(axis=1)
    id_max = df[data_columns].idxmax(axis=1)

    id_max.groupby(id_max).count().plot(kind="bar", fontsize=18, rot=0)
    plt.show()
    id_min.groupby(id_min).count().plot(kind="bar", fontsize=18, rot=0)
    plt.show()

    print("Success")
