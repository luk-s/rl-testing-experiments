from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd


def load_data(result_path: Union[str, Path]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    # Find the start of the real data in the result file
    config = {}
    start_line = 0
    with open(result_path, "r") as f:
        line = f.readline()
        line = line[:-1]
        if ":" in line:
            line = f.readline()
            line = line[:-1]
        if "," not in line:
            while line != "":
                # Parse the config
                if line != "":
                    name, value = line.split("=")
                    name, value = name.strip(), value.strip()
                    config[name] = value

                start_line += 1
                line = f.readline()
                line = line[:-1]
                # if line == "":
                #    start_line += 1

    if "," not in line:
        start_line += 1

    # Read in the data
    dataframe = pd.read_csv(result_path, header=start_line, skip_blank_lines=False)

    return dataframe, config


def flip_q_values(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    dataframe[column_name] = -dataframe[column_name]
    return dataframe


def compute_differences(
    dataframe: pd.DataFrame, column_name1: str, column_name2: str
) -> pd.DataFrame:
    dataframe["difference"] = dataframe[column_name1] - dataframe[column_name2]
    dataframe["difference"] = dataframe["difference"].abs()
    dataframe = dataframe.sort_values(by="difference", ascending=False)
    return dataframe


def compare_columns_and_filter(
    dataframe: pd.DataFrame, column_name1: str, column_name2: str, compare_string: str = "=="
):
    if compare_string == "==":
        dataframe2 = dataframe.loc[dataframe[column_name1] == dataframe[column_name2]]
    elif compare_string == "!=":
        dataframe2 = dataframe.loc[dataframe[column_name1] != dataframe[column_name2]]
    elif compare_string == ">=":
        dataframe2 = dataframe.loc[dataframe[column_name1] >= dataframe[column_name2]]
    elif compare_string == "<=":
        dataframe2 = dataframe.loc[dataframe[column_name1] <= dataframe[column_name2]]
    elif compare_string == "<":
        dataframe2 = dataframe.loc[dataframe[column_name1] < dataframe[column_name2]]
    elif compare_string == ">":
        dataframe2 = dataframe.loc[dataframe[column_name1] != dataframe[column_name2]]
    else:
        raise ValueError(
            f"The compare string '{compare_string}' is not supported! "
            "Please choose one from ['==', '!=', '>=', '<=', '<', '>']"
        )

    return dataframe2
