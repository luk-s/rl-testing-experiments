import argparse
from pathlib import Path
from typing import List
import pandas as pd


def combine_result_files(
    input_files: List[str], output_file: str, fen_column_name: str = "fen"
) -> None:
    """Combines multiple result .csv files into one.

    Args:
        input_files (List[str]): Paths to the result .csv files that should be combined.
        output_file (str): Path to the output file where the combined results should be stored.
    """
    # Read all input files
    dataframes: List[pd.DataFrame] = []
    for input_file in input_files:
        dataframes.append(pd.read_csv(input_file))

    # Sort the dataframes by the number of rows descending
    dataframes = sorted(dataframes, key=lambda df: len(df), reverse=True)

    # Combine the dataframes
    combined_dataframe: pd.DataFrame = dataframes[0]

    num_smaller = 0
    max_difference = 0

    for df in dataframes[1:]:
        # Iterate over all rows of the dataframe
        for row_index, row in df.iterrows():
            print(f"Processing row {row_index}/{len(df)} ")

            # Check if the FEN is already present in the combined dataframe
            if row[fen_column_name] in combined_dataframe[fen_column_name].tolist():
                # Get the index of the row in the combined dataframe
                index = combined_dataframe[
                    combined_dataframe[fen_column_name] == row[fen_column_name]
                ].index[0]

                # If the value in the "difference" column of this row is smaller than the value in
                # the "difference" column of the row in the combined dataframe, replace the row in
                # the combined dataframe with this row
                if row["difference"] < combined_dataframe.loc[index, "difference"]:
                    max_difference = max(
                        max_difference,
                        combined_dataframe.loc[index, "difference"] - row["difference"],
                    )
                    combined_dataframe.loc[index] = row
                    num_smaller += 1

            # If the FEN is not yet present in the combined dataframe, append it
            else:
                print("Adding new row to combined dataframe.")
                combined_dataframe = combined_dataframe.append(row)

    # Store the combined dataframe
    combined_dataframe.to_csv(output_file, index=False)

    print(f"Found {num_smaller} rows with a smaller difference value.")
    print(f"Maximum difference value: {max_difference}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple result .csv files into one.")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="Paths to the result .csv files that should be combined.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file where the combined results should be stored.",
    )
    parser.add_argument(
        "--fen_column_name",
        type=str,
        default="fen",
        help="Name of the column in the result .csv files that contains the FENs.",
    )
    args = parser.parse_args()

    combine_result_files(args.input_files, args.output_file, args.fen_column_name)
