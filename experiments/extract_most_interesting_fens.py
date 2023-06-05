import argparse
from typing import List
import pandas as pd
from pathlib import Path
import configparser

DATA_CONFIG_FOLDER = Path(__file__).absolute().parent / "configs" / "data_generator_configs"


def extract_most_interesting_fens(input_file: str, num: int = 1000) -> List[str]:
    """
    Extracts the most interesting FENs from a result .csv file and saves them to a new file.
    The new file is also used to create a new experiment data config file.

    Args:
        input_file (str): Path to a result .csv file which should be used to extract the most
            interesting FENs.
        num (int, optional): Number of most interesting FENs to extract. Defaults to 1000.
    """
    df = pd.read_csv(input_file)
    # Get all columns of the dataframe
    cols = df.columns.tolist()

    # Make sure that all necessary columns are present
    assert ("fen" in cols or "parent_fen" in cols) and "difference" in cols, (
        "The result .csv file must contain the columns 'fen' or 'parent_fen', as well as"
        " 'difference'."
    )

    # Check which fen column is present
    if "fen" in cols:
        fen_col = "fen"
    else:
        fen_col = "parent_fen"

    # Sort the dataframe by the difference column
    df = df.sort_values(by="difference", ascending=False)

    # Extract the most interesting FENs
    fens = df[fen_col].tolist()[:num]

    return fens


def store_interesting_fens(fens: List[str], output_path: str, overwrite: bool = False) -> None:
    """Stores the most interesting FENs to a file.

    Args:
        fens (List[str]): A list of FENs that should be stored.
        output_path (str): Path to a file where the most interesting FENs should be saved.
        overwrite (bool, optional): If True, the file will be overwritten if it already exists.
            Defaults to False.

    Raises:
        FileExistsError: If the output file already exists and overwrite is False.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"The output file '{output_path}' already exists. Please choose a different path or"
            " set overwrite to True."
        )

    with open(output_path, "w") as f:
        for fen in fens:
            f.write(fen + "\n")


def create_experiment_data_config_file(output_path: str, overwrite: bool = False) -> None:
    """Creates a new experiment data config file.

    Args:
        output_path (str): Path to the file where the FENs got stored.
        overwrite (bool, optional): If True, the file will be overwritten if it already exists.
            Defaults to False.

    Raises:
        FileExistsError: If the output file already exists and overwrite is False.
    """
    output_path: Path = Path(output_path)

    # Create the config file
    config = configparser.ConfigParser()
    config["General"] = {"data_generator_type": "fen_database_board_generator"}
    config["DataGeneratorConfig"] = {"database_name": output_path.name, "open_now": True}

    # Create a config file name
    config_file_name = output_path.name.split(".")[0] + "_fen.ini"

    # Check if the file already exists
    if (DATA_CONFIG_FOLDER / config_file_name).exists() and not overwrite:
        raise FileExistsError(
            f"The output file '{config_file_name}' already exists. Please choose a different path"
            " or set overwrite to True."
        )

    with open(DATA_CONFIG_FOLDER / config_file_name, "w") as config_file:
        config.write(config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract most interesting FENs")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to a result .csv file which should be used to extract the most interesting FENs",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help=(
            "Path to a file where the most interesting FENs should be saved. Is also used to create"
            " a new experiment data config file."
        ),
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="Number of most interesting FENs to extract",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If True, the output file will be overwritten if it already exists.",
    )
    args = parser.parse_args()

    # Extract the most interesting FENs
    print("Extracting the most interesting FENs...")
    fens = extract_most_interesting_fens(args.input_file, args.num)

    # Store the most interesting FENs
    print(f"Storing the most interesting FENs in the file '{args.output_file}'...")
    store_interesting_fens(fens, args.output_file, args.overwrite)

    # Create a new experiment data config file
    print(f"Creating a new experiment data config file in '{DATA_CONFIG_FOLDER}'...")
    create_experiment_data_config_file(args.output_file, args.overwrite)
