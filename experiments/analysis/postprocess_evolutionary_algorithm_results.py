import argparse
import math as m
from pathlib import Path
from typing import List, Optional

import chess
import pandas as pd
from load_results import load_data

from rl_testing.util.chess import has_undefended_attacked_pieces


def prepare_dataframe(file_path: Path, max_num_rows: Optional[int] = None) -> pd.DataFrame:
    # Load the data
    ev_dataframe = load_data(file_path)[0]

    # Drop all rows with duplicate fens
    # Apply the following function to each row of the dataframe
    ev_dataframe = ev_dataframe.drop_duplicates(subset="fen1")

    # Drop all rows where "fen2" is also in the "fen1" column
    fen_cache = set()

    # Iterate over all rows
    indices_to_remove = []
    for index, row in ev_dataframe.iterrows():
        # Get the fen
        fen1 = row["fen1"]
        fen2 = row["fen2"]

        # Check if the position is already in the cache
        if fen2 in fen_cache or fen1 in fen_cache:
            indices_to_remove.append(index)
        else:
            fen_cache.add(fen1)

    # Drop all rows with indices in 'indices_to_remove'
    ev_dataframe = ev_dataframe.drop(indices_to_remove)

    # Print the shape of the dataframe
    print("Before limiting:")
    print(ev_dataframe.shape)

    # Limit the number of rows
    if max_num_rows is not None:
        ev_dataframe = ev_dataframe.head(max_num_rows)

    ev_dataframe["position1"] = ev_dataframe["fen1"].apply(lambda fen: " ".join(fen.split()[:2]))
    ev_dataframe["position2"] = ev_dataframe["fen2"].apply(lambda fen: " ".join(fen.split()[:2]))

    ev_dataframe.drop_duplicates(subset="position1", inplace=True)

    # Iterate over all rows
    pos_cache = set()
    indices_to_remove = []
    for index, row in ev_dataframe.iterrows():
        pos1 = row["position1"]
        pos2 = row["position2"]

        # Check if the position is already in the cache
        if pos2 in pos_cache:
            indices_to_remove.append(index)
        else:
            pos_cache.add(pos1)

    # Drop all rows with indices in 'indices_to_remove'
    ev_dataframe = ev_dataframe.drop(indices_to_remove)

    # Remove all rows which don't contain a numeric string values value in the 'fitness1' and 'fitness2' column
    ev_dataframe = ev_dataframe[pd.to_numeric(ev_dataframe["fitness1"], errors="coerce").notnull()]
    ev_dataframe = ev_dataframe[pd.to_numeric(ev_dataframe["fitness2"], errors="coerce").notnull()]

    # Convert the 'fitness1' and 'fitness2' column to float
    ev_dataframe["fitness1"] = ev_dataframe["fitness1"].astype(float)
    ev_dataframe["fitness2"] = ev_dataframe["fitness2"].astype(float)

    # Compute the absolute difference between the 'fitness1' and 'fitness2' column
    ev_dataframe["difference"] = (ev_dataframe["fitness1"] - ev_dataframe["fitness2"]).abs()

    # Sort the dataframe w.r.t the difference column
    ev_dataframe = ev_dataframe.sort_values(by="difference", ascending=False)

    return ev_dataframe


def is_valid_board_transformation(fen: str) -> bool:
    allowed_num_pieces = {
        chess.ROOK: 2,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.QUEEN: 1,
        chess.KING: 1,
    }
    board = chess.Board(fen)
    piece_map = board.piece_map()
    pieces = list(piece_map.values())
    white_pieces = [piece for piece in pieces if piece.color == chess.WHITE]
    white_piece_types = [piece.piece_type for piece in white_pieces]
    black_pieces = [piece for piece in pieces if piece.color == chess.BLACK]
    black_piece_types = [piece.piece_type for piece in black_pieces]

    if has_undefended_attacked_pieces(board):
        return False

    if len(pieces) != 8:
        return False

    if len(white_pieces) != len(black_pieces):
        return False

    if len(set(white_piece_types)) != len(set(black_piece_types)):
        return False

    # Check if the number of pieces is correct
    for piece_type in allowed_num_pieces:
        if white_piece_types.count(piece_type) > allowed_num_pieces[piece_type]:
            return False
        if black_piece_types.count(piece_type) > allowed_num_pieces[piece_type]:
            return False

    return True


def filter_bad_positions(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Filter out all rows for which the 'fen1' column is not a valid board transformation
    dataframe = dataframe[dataframe["fen1"].apply(is_valid_board_transformation)]
    return dataframe


def get_percentages(raw_row: str) -> str:
    numbers = raw_row.split(" & ")
    original_divisor = numbers[0]

    numbers[0] = numbers[0].replace("k", "000")
    divisor = numbers[0]

    numbers = [int(number) for number in numbers]

    divisor = numbers[0]

    def round_to_1(x):
        if x == 0:
            return 0
        if x > 1:
            return round(x, 1)
        return round(x, -int(m.floor(m.log10(abs(x)))))

    def to_string(x):
        if x == 0:
            return "0\%"
        if x < 0.01:
            return "<0.01\%"
        return f"{x}\%"

    numbers = [round_to_1(100 * (number / divisor)) for number in numbers]

    numbers = [original_divisor] + [to_string(number) for number in numbers[1:]]
    return " & ".join(numbers)


def get_numbers(
    df: pd.DataFrame, score_columns: List[str], num_samples: int, bins: List[float]
) -> str:

    # Compute the maximum difference between any two scores in the score columns
    max_diff = df[score_columns].max(axis=1) - df[score_columns].min(axis=1)

    # For each bin, compute the number of differences which are larger than the value of the bin
    num_diffs = []
    for bin in bins:
        num_diffs.append((max_diff > bin).sum())

    return f"{num_samples // 1000}k & " + " & ".join([str(num_diff) for num_diff in num_diffs])


def print_violation_statistics(
    dataframe: pd.DataFrame, score_columns: List[str], num_samples: int, bins: List[float]
):
    # Sort the dataframe w.r.t the difference column
    dataframe = dataframe.sort_values(by="difference", ascending=False)

    violation_numbers = get_numbers(dataframe, score_columns, num_samples, bins)
    violation_percentages = get_percentages(violation_numbers)

    print(violation_percentages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--max_num_rows", type=int, default=None)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--print_violation_statistics", action="store_true")
    args = parser.parse_args()

    score_columns = ["fitness1", "fitness2"]
    ev_result_path = Path(args.file_path)
    bins = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

    ev_dataframe = prepare_dataframe(ev_result_path, max_num_rows=args.max_num_rows)
    print("After preparation:")
    print(ev_dataframe.shape)

    # Filter out all rows for which the 'fen1' column is not a valid board transformation
    ev_dataframe = filter_bad_positions(ev_dataframe)
    print(ev_dataframe.shape)

    if args.print_violation_statistics:
        print_violation_statistics(ev_dataframe, score_columns, args.max_num_rows, bins)

    # Store the dataframe as a csv file
    if args.save_csv:
        ev_dataframe.to_csv(
            str(ev_result_path.with_suffix("")) + "_filtered_sorted.csv", index=False, header=True
        )
