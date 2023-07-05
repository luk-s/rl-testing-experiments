import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import chess
import numpy as np
import pandas as pd
from load_results import load_data

from rl_testing.util.chess import extract_ply, stockfish_cp_to_leela_q


def convert_stockfish_scores_to_leela(
    input_path: str,
    score_columns: List[str],
    main_fen_column: str,
    increase_ply_for_columns: Optional[List[str]] = None,
    in_place: bool = False,
):
    """
    Converts the scores of a result file from Stockfish to Leela scores.

    Args:
        input_path (str): The path to the result file whose scores should be converted.
        score_columns (list): The columns that should be converted to Leela scores.
        in_place (bool, optional): Whether the input file should be overwritten. Defaults to False.
    """
    if increase_ply_for_columns is None:
        increase_ply_for_columns = []

    # Read the result file
    input_path = Path(input_path)
    df, _ = load_data(input_path)

    # Extract the plies from the main FEN
    df["ply"] = df[main_fen_column].apply(extract_ply)

    # For all score columns, convert the scores
    for score_column in score_columns:
        # Append '_cp' to the column name
        df.rename(columns={score_column: f"{score_column}_cp"}, inplace=True)

        # Check if the ply should be increased by 1
        if score_column in increase_ply_for_columns:
            additonal_ply = 1
        else:
            additonal_ply = 0

        # Convert the scores
        df[f"{score_column}"] = df.apply(
            lambda x: stockfish_cp_to_leela_q(x[score_column + "_cp"], x["ply"] + additonal_ply),
            axis=1,
        )

    # Write the result file
    if in_place:
        df.to_csv(input_path, index=False)
    else:
        output_path = input_path.parent / f"{input_path.stem}_q_scores.csv"
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    # Engine parameters
    parser.add_argument("--input_path", type=str, help="The path to the result file whose scores should be converted")  # noqa
    parser.add_argument("--score_columns", type=str, nargs="+", help="The columns that should be converted to Leela scores")  # noqa
    parser.add_argument("--main_fen_column", type=str, help="The column that contains the main FEN") # noqa
    parser.add_argument("--increase_ply_for_columns", type=str, nargs="+", help="The columns for which the ply should be increased by 1")  # noqa
    parser.add_argument("--in_place", action="store_true", help="Whether the input file should be overwritten")  # noqa
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################
    args = parser.parse_args()

    convert_stockfish_scores_to_leela(
        input_path=args.input_path,
        score_columns=args.score_columns,
        main_fen_column=args.main_fen_column,
        increase_ply_for_columns=args.increase_ply_for_columns,
        in_place=args.in_place,
    )

    """
    stockfish_score = -30
    # fen = "r6b/ppq2p1P/3rpk1B/3b4/2nP2P1/2P4R/PPQ2P2/1K1R1B2 w - - 3 220"
    # fen = "1r2r1k1/p2qpp1p/3p2p1/2pP2B1/2PbR3/1P4P1/P2Q1P1P/1R4K1 b - - 0 19"
    # fen = "8/3k1pp1/1rbpp2p/2p5/2PbP1P1/1P3PP1/1rNK4/1RN1R3 w - - 5 38"
    fen = "r3qnk1/4bppp/r2np3/p2pN1P1/1PpP1B2/2P1P3/1PB1QP1P/R5RK b - - 0 25"

    parts = fen.split(" ")
    fullmove = int(parts[-1]) - 1
    blacksturn = parts[-5] == "b"
    print(f"Fullmove: {fullmove}")
    gamePly = fullmove * 2 + blacksturn
    print(f"Game ply: {gamePly}")
    wdl = stockfish_cp_to_wdl(stockfish_score, gamePly)

    print(f"Stockfish score: {stockfish_score}")
    print(f"WDL: {wdl}")
    """
