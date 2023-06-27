from pathlib import Path
from typing import List, Optional, Tuple
import chess
import pandas as pd
import argparse
from load_results import load_data
import numpy as np


VALUE_MATE_IN_MAX_PLY = 31754
NormalizeToPawnValue = 361
VALUE_MATE = 32000

NORMALIZED_SCORE_MAX = VALUE_MATE_IN_MAX_PLY * 100 / NormalizeToPawnValue


def win_rate(score: float, ply: int) -> int:
    """This is a reimplementation of the Stockfish win rate function in C++. The original
    version can be found here: https://github.com/official-stockfish/Stockfish/blob/master/src/uci.cpp#L202
    (Stockfish release 15.1)

    Args:
        score (float): The Stockfish score in centipawns.
        ply (int): The number of plies.

    Returns:
        int: The win rate in per mille units rounded to the nearest value.
    """

    # The model only captures up to 240 plies, so limit the input and then rescale
    m = min(240, ply) / 64.0

    # The coefficients of a third-order polynomial fit is based on the fishtest data
    # for two parameters that need to transform eval to the argument of a logistic
    # function.
    as_list = [-0.58270499, 2.68512549, 15.24638015, 344.49745382]
    bs_list = [-2.65734562, 15.96509799, -20.69040836, 73.61029937]

    # Enforce that NormalizeToPawnValue corresponds to a 50% win rate at ply 64
    assert NormalizeToPawnValue == int(as_list[0] + as_list[1] + as_list[2] + as_list[3])

    a = (((as_list[0] * m + as_list[1]) * m + as_list[2]) * m) + as_list[3]
    b = (((bs_list[0] * m + bs_list[1]) * m + bs_list[2]) * m) + bs_list[3]

    # Transform the eval to centipawns with limited range
    x = np.clip(score, -4000.0, 4000.0)

    # Return the win rate in per mille units rounded to the nearest value
    return int(0.5 + 1000 / (1 + np.exp((a - x) / b)))


def stockfish_cp_to_wdl(score: int, ply: int) -> Tuple[int, int, int]:
    """
    Converts a Stockfish score from centipawns to WDL.

    Args:
        score (float): The Stockfish score in centipawns.
        ply (int): The number of plies.

    Returns:
        Tuple[int, int, int]: The WDL score.
    """
    # Undoing the Stockfish normalization
    if abs(score) <= NORMALIZED_SCORE_MAX:
        score = (score * NormalizeToPawnValue) / 100

    wdl_w = win_rate(score, ply)
    wdl_l = win_rate(-score, ply)
    wdl_d = 1000 - wdl_w - wdl_l

    return wdl_w, wdl_d, wdl_l


def stockfish_cp_to_leela_q(score: int, ply: int) -> float:
    win, draw, loss = stockfish_cp_to_wdl(score, ply)

    # Normalize to 0-1
    win, draw, loss = win / 1000, draw / 1000, loss / 1000

    # Convert to Leela Q
    return win - loss


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

    # Convert the scores
    def extract_ply(fen: str) -> int:
        _, turn, _, _, _, fullmove = fen.split(" ")
        blacksturn = turn == "b"
        return (int(fullmove) - 1) * 2 + blacksturn

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
