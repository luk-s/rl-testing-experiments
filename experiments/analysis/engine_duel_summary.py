from pathlib import Path
from typing import Dict
import chess.pgn
import argparse

RESULT_FOLDER_PATH = Path(__file__).absolute().parent.parent / "results/engine_duel"


def summarize_results(file_path: str) -> Dict[str, float]:
    results: Dict[str, float] = {}

    # Open the .pgn file
    with open(file_path, "r") as f:
        # Iterate over the headers of the games
        headers = chess.pgn.read_headers(f)

        while headers is not None:
            # Get the result of the game
            result = headers["Result"]

            # Read the players
            white_player = headers["White"]
            black_player = headers["Black"]

            # Update the results
            if result == "1-0":
                results[white_player] = results.get(white_player, 0) + 1
            elif result == "0-1":
                results[black_player] = results.get(black_player, 0) + 1
            elif result == "1/2-1/2":
                results[white_player] = results.get(white_player, 0) + 0.5
                results[black_player] = results.get(black_player, 0) + 0.5

            # Read the next game
            headers = chess.pgn.read_headers(f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ##################################
    #           CONFIG START         #
    ##################################
    # fmt: off
    parser.add_argument("--result_file_name",      type=str)  # noqa
    # fmt: on
    ##################################
    #           CONFIG END           #
    ##################################

    args = parser.parse_args()

    # Get the path to the result file
    result_file_path = RESULT_FOLDER_PATH / args.result_file_name

    # Summarize the results
    results = summarize_results(result_file_path)

    # Print the results
    for player, score in results.items():
        print(f"{player}: {score}")
