from typing import Generator, List, Optional

import chess.pgn


def get_fens_from_pgn(
    pgn_path: str, max_num: Optional[int] = None, use_dict: bool = True
) -> Generator[str, None, None]:
    """Get a list of FENs from a PGN file.

    Args:
        pgn_path (str): Path to the PGN file.
        max_num (int, optional): Maximum number of FENs to extract. Defaults to None.
        use_dict (bool, optional): Whether to use a dictionary to store the FENs. Defaults to True.

    Yields:
        Generator[str, None, None]: Generator of FENs.
    """
    fen_cache = set()
    with open(pgn_path) as pgn:
        while max_num is None or len(fen_cache) < max_num:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                if use_dict and fen not in fen_cache:
                    fen_cache.add(fen)
                    yield fen

                elif not use_dict:
                    yield fen


def convert_pgn_to_fens(pgn_path: str, output_path: str, max_num: Optional[int] = None):
    """Convert a PGN file to a list of FENs.

    Args:
        pgn_path (str): Path to the PGN file.
        output_path (str): Path to the output file.
        max_num (int, optional): Maximum number of FENs to extract. Defaults to None.
    """
    with open(output_path, "w") as f:
        for index, fen in enumerate(get_fens_from_pgn(pgn_path, max_num=max_num)):
            if index % 1000 == 0:
                print(f"Processed {index} FENs")
            f.write(f"{fen}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn_path", type=str, help="Path to the PGN file.")
    parser.add_argument("--output_path", type=str, help="Path to the output file.")
    parser.add_argument(
        "--max_num",
        type=int,
        default=None,
        help="Maximum number of FENs to extract. Defaults to None.",
    )
    args = parser.parse_args()

    convert_pgn_to_fens(args.pgn_path, args.output_path, max_num=args.max_num)
