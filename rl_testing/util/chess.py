import datetime
import io
from pathlib import Path
from typing import Union

import chess
import chess.svg
import imgkit
import matplotlib
import matplotlib.image as mimage
import matplotlib.pyplot as plt

AGGRESSIVE_VALIDATION = True


def remove_pawns(board: chess.Board) -> Union[chess.Board, str]:
    piece_map = board.piece_map()
    positions = list(piece_map.keys())
    for position in positions:
        if piece_map[position].symbol() in ["P", "p"]:
            del piece_map[position]

    new_board = chess.Board()
    new_board.set_piece_map(piece_map)
    new_board.turn = board.turn

    if not is_really_valid(new_board):
        return "failed"

    return new_board


def rotate_90_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return chess.flip_vertical(chess.flip_diagonal(board))


def rotate_180_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return rotate_90_clockwise(rotate_90_clockwise(board))


def rotate_270_clockwise(board: chess.Bitboard) -> chess.Bitboard:
    return rotate_180_clockwise(rotate_90_clockwise(board))


def is_really_valid(board: chess.Board) -> bool:
    check_squares = list(board.checkers())
    if len(check_squares) > 2:
        return False

    if AGGRESSIVE_VALIDATION and len(check_squares) == 2:
        return False

    elif len(check_squares) == 2:
        if (
            board.piece_at(check_squares[0]).piece_type
            == board.piece_at(check_squares[1]).piece_type
        ):
            return False
        symbol1 = board.piece_at(check_squares[0]).symbol().lower()
        symbol2 = board.piece_at(check_squares[1]).symbol().lower()
        symbols = "".join([symbol1, symbol2])

        if "p" in symbols:
            return False

    return board.is_valid()


def plot_board(
    board: chess.Board,
    title: str = "",
    fen: str = "",
    fontsize: int = 22,
    save: bool = True,
    show: bool = False,
    save_path: Union[str, Path] = "",
) -> None:
    # Get the XML representation of an SVG image of the board
    svg = chess.svg.board(board, size=800)

    # Convert the XML representation into an SVG image
    transformed = imgkit.from_string(svg, output_path=False, options={"format": "png"})

    # Trick matplotlib into thinking that transformed is actually a file
    with io.BytesIO(transformed) as image_bytes:

        # Read the image as if it was a file
        im = mimage.imread(image_bytes)

    # Plot the image
    plt.imshow(im)

    # Change the font size
    font = {"size": fontsize}
    matplotlib.rc("font", **font)

    # Make the axes invisible
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    if fen:
        font["size"] = fontsize - 4
        plt.xlabel(fen, fontdict=font)

    # x_axis.set_visible(False)
    plt.xticks([])

    plt.title(title, pad=10)

    if save:
        if save_path == "":
            time_now = str(datetime.datetime.now())
            time_now = time_now.replace(" ", "_")
            save_path = Path(f"board_{time_now}.png")

        save_path = Path(save_path)
        save_path.absolute().parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=400)

    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    board = chess.Board("6rk/1bp1q2p/p4p1Q/1p1ppPP1/3bP3/2NP3R/PPP3r1/1K1R4 b - - 0 24")
    board = remove_pawns(board)

    for i in range(4):
        print(board.fen())
        board = board.transform(rotate_90_clockwise)

    print(board.transform(chess.flip_vertical).fen())
